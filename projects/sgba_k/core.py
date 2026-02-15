"""SGBA 攻击实现模块"""

from dataclasses import dataclass
from typing import override

from torch.optim import AdamW, lr_scheduler as lrs

from main.arch import BaseVFLArch
from main.callback import OPT_TYPE, VFLCallback
from models.fcn import FCN
from projects.vfl.config import AppConfig, createLRS
from utils.common import F, copy, math, nn, tc
from utils.config import rng
from utils.define import StepVars
from utils.misc import accuracy, segment

# TODO: attackBtmIns getRecon


def sublist[T](lThis: list[T], lIndices: list[int]) -> list[T]:
	return [lThis[i] for i in lIndices]


@dataclass
class MethodArgs:
	"""SGBA 攻击方法参数配置类

	定义 SGBA 攻击过程中所需的各种参数配置，
	包括训练和验证时的混合比例、损失缩放比例等关键参数。

	Args:
		fRecLr: 生成网络的学习率
		fTrainAlpha: 训练时生成样本的混合比例
		fValAlpha: 验证时生成样本的混合比例
		lLossScales: 生成网络的 Loss 缩放比例
		lGradScales: 投毒目标的 Gradient 缩放比例
	"""

	fRecLr: float
	"""生成网络的学习率"""
	fSurLr: float
	"""代理网络的学习率"""
	fTrainAlpha: float
	"""训练时生成样本的混合比例"""
	fValAlpha: float
	"""验证时生成样本的混合比例"""
	lLossScales: tuple[float, float]
	"""生成网络的 Loss 缩放比例"""
	lGradScales: tuple[float, float]
	"""投毒目标的 Gradient 缩放比例"""


class SGBACb(VFLCallback):
	"""SGBA 攻击回调类，实现样本生成式后门攻击

	该回调类在垂直联邦学习训练过程中注入后门触发器，通过生成网络重构底层模型输出的嵌入表示，
	并在特定样本上应用混合比例来实现隐蔽的后门攻击目标。
	"""

	def __init__(self, args: MethodArgs, config: AppConfig) -> None:
		"""初始化实例。

		Args:
			args: SGBA 攻击方法参数配置
			config: 应用配置对象
		"""
		self.args = args
		self.cfg = config

		self.lRPs = [0, 1, 2]
		self.lSPs = [3]
		self.lAPs = self.lRPs + self.lSPs

		lPartyDims = self.cfg.model.lPartyDims
		nDim = sum(sublist(lPartyDims, self.lRPs))
		self.zRecNet = FCN([nDim, int(nDim * 0.75), nDim], False)  # ! 可变
		"""攻击者用于生成后门触发器的网络"""

		self.zSurNet = FCN([sum(sublist(lPartyDims, self.lAPs)), 256, 10])
		self.criSur = nn.CrossEntropyLoss()

	@override
	def onInitModule(self, m: BaseVFLArch) -> None:
		m.add_module('zRecNet', self.zRecNet)
		m.add_module('zSurNet', self.zSurNet)
		m.add_module('criSur', self.criSur)

	@override
	def onConfigOptims(self) -> OPT_TYPE:
		optRec = AdamW(self.zRecNet.parameters(), self.args.fRecLr)
		optSur = AdamW(self.zSurNet.parameters(), self.args.fSurLr)

		lrsRec = lrs.ConstantLR(optRec, 0.8, 5)
		# lrsRec = createLRS(optRec)
		lrsSur = createLRS(optSur)
		return [optRec, optSur], [lrsRec, lrsSur]

	def getRecon(
		self, lEmbeds: list[tc.Tensor], alpha: float = 1.0
	) -> tuple[list[tc.Tensor], tc.Tensor]:
		"""根据输入嵌入生成重构输出。

		Args:
			lEmbeds: 输入嵌入列表，每个元素为一个节点的嵌入表示
			alpha: 重构输出的混合比例，默认为 `1.0`

		Returns:
			重构输出列表，每个元素为一个节点的重构表示；
			重构损失值，为所有节点重构损失的总和。
		"""
		# 将嵌入拼接，并记录拼接前的维度
		tEmbed = tc.cat(lEmbeds, 1)
		lDims = [t.size(1) for t in lEmbeds]

		# 生成重构输出，并切分回原维度
		tRecOut = self.zRecNet(tEmbed)
		lRecOut = list(tc.split(tRecOut, lDims, 1))

		# 根据 alpha 混合重构输出，并切分回原维度
		tRecon = tRecOut * alpha + tEmbed * (1 - alpha)
		lRecons = list(tc.split(tRecon, lDims, 1))

		# 针对每个节点计算重构损失
		# vLoss = F.mse_loss(tEmbed, tRecon, reduction='sum') / len(v.indices)  # MSE Loss
		lLoss = [tc.norm(a - b, 2, 1).mean() for a, b in zip(lRecOut, lEmbeds, strict=True)]  # L2 Loss
		vLoss = tc.sum(tc.stack(lLoss))

		return lRecons, vLoss

	# ============
	# 训练阶段
	# ============

	@override
	def onTrainBtmIns(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch > 0:
			self.switch(m, v)

	def switch(self, m: BaseVFLArch, v: StepVars) -> None:
		"""用于恶意关联的样本切换"""
		# 获取投毒目的样本（属于目标类样本）的掩码
		self.tDstMask = tc.isin(v.indices, m.ns.tDstIdxs)
		# 获取受害类样本的掩码
		self.tVicMask = tc.isin(v.indices, m.ns.tVicIdxs)

		# 批次内部样本切换
		if self.tDstMask.any() and self.tVicMask.any():
			# 获取投毒目的样本（属于目标类样本）的位置（索引的索引）
			tDstPos = tc.nonzero(self.tDstMask, as_tuple=True)[0]
			# 获取受害类样本的位置（索引的索引）
			tVicPos = tc.nonzero(self.tVicMask, as_tuple=True)[0]

			# 有放回抽取
			tSelect = tc.randint(0, tVicPos.size(0), (tDstPos.size(0),), device=tVicPos.device)
			# 无放回抽取，但是来源可能少于目标
			# tSelect = tc.randperm(tVicPos.size(0), device=tVicPos.device)[: tDstPos.size(0)]

			# 批量样本切换
			for i in self.lRPs:
				v.lBtmIns[i][tDstPos] = v.lBtmIns[i][tVicPos[tSelect]]

	@override
	def onTrainBtmOut(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch > 0:
			self.poison(m, v)

	def poison(self, m: BaseVFLArch, v: StepVars) -> None:
		"""用于嵌入隐蔽性的生成式投毒"""
		# ! 不使用 detach()，让底层模型也更新，降低触发器生成网络的训练难度
		lEmbeds = sublist(v.lBtmOut, self.lRPs)
		_, self.vReconLoss = self.getRecon(lEmbeds)
		m.logDict({'loss/ReconTrain': self.vReconLoss})

		# # 投毒操作

		for i in self.lRPs:
			v.lBtmOut[i] = v.lBtmOut[i].clone()  # 避免 In-place 操作错误

		# ! 对于目标类（目的样本特征改成来源样本特征，建立目标类与来源样本的联系）
		if self.tDstMask.any():  # 如果找到投毒目的样本
			# ! 使用 detach() 避免投毒样本的梯度传播至底层模型，产生意外影响
			lEmbeds = [t[self.tDstMask].detach() for t in sublist(v.lBtmOut, self.lRPs)]
			lRecons, _ = self.getRecon(lEmbeds, self.args.fTrainAlpha)
			for i in self.lRPs:
				v.lBtmOut[i][self.tDstMask] = lRecons[i]

		# 受害类样本
		# if len(self.aVicPos) > 0:  #! 对受害类样本添加不完全触发器不应该触发后门
		# 	# 选择投毒来源样本（属于受害类样本）的位置
		# 	aSrcPos = rng().choice(self.aVicPos, math.ceil(len(self.aVicPos) * 0.1), False)
		# 	self.aSrcPos2 = aSrcPos
		# 	lEmbeds = [v.lBtmOut[i][aSrcPos].detach() for i in range(self.nAP)]
		# 	lRecons, _ = self.getRecon(lEmbeds, self.args.fTrainAlpha)

		# 	# 只有单个 AP 进行投毒
		# 	tIdxsI = tc.randint(self.nAP, (len(aSrcPos),))
		# 	for i in range(self.nAP):
		# 		tBoolSel = tIdxsI == i
		# 		if len(tBoolSel) > 0:
		# 			v.lBtmOut[i][aSrcPos][tBoolSel] = lRecons[i][tBoolSel]

	@override
	def onTrainLoss(self, m: BaseVFLArch, v: StepVars) -> None:
		optRec = m.getOptim(self.iOpt)
		optRec.zero_grad()
		optSur = m.getOptim(self.iOpt + 1)
		optSur.zero_grad()

	@override
	def onTrainBtmOutGrad(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch > 0:
			self.doSGBA(m, v)
			self.doSurUpdate(m, v)
			if self.tVicMask.any():
				self.doSurPoison(m, v)

	def doSGBA(self, m: BaseVFLArch, v: StepVars) -> None:
		"""SGBA 攻击"""
		p = segment(m.current_epoch, (15, 20), self.args.lLossScales)
		m.manual_backward(self.vReconLoss * p, retain_graph=True)

		if self.tDstMask.any():  # 如果找到投毒目标
			value = segment(m.current_epoch, (15, 20), self.args.lGradScales)
			for i in self.lRPs:
				v.lTopInsGrad[i][self.tDstMask] *= value

		# if len(self.aSrcPos2) > 0:
		# 	for i in range(self.nAP):
		# 		value = segment(m.current_epoch, (15, 20), (8.0, 4.0))
		# 		v.lTopInsGrad[i][self.aSrcPos2] *= value

	def doSurUpdate(self, m: BaseVFLArch, v: StepVars) -> None:
		"""进行代理模型训练。"""
		# 获取受控节点上传的嵌入，拼接
		tSurIns = tc.cat(sublist(v.lBtmOut, self.lAPs), dim=1)
		# 从计算图中分离
		tSurIns = tSurIns.detach().requires_grad_()
		# 输入代理模型，得到 Logits
		tSurOut = self.zSurNet(tSurIns)

		# 获取推理标签
		tInferLabels = m.ns.tInfers[tc.searchsorted(m.ns.tIDs, v.indices)]
		# 计算代理模型的分类损失
		vModelLoss = self.criSur(tSurOut, tInferLabels)  # 交叉熵损失

		# 计算代理模型产生的关于嵌入的梯度（代理模型参数的梯度未累积）
		[tSurGrad] = tc.autograd.grad(vModelLoss, [tSurIns], create_graph=True)
		# 获取受控节点接收的关于嵌入的梯度，拼接
		tRawGrad = tc.cat(sublist(v.lBtmOutGrad, self.lAPs), dim=1)
		# 计算代理模型的梯度损失
		vGradLoss = tc.norm(tSurGrad - tRawGrad, p=2)  # L2 损失

		# 计算代理模型的总损失
		tSurLoss = 5 * vModelLoss + 50 * vGradLoss  # ! 调整权重
		# 执行反向传播，更新代理模型参数
		m.manual_backward(tSurLoss, retain_graph=True)
		m.logDict({'lossSur/Sur': tSurLoss, 'lossSur/Grad': vGradLoss, 'lossSur/Model': vModelLoss})

	def doSurPoison(self, m: BaseVFLArch, v: StepVars) -> None:
		"""执行代理模型投毒操作。"""
		# 初步筛选受害类样本
		lVicEmbedsRP = [v.lBtmOut[i][self.tVicMask] for i in self.lRPs]
		lVicEmbedsSP = [v.lBtmOut[i][self.tVicMask] for i in self.lSPs]
		tVicIndices = v.indices[self.tVicMask]

		def _doSample(ratio: float = 0.1, single: bool = False) -> tuple[tc.Tensor, tc.Tensor]:
			"""采样并生成恶意嵌入。

			Args:
				ratio: 采样比例，默认为 `0.1`
				single: 是否仅对单个攻击节点进行投毒，默认为 `False`

			Returns:
				包含恶意嵌入和推理标签的元组
			"""
			with tc.no_grad():
				# 随机选择样本
				nSamples = math.ceil(ratio * len(tVicIndices))
				tPerm = tc.randperm(len(tVicIndices), device=m.device)[:nSamples]

				# 获取受控节点的嵌入
				lSampleRP = [t[tPerm] for t in lVicEmbedsRP]
				lSampleSP = [t[tPerm] for t in lVicEmbedsSP]
				# tClean = tc.cat(lSampleRP + lSampleSP, dim=1)  #: 拼接得到的良性嵌入

				# 获取推理标签
				tSampleLabels = m.ns.tInfers[tc.searchsorted(m.ns.tIDs, tVicIndices[tPerm])]

			# 只有攻击节点生成重构嵌入，辅助节点的嵌入不变
			lRecons, _ = self.getRecon(lSampleRP, self.args.fTrainAlpha)

			if single:
				# 对于每个样本，随机选择单个攻击节点进行投毒
				tWhichAP = tc.randint(high=len(lSampleRP), size=(len(lSampleRP[0]),))
				for i in range(len(lSampleRP)):
					lSampleRP[i][tWhichAP == i] = lRecons[i][tWhichAP == i]
			else:
				# 全部攻击节点进行投毒
				lSampleRP = lRecons

			tPoison = tc.cat(lSampleRP + lSampleSP, dim=1)  #: 拼接得到的恶意嵌入
			return tPoison, tSampleLabels

		# # 向受害类投毒（来源样本标签改成目的样本标签，建立目标类与来源样本的联系）
		tPoison, tInferLabels = _doSample()
		tTgtLabels = tc.full_like(tInferLabels, m.ns.iTgtLabel)  #: 目标类标签

		# 从计算图中分离
		tSurIns = tPoison.detach().requires_grad_()
		# 输入代理模型，得到 Logits
		tSurOut = self.zSurNet(tSurIns)
		# 计算代理模型的分类损失（标签已修改）
		vCELoss = self.criSur(tSurOut, tTgtLabels)  # 交叉熵损失
		# 计算代理模型的熵值损失
		vEntropy = -(F.softmax(tSurOut, dim=1) * F.log_softmax(tSurOut, dim=1)).sum(dim=1).mean()
		# 计算代理模型产生的关于嵌入的梯度（代理模型参数的梯度未累积）
		[tGrad] = tc.autograd.grad(1 * vCELoss - 5 * vEntropy, [tSurIns])  # ! 调整权重
		# 执行反向传播，更新生成器参数（代理模型参数未更新）
		m.manual_backward(tPoison, tGrad, retain_graph=True)
		m.logDict({'loss/SurVic': vCELoss, 'loss/SurVicEntropy': vEntropy})

		# # 向受害类投毒（单节点随机投毒）
		tPoison, tInferLabels = _doSample(single=True)

		# 从计算图中分离
		tSurIns = tPoison.detach().requires_grad_()
		# 输入代理模型，得到 Logits
		tSurOut = self.zSurNet(tSurIns)
		# 计算代理模型的分类损失（标签未修改）
		vCELoss = self.criSur(tSurOut, tInferLabels)  # 交叉熵损失
		# 计算代理模型产生的关于嵌入的梯度（代理模型参数的梯度未累积）
		[tGrad] = tc.autograd.grad(15 * vCELoss, tSurIns)  # ! 调整权重
		# 执行反向传播，更新生成器参数（代理模型参数未更新）
		m.manual_backward(tPoison, tGrad, retain_graph=True)
		m.logDict({'loss/SurVicPart': vCELoss})

	@override
	def onTrainOptimStep(self, m: BaseVFLArch, v: StepVars) -> None:
		optRec = m.getOptim(self.iOpt)
		optRec.step()
		optSur = m.getOptim(self.iOpt + 1)
		optSur.step()

	# ============
	# 验证阶段
	# ============

	@override
	def onValStepVars(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		d['Attack'] = copy(d['Origin'])

	@override
	def onValBtmOut(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		v = d['Attack']

		lEmbeds = sublist(v.lBtmOut, self.lRPs)
		lRecons, vReconLoss = self.getRecon(lEmbeds, self.args.fValAlpha)
		for i in self.lRPs:
			v.lBtmOut[i] = lRecons[i]

		m.logDict({'loss/ReconVal': vReconLoss})

	@override
	def onValLoss(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		for k, v in d.items():
			probs = F.softmax(v.zTopOut, 1)
			entropy = tc.distributions.Categorical(probs).entropy().mean().item()
			m.logDict({f'entropy/Val{k}': entropy})

			if m.current_epoch > 0:
				self.logSur(m, v, k)
				self.logSGBA(m, v, k)

	def logSur(self, m: BaseVFLArch, v: StepVars, k: str) -> None:
		tSurIns = tc.cat(sublist(v.lTopIns, self.lAPs), 1)
		tSurOut = self.zSurNet(tSurIns)

		[acc1, acc3] = accuracy(tSurOut, v.labels, (1, 3))
		loss = self.criSur(tSurOut, v.labels)

		name = f'Val{v.iLoaderIdx}_{k}'
		m.logDict({f'lossSur/{name}': loss, f'accSur/{name}/Top1': acc1, f'accSur/{name}/Top3': acc3})

		tTopPred = v.zTopOut.argmax(1)
		[acc1, acc3] = accuracy(tSurOut, tTopPred, (1, 3))

		name = f'Val{v.iLoaderIdx}_{k}'
		m.logDict({f'accGod/{name}/Top1': acc1, f'accGod/{name}/Top3': acc3})

		mask = v.labels != m.ns.iTgtLabel
		if mask.any():
			tTopOut = tSurOut[mask]

			tLabels = v.labels[mask]
			tTgtLabels = tc.full_like(tLabels, m.ns.iTgtLabel)

			[acc1, acc3] = accuracy(tTopOut, tTgtLabels, (1, 3))
			loss = m.criterion(tTopOut, tTgtLabels)

			name = f'Val{v.iLoaderIdx}_{k}'
			m.logDict(
				{f'lossSurTgt/{name}': loss, f'accSurTgt/{name}/Top1': acc1, f'accSurTgt/{name}/Top3': acc3}
			)

	def logSGBA(self, m: BaseVFLArch, v: StepVars, k: str) -> None:
		mask = v.labels != m.ns.iTgtLabel
		if mask.any():
			tTopOut = v.zTopOut[mask]

			tLabels = v.labels[mask]
			tTgtLabels = tc.full_like(tLabels, m.ns.iTgtLabel)

			[acc1, acc3] = accuracy(tTopOut, tTgtLabels, (1, 3))
			loss = m.criterion(tTopOut, tTgtLabels)

			name = f'Val{v.iLoaderIdx}_{k}'
			m.logDict({f'lossTgt/{name}': loss, f'accTgt/{name}/Top1': acc1, f'accTgt/{name}/Top3': acc3})
