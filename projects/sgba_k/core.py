"""SGBA 攻击实现模块"""

from dataclasses import dataclass
from typing import override

from torch.optim import AdamW

from main.arch import BaseVFLArch
from main.callback import OPT_TYPE, VFLCallback
from models.fcn import FCN
from projects.vfl.config import AppConfig, createLRS
from utils.common import F, copy, math, nn, tc
from utils.define import StepVars
from utils.misc import accuracy, checkModel, checkModelGrad, segment


def sublist[T](lThis: list[T], lIndices: list[int]) -> list[T]:
	"""根据索引列表提取原始列表中的元素。

	Args:
		lThis: 原始列表
		lIndices: 索引列表，包含待提取元素的索引

	Returns:
		由指定索引元素构成的新列表
	"""
	return [lThis[i] for i in lIndices]


def calcVecLoss(a: tc.Tensor, b: tc.Tensor, method: str = 'MSE') -> tc.Tensor:
	"""计算批次损失，支持多种损失函数。

	Args:
		a: 预测向量，形状为 `(nBatch, nDim)`
		b: 目标向量，形状为 `(nBatch, nDim)`
		method: 损失函数，可选 `'SSE'`、`'SSE/N'`、`'MSE'`、`'L2.ND'`、`'L2.D/N'`

	Returns:
		批次损失

	Raises:
		ValueError: 损失函数未知
	"""
	if method == 'SSE':
		loss = F.mse_loss(a, b, reduction='sum')
	elif method == 'SSE/N':  # 样本 SSE，批量平均
		loss = F.mse_loss(a, b, reduction='sum') / len(a)
	elif method == 'MSE':
		loss = F.mse_loss(a, b, reduction='mean')
	elif method == 'Huber/N':  # 样本 Huber 损失，批量平均
		loss = F.huber_loss(a, b, reduction='sum', delta=1.0) / len(a)
	elif method == 'Huber':
		loss = F.huber_loss(a, b, reduction='mean', delta=1.0)
	elif method == 'L2/N':  # 样本 L2 距离，批量平均
		loss = tc.norm(a - b, p=2, dim=1).mean()
	elif method == 'L2':
		loss = tc.norm(a - b, p=2)
	else:
		msg = f'损失函数未知：{method}！'
		raise ValueError(msg)
	return loss


def calcFidelity(tInputLogits: tc.Tensor, tRealLogits: tc.Tensor) -> tuple[float, float]:
	"""计算保真度指标。

	Args:
			tInputLogits: 输入 Logits，形状 (batch_size, num_classes)
			tRealLogits: 真实 Logits，形状 (batch_size, num_classes)

	Returns:
			consistency_rate: 决策一致率 (标量，范围 0.0 ~ 1.0)
			kl_divergence: KL 散度均值 (标量)
	"""
	# 1. 计算决策一致率 (使用 Logits 求 argmax)
	tRealPreds = tc.argmax(tRealLogits, dim=1)
	tInputPreds = tc.argmax(tInputLogits, dim=1)
	rConsistency: float = (tRealPreds == tInputPreds).float().mean().item()

	# 2. 计算 KL 散度
	# 将 Logits 转换为 Probs
	tRealProbs = F.softmax(tRealLogits, dim=1)
	tInputProbs = F.softmax(tInputLogits, dim=1)
	# 计算代理模型的 LogProbs
	tInputLogProbs = tc.log(tInputProbs + 1e-8)
	# 计算 KL 散度
	fKLDiv: float = F.kl_div(tInputLogProbs, tRealProbs, reduction='batchmean').item()

	return rConsistency, fKLDiv


@dataclass
class MethodArgs:
	"""SGBA 攻击方法参数配置类

	Args:
		fRecLr: 生成网络的学习率
		fSurLr: 代理网络的学习率
		fTrainAlpha: 训练时生成样本的混合比例
		fValAlpha: 验证时生成样本的混合比例
		lLossScales: 生成网络的损失缩放比例
		lGradScales: 投毒目标的梯度缩放比例
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
	"""生成网络的损失缩放比例"""
	lGradScales: tuple[float, float]
	"""投毒目标的梯度缩放比例"""
	milestones: list[int]
	gamma: float
	fSurLambda1: float
	"""代理模型训练的分类损失系数"""
	fSurLambda2: float
	"""代理模型训练的梯度损失系数"""
	lVicScales: tuple[float, float]
	"""多节点投毒的分类损失系数"""
	lVicEntropyScales: tuple[float, float]
	"""多节点投毒的熵值损失系数"""
	lVicPartScales: tuple[float, float]
	"""单节点投毒的分类损失系数"""


class SGBACb(VFLCallback):
	"""SGBA 攻击回调类"""

	def __init__(self, args: MethodArgs, config: AppConfig) -> None:
		"""初始化实例。

		Args:
			args: SGBA 攻击方法参数配置
			config: 应用配置
		"""
		self.args = args
		"""SGBA 攻击方法参数配置"""
		self.cfg = config
		"""应用配置"""

		self.lRPs = [1, 2, 5]
		"""攻击节点的索引列表"""
		self.lSPs = [6]
		"""辅助节点的索引列表"""
		self.lAPs = self.lRPs + self.lSPs
		"""全部受控节点的索引列表"""

		lPartyDims = self.cfg.model.lPartyDims
		nDim = sum(sublist(lPartyDims, self.lRPs))
		self.zRecNet = FCN(lDims=[nDim, int(nDim * 0.8), int(nDim * 0.8), nDim], hasBN=False)  # ! 可变
		"""攻击者用于生成后门触发器的网络"""

		self.zSurNet = FCN(lDims=[sum(sublist(lPartyDims, self.lAPs)), 256, 10])
		"""代理模型"""
		self.criSur = nn.CrossEntropyLoss()
		"""代理模型的损失函数"""

	@override
	def onInitModule(self, m: BaseVFLArch) -> None:
		m.add_module('zRecNet', self.zRecNet)
		m.add_module('zSurNet', self.zSurNet)
		m.add_module('criSur', self.criSur)

	@override
	def onConfigOptims(self) -> OPT_TYPE:
		optRec = AdamW(self.zRecNet.parameters(), lr=self.args.fRecLr)
		optSur = AdamW(self.zSurNet.parameters(), lr=self.args.fSurLr)

		lrsRec = createLRS(optRec, milestones=[5, 10, 20], gamma=0.8)
		lrsSur = createLRS(optSur, milestones=[5, 10, 20], gamma=0.8)

		return [optRec, optSur], [lrsRec, lrsSur]

	def getRecon(
		self, lEmbeds: list[tc.Tensor], alpha: float = 1.0
	) -> tuple[list[tc.Tensor], tc.Tensor, tc.Tensor]:
		"""生成重构嵌入。

		Args:
			lEmbeds: 原始嵌入列表，每个元素为一个节点的原始嵌入
			alpha: 生成样本的混合比例，默认为 `1.0`

		Returns:
			重构嵌入列表，每个元素为一个节点的重构嵌入；
			重构损失值，为所有节点重构损失的总和。
		"""
		# 将嵌入拼接，并记录拼接前的维度
		tEmbed = tc.cat(lEmbeds, dim=1)
		lDims = [t.size(dim=1) for t in lEmbeds]

		# 生成重构输出，并切分回原维度
		tRecOut = self.zRecNet(tEmbed)
		lRecOut = list(tc.split(tRecOut, lDims, dim=1))

		# 混合重构输出，并切分回原维度
		tRecon = alpha * tRecOut + (1 - alpha) * tEmbed
		lRecons = list(tc.split(tRecon, lDims, dim=1))

		# 针对每个节点计算重构损失
		lLoss = [calcVecLoss(a, b, 'Huber/N') for a, b in zip(lRecOut, lEmbeds, strict=True)]
		vLoss1 = tc.mean(tc.stack(lLoss))
		vLoss2 = 10 * tc.std(tc.stack(lLoss))

		return lRecons, vLoss1, vLoss2

	# ============
	# 训练阶段
	# ============

	@override
	def onTrainBtmIns(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch > 0:
			self.switch(m, v)

	def switch(self, m: BaseVFLArch, v: StepVars) -> None:
		"""执行样本切换。"""
		# 获取投毒目的样本（属于目标类样本）的掩码
		self.tDstMask = tc.isin(v.indices, m.ns.tDstIds)
		# 获取受害类样本的掩码
		self.tVicMask = tc.isin(v.indices, m.ns.tVicIds)

		# # 在批次内部执行样本切换
		if self.tDstMask.any() and self.tVicMask.any():
			# 获取投毒目的样本（属于目标类样本）的位置（索引的索引）
			[tDstPos] = tc.nonzero(self.tDstMask, as_tuple=True)
			# 获取受害类样本的位置（索引的索引）
			[tVicPos] = tc.nonzero(self.tVicMask, as_tuple=True)

			# 方案一：有放回抽取
			tSelect = tc.randint(high=tVicPos.size(dim=0), size=(tDstPos.size(dim=0),), device=m.device)
			# 方案二：无放回抽取，但是来源样本可能少于目的样本
			# tSelect = tc.randperm(tVicPos.size(dim=0), device=m.device)[: tDstPos.size(dim=0)]

			# 批量样本切换
			for idx in self.lRPs:
				v.lBtmIns[idx][tDstPos] = v.lBtmIns[idx][tVicPos[tSelect]]

	@override
	def onTrainBtmOut(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch > 0:
			self.poison(m, v)

	def poison(self, m: BaseVFLArch, v: StepVars) -> None:
		"""执行生成式投毒。"""
		# 计算重构损失
		# ! 不使用 detach()，让底层模型也更新，降低生成网络的训练难度
		lEmbeds = sublist(v.lBtmOut, self.lRPs)
		_, vLoss1, vLoss2 = self.getRecon(lEmbeds)
		self.vReconLoss = vLoss1 + vLoss2
		m.logDict(
			{
				'loss/ReconTrain': self.vReconLoss,
				'loss/ReconTrain1': vLoss1,
				'loss/ReconTrain2': vLoss2,
			}
		)

		for idx in self.lRPs:
			v.lBtmOut[idx] = v.lBtmOut[idx].clone()  # 避免 In-place 操作错误

		# # 向目标类投毒（目的样本特征改成来源样本特征，建立目标类与来源样本的联系）
		if self.tDstMask.any():
			# ! 使用 detach() 避免投毒样本的梯度传播至底层模型，产生意外影响
			lSubEmbeds = [t[self.tDstMask].detach() for t in sublist(v.lBtmOut, self.lRPs)]
			lSubRecons, _, _ = self.getRecon(lSubEmbeds, self.args.fTrainAlpha)
			for i, idx in enumerate(self.lRPs):
				v.lBtmOut[idx][self.tDstMask] = lSubRecons[i]

		# # 向受害类投毒（单节点随机投毒）
		# if len(self.aVicPos) > 0:  # ! 对受害类样本添加不完全触发器不应该触发后门
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
		"""执行 SGBA 攻击。"""
		# 重构损失的缩放与反向传播
		p = segment(m.current_epoch, ins=(10, 30), out=self.args.lLossScales)
		m.logDict({'value/LossScale': p})
		m.manual_backward(self.vReconLoss * p, retain_graph=True)

		# 目的样本梯度的缩放
		value = segment(m.current_epoch, ins=(10, 30), out=self.args.lGradScales)
		m.logDict({'value/GradScale': value})
		if self.tDstMask.any():
			for idx in self.lRPs:
				v.lTopInsGrad[idx][self.tDstMask] *= value

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
		vGradLoss = calcVecLoss(tSurGrad, tRawGrad, method='Huber/N')

		# 计算代理模型的总损失
		tSurLoss = self.args.fSurLambda1 * vModelLoss + self.args.fSurLambda2 * vGradLoss
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
			lRecons, _, _ = self.getRecon(lSampleRP, self.args.fTrainAlpha)

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

		v1 = segment(m.current_epoch, ins=(15, 25), out=self.args.lVicScales)
		v2 = segment(m.current_epoch, ins=(15, 25), out=self.args.lVicEntropyScales)
		m.logDict({'value/SurVic': v1, 'value/SurVicEntropy': v2})
		[tGrad] = tc.autograd.grad(v1 * vCELoss - v2 * vEntropy, [tSurIns])
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

		v1 = segment(m.current_epoch, ins=(15, 25), out=self.args.lVicPartScales)
		m.logDict({'value/SurVicPart': v1})
		[tGrad] = tc.autograd.grad(v1 * vCELoss, [tSurIns])
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

		# 计算重构损失
		lEmbeds = sublist(v.lBtmOut, self.lRPs)
		lRecons, vLoss1, vLoss2 = self.getRecon(lEmbeds, self.args.fValAlpha)
		vReconLoss = vLoss1 + vLoss2
		for i, idx in enumerate(self.lRPs):
			v.lBtmOut[idx] = lRecons[i]

		m.logDict({'loss/ReconVal': vReconLoss, 'loss/ReconVal1': vLoss1, 'loss/ReconVal2': vLoss2})

	@override
	def onValLoss(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		for k, v in d.items():
			probs = F.softmax(v.zTopOut, 1)
			entropy = tc.distributions.Categorical(probs).entropy().mean().item()
			m.logDict({f'entropy/Val{k}': entropy})

			if m.current_epoch > 0:
				self.logSur(m, k, v)
				self.logSGBA(m, k, v)

	def logSur(self, m: BaseVFLArch, k: str, v: StepVars) -> None:
		"""记录代理模型损失。"""
		logk = f'Val{v.iLoaderIdx}_{k}'
		# 获取真实模型的 Logits
		tRealLogits = v.zTopOut
		# 获取代理模型的 Logits
		tSurIns = tc.cat(sublist(v.lTopIns, self.lAPs), dim=1)
		tSurLogits = self.zSurNet(tSurIns)

		# 计算代理模型的保真度指标
		rConsistency, fKLDiv = calcFidelity(tSurLogits, tRealLogits)
		m.logDict({f'accGod/{logk}/Consistency': rConsistency, f'accGod/{logk}/KLDiv': fKLDiv})

		# 计算代理模型的预测正确率
		loss = self.criSur(tSurLogits, v.labels)
		[acc1, acc3] = accuracy(tSurLogits, v.labels, (1, 3))
		m.logDict({f'lossSur/{logk}': loss, f'accSur/{logk}/Top1': acc1, f'accSur/{logk}/Top3': acc3})

		# 获取非目标类样本的掩码
		mask = v.labels != m.ns.iTgtLabel
		if mask.any():
			tTgtLogits = tSurLogits[mask]
			tTgtLabels = tc.full_like(v.labels[mask], m.ns.iTgtLabel)

			loss = m.criterion(tTgtLogits, tTgtLabels)
			[acc1, acc3] = accuracy(tTgtLogits, tTgtLabels, (1, 3))
			m.logDict(
				{f'lossSurTgt/{logk}': loss, f'accSurTgt/{logk}/Top1': acc1, f'accSurTgt/{logk}/Top3': acc3}
			)

	@staticmethod
	def logSGBA(m: BaseVFLArch, k: str, v: StepVars) -> None:
		"""记录 SGBA 损失。"""
		logk = f'Val{v.iLoaderIdx}_{k}'
		# 获取非目标类样本的掩码
		mask = v.labels != m.ns.iTgtLabel
		if mask.any():
			tTgtLogits = v.zTopOut[mask]
			tTgtLabels = tc.full_like(v.labels[mask], m.ns.iTgtLabel)

			loss = m.criterion(tTgtLogits, tTgtLabels)
			[acc1, acc3] = accuracy(tTgtLogits, tTgtLabels, (1, 3))
			m.logDict({f'lossTgt/{logk}': loss, f'accTgt/{logk}/Top1': acc1, f'accTgt/{logk}/Top3': acc3})
