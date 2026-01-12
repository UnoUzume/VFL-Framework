"""SGBA 攻击实现模块"""

from dataclasses import dataclass
from typing import override

from torch.optim import AdamW, lr_scheduler as lrs

from main.arch import BaseVFLArch
from main.callback import OPT_TYPE, VFLCallback
from models.fcn import FCN
from projects.vfl.config import AppConfig, createLRS
from utils.common import F, copy, math, nn, np, tc
from utils.config import rng
from utils.define import StepVars
from utils.misc import accuracy, segment

# TODO: attackBtmIns getRecon


def selectByList[T](lTensors: list[T], lIndices: list[int]) -> list[T]:
	return [lTensors[i] for i in lIndices]


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

		self.lRPs = [0]
		self.lSPs = [1]
		self.lAPs = self.lRPs + self.lSPs

		lPartyDims = self.cfg.model.lPartyDims
		nDim = sum(selectByList(lPartyDims, self.lRPs))
		self.zRecNet = FCN([nDim, int(nDim * 0.75), nDim], False)  #! 可变
		"""攻击者用于生成后门触发器的网络"""

		self.zSurNet = FCN([sum(selectByList(lPartyDims, self.lAPs)), 256, 10])
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
		tEmbed = tc.cat(lEmbeds, 1)
		lDims = [t.size(1) for t in lEmbeds]

		tRecOut = self.zRecNet(tEmbed)
		lRecOut = list(tc.split(tRecOut, lDims, 1))

		lLoss = [tc.norm(a - b, 2, 1).mean() for a, b in zip(lEmbeds, lRecOut, strict=True)]
		vLoss = tc.Tensor(lLoss).sum()

		tRecon = tRecOut * alpha + tEmbed * (1 - alpha)
		lRecons = list(tc.split(tRecon, lDims, 1))

		# self.lossRec = F.mse_loss(tEmbed, tRecon, reduction='sum') / len(v.indices)
		return lRecons, vLoss

	# ============
	# 训练阶段
	# ============

	@override
	def onTrainBtmIns(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch > 0:
			self.attackBtmIns(m, v)

	def attackBtmIns(self, m: BaseVFLArch, v: StepVars) -> None:
		aBatchIdxs = v.indices.cpu().numpy()
		# 获取投毒目的样本（属于目标类样本）的位置（索引的索引）
		aDstPos = np.flatnonzero(np.isin(aBatchIdxs, m.ns.tDstIdxs.cpu().numpy()))  #: aBatchIdxs_DstPos
		self.aDstPos = aDstPos
		# 获取受害类样本的位置（索引的索引）
		aVicPos = np.flatnonzero(np.isin(aBatchIdxs, m.ns.tVicIdxs.cpu().numpy()))  #: aBatchIdxs_VicPos
		self.aVicPos = aVicPos

		if len(aDstPos) > 0 and len(aVicPos) > 0:
			# 选择投毒来源样本（属于受害类样本）的位置
			aSrcPos = rng().choice(aVicPos, len(aDstPos), len(aVicPos) < len(aDstPos))
			for dst, src in zip(aDstPos, aSrcPos, strict=True):
				for i in self.lRPs:
					v.lBtmIns[i][dst] = v.lBtmIns[i][src]

	@override
	def onTrainBtmOut(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch > 0:
			self.attackBtmOut(m, v)

	def attackBtmOut(self, m: BaseVFLArch, v: StepVars) -> None:
		#! 不使用 detach()，让底层模型也更新，降低触发器生成网络的训练难度
		lEmbeds = selectByList(v.lBtmOut, self.lRPs)
		_, self.vReconLoss = self.getRecon(lEmbeds)
		m.logDict({'loss/ReconTrain': self.vReconLoss})

		# # 投毒操作

		for i in self.lRPs:
			v.lBtmOut[i] = v.lBtmOut[i].clone()  # 避免 In-place 操作错误

		# 目标类样本（目的样本 -> 来源样本，建立目标类与来源样本的联系）
		if len(self.aDstPos) > 0:  # 如果找到投毒目的样本
			#! 使用 detach() 避免投毒样本的梯度传播至底层模型，产生意外影响
			lEmbeds = [t[self.aDstPos].detach() for t in selectByList(v.lBtmOut, self.lRPs)]
			lRecons, _ = self.getRecon(lEmbeds, self.args.fTrainAlpha)
			for i in self.lRPs:
				v.lBtmOut[i][self.aDstPos] = lRecons[i]

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
	def onTrainTopInsGrad(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch > 0:
			# self.doSur(m, v)
			self.doSGBA(m, v)

	def doSur(self, m: BaseVFLArch, v: StepVars) -> None:
		"""代理模型训练"""
		# 模型损失
		tSurIns = tc.cat(selectByList(v.lTopIns, self.lAPs), 1).detach().requires_grad_()
		tSurOut = self.zSurNet(tSurIns)

		tSurLabels = m.ns.tPreds[tc.searchsorted(m.ns.tIDs, v.indices)]

		vModelLoss = self.criSur(tSurOut, tSurLabels)

		# 梯度损失
		[tSurGrad] = tc.autograd.grad(vModelLoss, [tSurIns], create_graph=True)
		tRawGrad = tc.cat(selectByList(v.lTopInsGrad, self.lAPs), 1)
		vGradLoss = tc.norm(tSurGrad - tRawGrad, 2)

		# 总损失
		tSurLoss = 5 * vModelLoss + 50 * vGradLoss  #! 调整权重
		m.manual_backward(tSurLoss, retain_graph=True)

		m.logDict({'lossSur/Sur': tSurLoss, 'lossSur/Grad': vGradLoss, 'lossSur/Model': vModelLoss})

		# # 投毒操作

		# 受害类样本（目的标签 <- 来源标签，建立目标类与来源样本的联系）

		if len(self.aVicPos) > 0:
			with tc.no_grad():
				aSrcPos = rng().choice(self.aVicPos, math.ceil(len(self.aVicPos) * 0.05), False)
				lEmbeds = [v.lBtmOut[i][aSrcPos] for i in self.lAPs]
				# tClean = tc.cat(lEmbeds, 1)

			lRecons, _ = self.getRecon(lEmbeds[: len(self.lRPs)], self.args.fTrainAlpha)
			lEmbeds = [*lRecons, *lEmbeds[len(self.lRPs) :]]
			tPoison = tc.cat(lEmbeds, 1)

			tSurIns = tPoison.detach().requires_grad_()
			tSurOut = self.zSurNet(tSurIns)
			tTgtLabels = tc.full_like(tSurLabels[aSrcPos], m.ns.iTgtLabel)
			vCELoss = self.criSur(tSurOut, tTgtLabels)
			vEntropy = -(F.softmax(tSurOut, 1) * F.log_softmax(tSurOut, 1)).sum(1).mean()
			[tGrad] = tc.autograd.grad(2 * vCELoss - 2 * vEntropy, [tSurIns])
			m.manual_backward(tPoison, tGrad, retain_graph=True)
			m.logDict({'loss/SurVic': vCELoss, 'loss/SurVicEntropy': vEntropy})

		#! 对受害类样本添加不完全触发器不应该触发后门

		# if len(self.aVicPos) > 0:
		# 	with tc.no_grad():
		# 		aSrcPos = rng().choice(self.aVicPos, math.ceil(len(self.aVicPos) * 0.02), False)
		# 		lEmbeds = [t[aSrcPos] for t in v.lBtmOut[: self.nAP + 1]]

		# 	lRecons, _ = self.getRecon(lEmbeds[: self.nAP], self.args.fTrainAlpha)
		# 	tWhichAP = tc.randint(self.nAP, (len(lEmbeds[0]),))  # 只有单个 AP 进行投毒
		# 	for i in range(self.nAP):
		# 		lEmbeds[i][tWhichAP == i] = lRecons[i][tWhichAP == i]
		# 	tPoison = tc.cat(lEmbeds, 1)

		# 	tSurIns = tPoison.detach()
		# 	tSurOut = self.zSurNet(tSurIns)
		# 	vCELoss = self.criSur(tSurOut, tSurLabels[aSrcPos])
		# 	[tGrad] = tc.autograd.grad(vCELoss, tSurIns)
		# 	m.manual_backward(tPoison, tGrad, retain_graph=True)
		# 	m.logDict({'loss/SurVicPart': vCELoss})

	def doSGBA(self, m: BaseVFLArch, v: StepVars) -> None:
		"""SGBA 攻击"""
		p = segment(m.current_epoch, (15, 20), self.args.lLossScales)
		m.manual_backward(self.vReconLoss * p, retain_graph=True)

		if len(self.aDstPos) > 0:  # 如果找到投毒目标
			for i in self.lRPs:
				value = segment(m.current_epoch, (15, 20), self.args.lGradScales)
				v.lTopInsGrad[i][self.aDstPos] *= value

		# if len(self.aSrcPos2) > 0:
		# 	for i in range(self.nAP):
		# 		value = segment(m.current_epoch, (15, 20), (8.0, 4.0))
		# 		v.lTopInsGrad[i][self.aSrcPos2] *= value

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

		lEmbeds = [v.lBtmOut[i] for i in self.lRPs]
		lRecons, vReconLoss = self.getRecon(lEmbeds, self.args.fValAlpha)
		for i in self.lRPs:
			v.lBtmOut[i] = lRecons[i]

		m.logDict({'loss/ReconVal': vReconLoss})

	@override
	def onValLoss(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		for k, v in d.items():
			probs = F.softmax(v.zTopOut, 1)
			entropy = tc.distributions.Categorical(probs).entropy().mean().item()  # type: ignore[no-untyped-call]
			m.logDict({f'entropy/Val{k}': entropy})

			if m.current_epoch > 0:
				#! self.logSur(m, v, k)
				self.logSGBA(m, v, k)

	def logSur(self, m: BaseVFLArch, v: StepVars, k: str) -> None:
		tSurIns = tc.cat(selectByList(v.lTopIns, self.lAPs), 1)
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
