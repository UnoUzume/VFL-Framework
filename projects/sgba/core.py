"""SGBA 攻击实现模块"""

from dataclasses import dataclass
from typing import override

from torch.optim import AdamW, lr_scheduler as lrs

from main.arch import BaseVFLArch
from main.callback import OPT_TYPE, VFLCallback
from models.fcn import FCN
from projects.vfl.config import AppConfig
from utils.common import F, copy, np, tc
from utils.config import rng
from utils.define import StepVars
from utils.misc import accuracy, segment


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

		nDim = self.cfg.model.lPartyDims[0]
		self.zRecNet = FCN([nDim, int(nDim * 0.75), nDim], False)  #! 可变
		"""攻击者用于生成后门触发器的网络"""

	@override
	def onInitModule(self, m: BaseVFLArch) -> None:
		m.add_module('zRecNet', self.zRecNet)

	@override
	def onConfigOptims(self) -> OPT_TYPE:
		optRec = AdamW(self.zRecNet.parameters(), self.args.fRecLr)
		lrsRec = lrs.ConstantLR(optRec, 0.8, 5)
		return [optRec], [lrsRec]

	def getRecon(self, tEmbed: tc.Tensor, alpha: float = 1.0) -> tuple[tc.Tensor, tc.Tensor]:
		tRecon = self.zRecNet(tEmbed) * alpha + tEmbed * (1 - alpha)
		vLoss = tc.norm(tEmbed - tRecon, 2, 1).mean()

		# self.lossRec = F.mse_loss(tEmbed, tRecon, reduction='sum') / len(v.indices)
		return tRecon, vLoss

	# ============
	# 训练阶段
	# ============

	@override
	def onTrainBtmIns(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch > 0:
			self.attackBtmIns(m, v)

	def attackBtmIns(self, m: BaseVFLArch, v: StepVars) -> None:
		# 获取当前批次中投毒目的样本、非目标类样本的位置（索引的索引）
		aBatchIdxs = v.indices.cpu().numpy()
		aDstPos = np.flatnonzero(np.isin(aBatchIdxs, m.ns.aDstIdxs))  #: aBatchIdxs_DstPos
		self.aDstPos = aDstPos
		aVicPos = np.flatnonzero(np.isin(aBatchIdxs, m.ns.aVicIdxs))  #: aBatchIdxs_VicPos

		if len(aDstPos) > 0 and len(aVicPos) > 0:
			# 选择投毒来源样本（属于受害类样本）的位置
			aSrcPos = rng().choice(aVicPos, len(aDstPos), len(aVicPos) < len(aDstPos))
			for dst, src in zip(aDstPos, aSrcPos, strict=True):
				v.lBtmIns[0][dst] = v.lBtmIns[0][src]

		# if len(aDstPos) > 0:  # 如果找到投毒目标
		# 	# 寻找投毒目标对应的 self.aDstIdxs 的元素位置，同时作为 self.aSrcIdxs 的元素位置
		# 	aSrcIdxs_Pos = _aDstIdxs_Pos = np.where(aBatchIdxs[aDstPos, None] == m.ns.aDstIdxs)[1]
		# 	# 获取投毒来源的元素
		# 	aSrcIdxsSubset = m.ns.aSrcIdxs[aSrcIdxs_Pos]

		# 	for pos, src_idx in zip(self.aDstPos, aSrcIdxsSubset, strict=True):
		# 		image = m.module.dsTrain[src_idx.item()][0]  # pyright: ignore[reportAttributeAccessIssue]

		# 		parts = m.module.fnSplit(image.unsqueeze(0), m.module.nParty)
		# 		parts = [m.module.tfCurrent(part) for part in parts]
		# 		v.lBtmIns[0][pos] = parts[0]

	@override
	def onTrainBtmOut(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch > 0:
			self.attackBtmOut(m, v)

	def attackBtmOut(self, m: BaseVFLArch, v: StepVars) -> None:
		#! 不使用 detach()，让底层模型也更新，降低触发器生成网络的训练难度
		tEmbed = v.lBtmOut[0]
		_, self.vReconLoss = self.getRecon(tEmbed)
		m.logDict({'loss/ReconTrain': self.vReconLoss})

		# # 投毒操作
		v.lBtmOut[0] = v.lBtmOut[0].clone()  # 避免 In-place 操作错误

		# 目标类样本（目的样本 -> 来源样本，建立目标类与来源样本的联系）
		if len(self.aDstPos) > 0:  # 如果找到投毒目的样本
			#! 使用 detach() 避免投毒样本的梯度传播至底层模型，产生意外影响
			tEmbed = v.lBtmOut[0][self.aDstPos].detach()
			tRecon, _ = self.getRecon(tEmbed, self.args.fTrainAlpha)
			v.lBtmOut[0][self.aDstPos] = tRecon

	@override
	def onTrainLoss(self, m: BaseVFLArch, v: StepVars) -> None:
		optRec = m.getOptim(self.iOpt)
		optRec.zero_grad()

	@override
	def onTrainTopInsGrad(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch > 0:
			self.doSGBA(m, v)

	def doSGBA(self, m: BaseVFLArch, v: StepVars) -> None:
		"""SGBA 攻击"""
		p = segment(m.current_epoch, (15, 20), self.args.lLossScales)
		m.manual_backward(self.vReconLoss * p, retain_graph=True)

		if len(self.aDstPos) > 0:  # 如果找到投毒目标
			value = segment(m.current_epoch, (15, 20), self.args.lGradScales)
			v.lTopInsGrad[0][self.aDstPos] *= value

	@override
	def onTrainOptimStep(self, m: BaseVFLArch, v: StepVars) -> None:
		optRec = m.getOptim(self.iOpt)
		optRec.step()

	# ============
	# 验证阶段
	# ============

	@override
	def onValStepVars(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		d['Attack'] = copy(d['Origin'])

	@override
	def onValBtmOut(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		v = d['Attack']

		tEmbed = v.lBtmOut[0]
		tRecon, vReconLoss = self.getRecon(tEmbed, self.args.fValAlpha)
		v.lBtmOut[0] = tRecon

		m.logDict({'loss/ReconVal': vReconLoss})

	@override
	def onValLoss(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		for k, v in d.items():
			probs = F.softmax(v.zTopOut, 1)
			entropy = tc.distributions.Categorical(probs).entropy().mean().item()  # type: ignore[no-untyped-call]
			m.logDict({f'entropy/Val{k}': entropy})

			if m.current_epoch > 0:
				self.logSGBA(m, v, k)

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
