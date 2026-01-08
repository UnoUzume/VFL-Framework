"""SGBA 攻击实现模块"""

from dataclasses import dataclass
from typing import override

from torch.optim import Adam

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

		nDim = config.model.lPartyDims[0]
		self.zRecNet = FCN([nDim, int(nDim * 0.75), nDim], False, 'relu')  #! 可变
		"""攻击者用于生成后门触发器的网络"""

	@override
	def onInitModule(self, m: BaseVFLArch) -> None:
		m.logText.info(f'{self.__class__.__name__}.onInitModule()')
		m.add_module('zRecNet', self.zRecNet)

	@override
	def onConfigOptims(self) -> OPT_TYPE:
		optRec = Adam(self.zRecNet.parameters(), self.args.fRecLr)
		return [optRec]

	# ============
	# 训练阶段
	# ============

	@override
	def onFitStart(self, m: BaseVFLArch) -> None:
		m.logText.info(m.trainer.log_dir)

	@override
	def onTrainBtmIns(self, m: BaseVFLArch, v: StepVars) -> None:
		optRec = m.getOptim(self.iOpt)
		optRec.zero_grad()

		if m.current_epoch < 1:
			return

		# 获取当前批次中投毒目的样本、非目标类样本的位置（索引的索引）
		aBatchIdxs = v.indices.cpu().numpy()
		aDstPos = np.flatnonzero(np.isin(aBatchIdxs, m.ns.aDstIdxs))  #: aBatchIdxs_DstPos
		self.aDstPos = aDstPos
		aVicPos = np.flatnonzero(np.isin(aBatchIdxs, m.ns.aVicIdxs))  #: aBatchIdxs_VicPos

		# if len(aDstPos) > 0 and len(aVicPos) > 0:  # 如果找到投毒目标
		# 	aSrcPos = rng().choice(aVicPos, len(aDstPos), len(aVicPos) < len(aDstPos))
		# 	for dst, src in zip(aDstPos, aSrcPos, strict=True):
		# 		v.lBtmIns[0][dst] = v.lBtmIns[0][src]

		if len(aDstPos) > 0:  # 如果找到投毒目标
			# 寻找投毒目标对应的 self.aDstIdxs 的元素位置，同时作为 self.aSrcIdxs 的元素位置
			aSrcIdxs_Pos = _aDstIdxs_Pos = np.where(aBatchIdxs[aDstPos, None] == m.ns.aDstIdxs)[1]
			# 获取投毒来源的元素
			aSrcIdxsSubset = m.ns.aSrcIdxs[aSrcIdxs_Pos]

			for pos, src_idx in zip(self.aDstPos, aSrcIdxsSubset, strict=True):
				image = m.module.dsTrain[src_idx.item()][0]  # pyright: ignore[reportAttributeAccessIssue]

				parts = m.module.fnSplit(image.unsqueeze(0), m.module.nParty)
				parts = [m.module.tfCurrent(part) for part in parts]
				v.lBtmIns[0][pos] = parts[0]

	@override
	def onTrainBtmOut(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch < 1:
			return

		#! 不使用 detach()，让底层模型也更新，降低触发器生成网络的训练难度
		tEmbeds = v.lBtmOut[0]
		tRecon = self.zRecNet(tEmbeds)
		self.lossRec = tc.norm(tEmbeds - tRecon, 2, 1).mean()
		# self.lossRec = F.mse_loss(tEmbeds, tRecon, reduction='sum') / len(v.indices)
		m.logDict({'loss/Recon': self.lossRec})

		if len(self.aDstPos) > 0:  # 如果找到投毒目标
			alpha = self.args.fTrainAlpha

			#! 使用 detach() 避免投毒样本的梯度传播至底层模型，产生意外影响
			tEmbeds_ = v.lBtmOut[0][self.aDstPos].detach()
			tRecon_ = self.zRecNet(tEmbeds_) * alpha + tEmbeds_ * (1 - alpha)
			v.lBtmOut[0] = v.lBtmOut[0].clone()
			v.lBtmOut[0][self.aDstPos] = tRecon_

	@override
	def onTrainTopInsGrad(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch < 1:
			return

		p = segment(m.current_epoch, (15, 20), self.args.lLossScales)
		m.manual_backward(self.lossRec * p, retain_graph=True)

		if len(self.aDstPos) > 0:  # 如果找到投毒目标
			v.lTopInsGrad[0][self.aDstPos] *= segment(m.current_epoch, (15, 20), self.args.lGradScales)

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
		alpha = self.args.fValAlpha

		v = d['Attack']
		tEmbeds = v.lBtmOut[0]
		tRecon = self.zRecNet(tEmbeds) * alpha + tEmbeds * (1 - alpha)
		v.lBtmOut[0] = tRecon

		lossRecon = tc.norm(tEmbeds - tRecon, 2, 1).mean()
		m.logDict({'loss/Pattern': lossRecon})

	@override
	def onValLoss(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		for k, v in d.items():
			# probs = tc.nn.functional.softmax(v.zTopOut, 1)
			# entropy = tc.distributions.Categorical(probs).entropy().mean().item()
			# self.logDict({f'entropy/Val{k}': entropy})

			mask = v.labels != m.ns.iTgtLabel
			if not mask.any():
				continue

			labels = v.labels[mask]
			zTopOut = v.zTopOut[mask]

			tTgtLabels = tc.full_like(labels, m.ns.iTgtLabel)
			[accT1, accT3] = accuracy(zTopOut, tTgtLabels, (1, 3))
			lossT = m.criterion(zTopOut, tTgtLabels)

			sName = f'Val{k}/Tgt'
			m.logDict({f'loss/{sName}': lossT, f'acc/{sName}/Top1': accT1, f'acc/{sName}/Top3': accT3})
