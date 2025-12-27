"""LFBA 架构模块

本模块实现了基于标签翻转的后门攻击方法，包括本地模型、学习率调度器、
触发器转换以及完整的 VFL 架构实现。
"""

from typing import override

from main.arch import BaseVFLArch
from main.callback import VFLCallback
from utils.common import copy, np, tc
from utils.config import rng
from utils.define import StepVars
from utils.misc import accuracy
from utils.vision import createPostTrans, createSpatialTrans

from .method import AddTrigger


class LFBACb(VFLCallback):
	"""LFBA 架构实现类

	实现 LFBA 攻击的纵向联邦学习架构，包括模型训练、验证和攻击逻辑。
	"""

	@override
	def onInitModule(self, m: BaseVFLArch) -> None:
		m.logText.info(f'{self.__class__.__name__}.onInitModule()')

	# ============
	# 训练阶段
	# ============

	@override
	def onFitStart(self, m: BaseVFLArch) -> None:
		m.logText.info(m.trainer.log_dir)

		# 数据转换
		self.tfNormal = createSpatialTrans(m.module.lNormalTrans)
		self.tfAugment = createSpatialTrans(m.module.lAugmentTrans)
		self.tfTrigger = AddTrigger()
		self.tfPost = createPostTrans()

	@override
	def onTrainBtmIns(self, m: 'BaseVFLArch', v: StepVars) -> None:
		v.lBtmIns = [self.tfAugment(images) for images in v.lBtmIns]

		if m.current_epoch >= 1:
			# 获取当前批次中投毒目的样本、非目标类样本的位置（索引的索引）
			aBatchIdxs = v.indices.cpu().numpy()
			aDstPos = np.flatnonzero(np.isin(aBatchIdxs, m.ns.aDstIdxs))  #: aBatchIdxs_DstPos
			aNonPos = np.flatnonzero(np.isin(aBatchIdxs, m.ns.aNonIdxs))  #: aBatchIdxs_NonPos

			if len(aDstPos) > 0 and len(aNonPos) > 0:  # 如果找到投毒目标
				aSrcPos = rng().choice(aNonPos, len(aDstPos), len(aNonPos) < len(aDstPos))
				for dst, src in zip(aDstPos, aSrcPos, strict=True):
					v.lBtmIns[0][dst] = self.tfTrigger(v.lBtmIns[0][src])

		v.lBtmIns = [self.tfPost(images) for images in v.lBtmIns]

	# ============
	# 验证阶段
	# ============

	@override
	def onValStepVars(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		d['Attack'] = copy(d['Origin'])

	@override
	def onValBtmIns(self, m: 'BaseVFLArch', d: dict[str, StepVars]) -> None:
		v = d['Origin']
		v.lBtmIns = [self.tfNormal(images) for images in v.lBtmIns]
		v.lBtmIns = [self.tfPost(images) for images in v.lBtmIns]

		v = d['Attack']
		v.lBtmIns = [self.tfNormal(images) for images in v.lBtmIns]
		v.lBtmIns[0] = self.tfTrigger(v.lBtmIns[0])
		v.lBtmIns = [self.tfPost(images) for images in v.lBtmIns]

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
