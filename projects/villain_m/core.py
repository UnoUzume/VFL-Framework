"""Villain 攻击实现模块"""

from typing import override

from main.arch import BaseVFLArch
from main.callback import VFLCallback
from utils.common import copy, np, tc
from utils.config import rng
from utils.define import StepVars
from utils.misc import accuracy


def createEps(embeds: tc.Tensor, beta: float = 0.4, isAugment: bool = False) -> tc.Tensor:
	"""生成用于后门攻击的触发器向量。

	该方法通过分析嵌入特征的标准差，选取敏感维度并构造触发器掩码，
	最终生成符合 VILLAIN 攻击要求的触发器向量。

	Args:
		embeds: 输入的嵌入特征张量
		beta: 触发器强度系数，默认为 `0.4`
		isAugment: 是否进行数据增强，默认为 `False`

	Returns:
		生成的触发器向量，形状与输入嵌入特征相同
	"""
	embed_std = tc.std(embeds, 0)
	_, top_indices = tc.topk(embed_std, 64)
	# m_elements_indices = torch.argsort(embed_std, descending=True)

	# if isAugment:  # * Backdoor Augmentation: Dropout
	# 	top_indices = RNG.choice(top_indices.cpu(), int(len(top_indices) * 0.75), False)

	# 构造触发器掩码 M
	mask = tc.zeros_like(embeds[0])
	mask[top_indices] = 1.0

	if isAugment:  # * Backdoor Augmentation: Shifting
		gamma = rng().uniform(0.6, 1.2)
		mask *= gamma

	# ? @Bai2023VILLAIN:
	# ? the average standard deviation of elements in the backdoor dimension of all samples
	delta = tc.std(embeds[:, top_indices], 0).mean().item()
	# delta = tc.std(embeds[:, top_indices], 1).mean().item()
	pattern = tc.full_like(embeds[0], delta)
	pattern[2::4] *= -1.0
	pattern[3::4] *= -1.0

	# 触发器 E = M ⊗ (β · Δ)
	eps = mask * beta * pattern
	return eps


class VillainCb(VFLCallback):
	"""VILLAIN 攻击实现，一种针对垂直联邦学习的后门攻击方法

	该回调类实现了 VILLAIN 攻击的核心逻辑，包括在训练阶段向嵌入特征中注入后门触发器，
	在验证/测试阶段触发后门行为，使模型将特定输入错误分类为目标类别。
	"""

	@override
	def onInitModule(self, m: BaseVFLArch) -> None:
		m.logText.info(f'{self.__class__.__name__}.onInitModule()')

	# ============
	# 训练阶段
	# ============

	@override
	def onTrainBtmIns(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch < 7:
			return
		# 获取当前批次中投毒目的样本、非目标类样本的位置（索引的索引）
		aBatchIdxs = v.indices.cpu().numpy()
		aDstPos = np.flatnonzero(np.isin(aBatchIdxs, m.ns.aDstIdxs))  #: aBatchIdxs_DstPos
		self.aDstPos = aDstPos
		aVicPos = np.flatnonzero(np.isin(aBatchIdxs, m.ns.aVicIdxs))  #: aBatchIdxs_VicPos

		if len(aDstPos) > 0 and len(aVicPos) > 0:  # 如果找到投毒目标
			aSrcPos = rng().choice(aVicPos, len(aDstPos), len(aVicPos) < len(aDstPos))
			for dst, src in zip(aDstPos, aSrcPos, strict=True):
				v.lBtmIns[0][dst] = v.lBtmIns[0][src]

	@override
	def onTrainBtmOut(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch < 7:
			return

		if len(self.aDstPos) > 0:  # 如果找到投毒目标
			v.lBtmOut[0][self.aDstPos] += createEps(v.lBtmOut[0].detach(), 1, True)

	# ============
	# 验证阶段
	# ============

	@override
	def onValStepVars(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		d['Attack'] = copy(d['Origin'])

	@override
	def onValBtmOut(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		v = d['Attack']
		v.lBtmOut[0] += createEps(v.lBtmOut[0].detach(), 2)

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
