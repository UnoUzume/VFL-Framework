"""Villain 攻击实现模块"""

from typing import override

from main.arch import BaseVFLArch
from main.callback import VFLCallback
from utils.common import copy, tc
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
	embed_std = tc.std(embeds, dim=0)
	_, top_indices = tc.topk(embed_std, k=embeds.size(dim=1) // 2)
	# m_elements_indices = torch.argsort(embed_std, descending=True)

	# if isAugment:  # * Backdoor Augmentation: Dropout
	# 	top_indices = RNG.choice(top_indices.cpu(), int(len(top_indices) * 0.75), False)

	# 构造触发器掩码 M
	mask = tc.zeros_like(embeds[0])
	mask[top_indices] = 1.0

	if isAugment:  # * Backdoor Augmentation: Shifting
		gamma = rng().uniform(low=0.6, high=1.2)
		mask *= gamma

	# ? @Bai2023VILLAIN:
	# ? the average standard deviation of elements in the backdoor dimension of all samples
	delta: float = tc.std(embeds[:, top_indices], dim=0).mean().item()
	# delta: float = tc.std(embeds[:, top_indices], dim=1).mean().item()
	pattern = tc.full_like(embeds[0], fill_value=delta)
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

	def __init__(self) -> None:
		"""初始化实例。"""
		self.iRP = 0
		"""攻击节点的索引"""

	# ============
	# 训练阶段
	# ============

	@override
	def onTrainBtmIns(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch > 7:
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
			tSelect = tc.randint(high=len(tVicPos), size=(len(tDstPos),), device=m.device)
			# 方案二：无放回抽取，但是来源样本可能少于目的样本
			# tSelect = tc.randperm(len(tVicPos), device=m.device)[: len(tDstPos)]

			# 批量样本切换
			v.lBtmIns[self.iRP][tDstPos] = v.lBtmIns[self.iRP][tVicPos[tSelect]]

	@override
	def onTrainBtmOut(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch > 7:
			self.poison(v)

	def poison(self, v: StepVars) -> None:
		"""执行投毒。"""
		if self.tDstMask.any():
			eps = createEps(embeds=v.lBtmOut[self.iRP].detach(), beta=1, isAugment=True)
			v.lBtmOut[self.iRP][self.tDstMask] += eps

	# ============
	# 验证阶段
	# ============

	@override
	def onValStepVars(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		d['Attack'] = copy(d['Origin'])

	@override
	def onValBtmOut(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		v = d['Attack']
		v.lBtmOut[self.iRP] += createEps(embeds=v.lBtmOut[self.iRP].detach(), beta=2)

	@override
	def onValLoss(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		for k, v in d.items():
			self.logVal(m, k, v)

	@staticmethod
	def logVal(m: BaseVFLArch, k: str, v: StepVars) -> None:
		"""记录验证阶段的损失。"""
		logk = f'Val{v.iLoaderIdx}_{k}'
		# 获取非目标类样本的掩码
		mask = v.labels != m.ns.iTgtLabel
		if mask.any():
			tTgtLogits = v.zTopOut[mask]
			tTgtLabels = tc.full_like(v.labels[mask], m.ns.iTgtLabel)

			loss = m.criterion(tTgtLogits, tTgtLabels)
			[acc1, acc3] = accuracy(lprobs=tTgtLogits, target=tTgtLabels, topk=(1, 3))
			m.logDict({f'lossTgt/{logk}': loss, f'accTgt/{logk}/Top1': acc1, f'accTgt/{logk}/Top3': acc3})
