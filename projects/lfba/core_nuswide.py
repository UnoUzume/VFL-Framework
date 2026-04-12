"""LFBA 攻击模块"""

from typing import override

from main.arch import BaseVFLArch
from main.callback import VFLCallback
from utils.common import copy, tc
from utils.define import StepVars
from utils.misc import accuracy


class LFBACb(VFLCallback):
	"""LFBA 攻击回调类"""

	def __init__(self) -> None:
		"""初始化实例。"""
		self.iRP = 0
		"""攻击节点的索引"""

	# ============
	# 训练阶段
	# ============

	@override
	def onTrainBtmIns(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch >= 1:
			self.switch(m, v)

	def switch(self, m: BaseVFLArch, v: StepVars) -> None:
		"""执行样本切换。"""
		# 获取目标类样本中投毒目的样本的掩码
		self.tDstMask = tc.isin(v.indices, m.ns.tDstIds)
		# 获取受害类样本的掩码
		self.tVicMask = tc.isin(v.indices, m.ns.tVicIds)

		# 在批次内部执行样本切换
		if self.tDstMask.any() and self.tVicMask.any():
			# 获取目标类样本中投毒目的样本的位置
			[tDstPos] = tc.nonzero(self.tDstMask, as_tuple=True)
			# 获取受害类样本的位置
			[tVicPos] = tc.nonzero(self.tVicMask, as_tuple=True)

			# 方案一：有放回抽取
			tSelect = tc.randint(high=len(tVicPos), size=(len(tDstPos),), device=m.device)
			# 方案二：无放回抽取，但是来源样本可能少于目的样本
			# tSelect = tc.randperm(n=len(tVicPos), device=m.device)[: len(tDstPos)]

			# 批量样本切换
			v.lBtmIns[self.iRP][tDstPos] = v.lBtmIns[self.iRP][tVicPos][tSelect]
			v.lBtmIns[self.iRP] = v.lBtmIns[self.iRP].clone()
			v.lBtmIns[self.iRP][tDstPos][:, :10] = 1

	# ============
	# 验证阶段
	# ============

	@override
	def onValStepVars(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		d['Attack'] = copy(d['Origin'])

	@override
	def onValBtmIns(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		v = d['Attack']
		v.lBtmIns[self.iRP] = v.lBtmIns[self.iRP].clone()
		v.lBtmIns[self.iRP][:, :10] = 1

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
