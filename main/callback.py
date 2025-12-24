"""回调模块，定义了垂直联邦学习（VFL）生命周期的回调接口。

本模块提供了 `VFLCallback` 基类，允许开发者通过钩子函数介入 VFL 训练的各个阶段
（如前向传播前后、梯度计算前后等）。同时提供了 `@priority` 装饰器用于控制回调执行顺序。
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from torch.optim import Optimizer, lr_scheduler as lr

from utils.define import StepVars

if TYPE_CHECKING:
	from .arch import BaseVFLArch


def priority[F: Callable[..., Any]](level: int = 0) -> Callable[[F], F]:
	"""【**装饰器**】设置回调方法的执行优先级。

	在同一个钩子点（如 `onTrainStepVars()`）如果有多个回调函数，
	优先级数值**越大**的越先执行。

	Args:
		level: 优先级数值，默认为 `0`。

	Returns:
		装饰后的函数，带有 `priority` 属性。

	Example:
		>>> class MyCallback(VFLCallback):
		...   @priority(100)
		...   def onTrainStepVars(self, m, v):
		...     print('我先执行！')
	"""

	def decorator(func: F) -> F:
		func.priority = level  # type: ignore[attr-defined]
		return func

	return decorator


OPT_TYPE = list[Optimizer] | tuple[list[Optimizer], list[lr.LRScheduler]]
"""优化器配置的类型别名"""


class VFLCallback:
	"""VFL 回调基类。

	所有自定义的 VFL 逻辑（如攻击、防御、特定指标记录）都应继承此类，
	并重写相应的钩子方法。

	注意：
		- 方法名中的 `Btm` 指底层模型 (Bottom Model)。
		- 方法名中的 `Top` 指顶层模型 (Top Model)。
		- 验证阶段 (`Val`) 的参数 `d` 是字典，而训练/测试阶段是单个 `StepVars`。
	"""

	_iOpt: int | None = None
	"""【内部属性】回调中优化器的初始索引"""
	_iLRS: int | None = None
	"""【内部属性】回调中学习率调度器的初始索引"""

	@property
	def iOpt(self) -> int:
		"""回调中优化器的初始索引"""
		if self._iOpt is None:
			msg = '回调中优化器的初始索引未设置！'
			raise ValueError(msg)
		return self._iOpt

	@iOpt.setter
	def iOpt(self, value: int) -> None:
		self._iOpt = value

	@property
	def iLRS(self) -> int:
		"""回调中学习率调度器的初始索引"""
		if self._iLRS is None:
			msg = '回调中学习率调度器的初始索引未设置！'
			raise ValueError(msg)
		return self._iLRS

	@iLRS.setter
	def iLRS(self, value: int) -> None:
		self._iLRS = value

	# ============
	# 初始化与配置
	# ============

	def onInitModule(self, m: 'BaseVFLArch') -> None:
		"""在 VFL 架构初始化完成时调用。

		Args:
			m: VFL 架构实例。
		"""

	def onConfigOptims(self) -> OPT_TYPE | None:
		"""配置优化器时调用。

		允许回调函数注册自己的优化器（例如用于攻击生成的优化器）。

		Returns:
			优化器列表，或 (优化器列表，调度器列表) 的元组。如果不添加，返回 `None`。
		"""
		return None

	# ============
	# Checkpoint (检查点)
	# ============

	def onSaveCheckpoint(self, m: 'BaseVFLArch', ckpt: dict[str, Any]) -> None:
		"""保存模型检查点时调用。

		Args:
			m: VFL 架构实例。
			ckpt: 包含模型状态的字典。可以直接向其中添加自定义数据。
		"""

	def onLoadCheckpoint(self, m: 'BaseVFLArch', ckpt: dict[str, Any]) -> None:
		"""加载模型检查点时调用。

		Args:
			m: VFL 架构实例。
			ckpt: 加载的检查点字典。可从中恢复自定义数据。
		"""

	# ============
	# 训练过程的回调函数
	# ============

	def onFitStart(self, m: 'BaseVFLArch') -> None:
		"""在整个训练流程开始前调用。"""

	def onTrainEpochStart(self, m: 'BaseVFLArch') -> None:
		"""在每个训练轮次（Epoch）开始时调用。"""

	def onTrainStepVars(self, m: 'BaseVFLArch', v: StepVars) -> None:
		"""在训练 Step 开始，`StepVars` 初始化后调用。

		此时 `v.batch` 已可用，但数据尚未输入模型。
		"""

	def onTrainBtmIns(self, m: 'BaseVFLArch', v: StepVars) -> None:
		"""在底层模型输入 (`v.lBtmIns`) 准备好后调用。"""

	def onTrainBtmOut(self, m: 'BaseVFLArch', v: StepVars) -> None:
		"""在底层模型前向传播完成，输出 (`v.lBtmOut`) 产生后调用。

		常用于截取底层输出（Embedding），或保存用于后续分析。
		"""

	def onTrainTopIns(self, m: 'BaseVFLArch', v: StepVars) -> None:
		"""在顶层模型输入 (`v.lTopIns`) 准备好后调用。

		注意：`v.lTopIns` 通常是 `v.lBtmOut` 的 Clone 并带有梯度。

		常用于对输入特征添加噪声（防御）或执行特征攻击。
		"""

	def onTrainTopOut(self, m: 'BaseVFLArch', v: StepVars) -> None:
		"""在顶层模型前向传播完成，输出 (`v.zTopOut`) 产生后调用。"""

	def onTrainLoss(self, m: 'BaseVFLArch', v: StepVars) -> None:
		"""在损失 (`v.loss`) 计算完成后调用。

		常用于修改 Loss（如添加正则项）或记录 Loss。
		"""

	def onTrainTopInsGrad(self, m: 'BaseVFLArch', v: StepVars) -> None:
		"""在顶层模型反向传播后，获得对输入特征的梯度 (`v.lTopInsGrad`) 时调用。

		常用于梯度反转攻击、梯度加噪防御。
		"""

	def onTrainOptimStep(self, m: 'BaseVFLArch', v: StepVars) -> None:
		"""在优化器更新参数 (`step()`) 后调用。"""

	def onTrainEpochEnd(self, m: 'BaseVFLArch') -> None:
		"""在训练轮次结束时调用。**位于 onValEpochEnd() 之后。**"""

	# ============
	# 验证过程的回调函数
	# ============

	def onValEpochStart(self, m: 'BaseVFLArch') -> None:
		"""在验证轮次开始时调用。"""

	def onValStepVars(self, m: 'BaseVFLArch', d: dict[str, StepVars]) -> None:
		"""在验证 Step 开始，`StepVars` 初始化后调用。

		Args:
			m: VFL 架构实例。
			d: 包含 `StepVars` 的字典。
		"""

	def onValBtmIns(self, m: 'BaseVFLArch', d: dict[str, StepVars]) -> None:
		"""在验证阶段底层输入准备好后调用。"""

	def onValBtmOut(self, m: 'BaseVFLArch', d: dict[str, StepVars]) -> None:
		"""在验证阶段底层输出产生后调用。"""

	def onValTopIns(self, m: 'BaseVFLArch', d: dict[str, StepVars]) -> None:
		"""在验证阶段顶层输入准备好后调用。"""

	def onValTopOut(self, m: 'BaseVFLArch', d: dict[str, StepVars]) -> None:
		"""在验证阶段顶层输出产生后调用。"""

	def onValLoss(self, m: 'BaseVFLArch', d: dict[str, StepVars]) -> None:
		"""在验证阶段损失计算完成后调用。"""

	def onValEpochEnd(self, m: 'BaseVFLArch') -> None:
		"""在验证轮次结束时调用。"""

	# ============
	# 测试过程的回调函数
	# ============

	def onTestEpochStart(self, m: 'BaseVFLArch') -> None:
		"""在测试轮次开始时调用。"""

	def onTestStepVars(self, m: 'BaseVFLArch', v: StepVars) -> None:
		"""在测试 Step 开始时调用。"""

	def onTestBtmIns(self, m: 'BaseVFLArch', v: StepVars) -> None:
		"""在测试阶段底层输入准备好后调用。"""

	def onTestBtmOut(self, m: 'BaseVFLArch', v: StepVars) -> None:
		"""在测试阶段底层输出产生后调用。"""

	def onTestTopIns(self, m: 'BaseVFLArch', v: StepVars) -> None:
		"""在测试阶段顶层输入准备好后调用。"""

	def onTestTopOut(self, m: 'BaseVFLArch', v: StepVars) -> None:
		"""在测试阶段顶层输出产生后调用。"""

	def onTestLoss(self, m: 'BaseVFLArch', v: StepVars) -> None:
		"""在测试阶段损失计算完成后调用。"""

	def onTestEpochEnd(self, m: 'BaseVFLArch') -> None:
		"""在测试轮次结束时调用。"""
