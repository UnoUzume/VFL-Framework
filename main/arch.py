"""架构模块，定义了基于 PyTorch Lightning 的模型架构基类。

本模块包含构建垂直联邦学习（VFL）系统所需的两大基类：

- `LightningArch`: 基础封装，提供日志和优化器管理。
- `BaseVFLArch`: 核心 VFL 逻辑，实现底层模型与顶层模型的协同训练与梯度交互。
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, override

from lightning.pytorch.core.optimizer import LightningOptimizer
from torch.optim import Optimizer, lr_scheduler as lr

from utils.common import L, Path, nn
from utils.define import StepVars
from utils.logger import createFileLogger
from utils.misc import ensureList, notNone
from utils.module import BaseDataModule

from .callback import VFLCallback


class LightningArch(L.LightningModule, ABC):
	"""Lightning 架构基类，封装了 `LightningModule` 的基本功能。

	该类提供了模型初始化、日志记录、优化器配置等基础功能，并启用了手动优化模式。
	它是一个抽象基类，要求子类必须实现 `configOptims()` 方法。
	"""

	def __init__(self, dpRoot: Path | str) -> None:
		"""初始化实例。

		Args:
			dpRoot: 模型保存和日志记录的根目录路径
		"""
		super().__init__()
		self.automatic_optimization = False  #! 启用手动优化

		self.dpRoot = Path(dpRoot)
		"""模型保存和日志记录的根目录路径"""
		self.dpRoot.mkdir(parents=True, exist_ok=True)

		self.logText = createFileLogger(self.dpRoot / 'train.log')
		"""文本日志记录器"""

		self.ns = SimpleNamespace()
		"""命名空间，用于存储模型训练过程中的临时变量"""

	@property
	def module(self) -> BaseDataModule:
		"""当前的 `DataModule` 实例"""
		return notNone(self.trainer.datamodule)  # type: ignore[attr-defined]

	def logDict(self, dPair: dict[str, Any], nBatchSize: int | None = None) -> None:
		"""记录字典形式的日志。

		Args:
			dPair: 包含要记录的键值对的字典
			nBatchSize: 批次大小，用于计算平均指标，_可选_
		"""
		self.log_dict(dPair, add_dataloader_idx=False, batch_size=nBatchSize)

	@override
	def configure_optimizers(self) -> tuple[list[Optimizer], list[lr.LRScheduler]]:
		"""配置优化器和学习率调度器。

		这是 `LightningModule` 的标准钩子，它代理调用了本类的抽象方法 `configOptims()`。

		Returns:
			包含优化器列表和学习率调度器列表的元组
		"""
		return self.configOptims()

	@abstractmethod
	def configOptims(self) -> tuple[list[Optimizer], list[lr.LRScheduler]]:
		"""【**抽象方法**】配置优化器和学习率调度器。

		子类必须实现此方法以定义具体的优化器逻辑。

		Returns:
				包含优化器列表和学习率调度器列表的元组
		"""

	def getOptims(self) -> list[LightningOptimizer]:
		"""获取所有优化器的列表。

		Returns:
			所有优化器的列表
		"""
		return ensureList(self.optimizers())

	def getOptim(self, i: int) -> LightningOptimizer:
		"""根据索引获取特定的优化器。

		Args:
			i: 优化器的索引

		Returns:
			指定索引的优化器
		"""
		return self.getOptims()[i]


class BaseVFLArch(LightningArch, VFLCallback, ABC):
	"""垂直联邦学习（VFL）架构基类。

	该类继承自 `LightningArch` 和 `VFLCallback`，实现了标准的 Split Learning 训练循环。
	它通过回调机制 (`lCallbacks`) 支持灵活的扩展，如梯度攻击、防御、指标计算等。
	"""

	lBtmNets: nn.ModuleList
	"""底层模型列表"""
	zTopNet: nn.Module
	"""顶层模型"""

	def __init__(self, dpRoot: Path | str, lCallbacks: list[VFLCallback] | None = None) -> None:
		"""初始化实例。

		Args:
			dpRoot: 模型保存和日志记录的根目录路径
			lCallbacks: 额外的回调函数列表，_可选_
		"""
		super().__init__(dpRoot)

		self.dCallbackCache: dict[str, list[VFLCallback]] = {}
		"""回调函数缓存，按方法名存储排序后的回调列表"""

		# 将自身作为第一个回调，确保自身的逻辑（如 `onTrainStepVars()`）最先执行（如果优先级相同）
		self.lCallbacks = [self, *ensureList(lCallbacks)]  # type: ignore[arg-type]
		"""所有回调函数的列表，包括自身"""

		# 初始化所有模块的回调
		for cb in self.lCallbacks:
			cb.onInitModule(self)

		self.criterion = nn.CrossEntropyLoss()
		"""损失函数"""

	# ============
	# Checkpoint 回调
	# ============

	@override
	def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
		"""保存模型检查点时的回调方法。

		Args:
			checkpoint: 要保存的检查点字典
		"""
		checkpoint['ns'] = self.ns
		for cb in self.lCallbacks:
			cb.onSaveCheckpoint(self, checkpoint)

	@override
	def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
		"""加载模型检查点时的回调方法。

		Args:
			checkpoint: 要加载的检查点字典
		"""
		self.ns = checkpoint['ns']
		for cb in self.lCallbacks:
			cb.onLoadCheckpoint(self, checkpoint)

	# ============
	# 核心工具方法
	# ============

	def _executeCallback(
		self, method: Callable[..., Any], v: dict[str, StepVars] | StepVars | None = None
	) -> None:
		"""执行指定的回调方法链。

		根据 `priority` 属性对回调进行排序并依次执行。

		Args:
			method: 回调方法
			v: 传递给回调的上下文变量，_可选_
		"""
		name = method.__name__
		if name not in self.dCallbackCache:
			# 仅在第一次调用该 Callback 时排序，缓存以优化性能
			self.dCallbackCache[name] = sorted(
				self.lCallbacks, key=lambda cb: getattr(getattr(cb, name), 'priority', 0), reverse=True
			)

		for cb in self.dCallbackCache[name]:
			# 获取实例绑定的方法
			bound = getattr(cb, name)
			if v is None:
				bound(self)
			else:
				bound(self, v)

	@override
	def configOptims(self) -> tuple[list[Optimizer], list[lr.LRScheduler]]:
		"""实现基类的 `configOptims()`，通过回调收集所有优化器。

		Returns:
			优化器列表和学习率调度器列表的元组
		"""
		lOptAll: list[Optimizer] = []
		lLRSAll: list[lr.LRScheduler] = []

		for cb in self.lCallbacks:
			res = cb.onConfigOptims()
			if res is None:
				continue

			cb.iOpt = len(lOptAll)
			cb.iLRS = len(lLRSAll)

			if isinstance(res, tuple):
				[lOpts, lLRSs] = res
			else:
				lOpts = res
				lLRSs = []

			lOptAll.extend(lOpts)
			lLRSAll.extend(lLRSs)

		return lOptAll, lLRSAll

	def getBtmOptims(self) -> list[LightningOptimizer]:
		"""获取底层模型的优化器列表。

		Returns:
			底层模型优化器的列表
		"""
		# 假设优化器顺序与 `getOptims()` 列表的前 N 个对应底层模型
		# 注意：这依赖于 `configOptims()` 中的添加顺序，需保持一致性
		return self.getOptims()[: len(self.lBtmNets)]

	def getTopOptim(self) -> LightningOptimizer:
		"""获取顶层模型的优化器。

		Returns:
			顶层模型的优化器
		"""
		return self.getOptims()[len(self.lBtmNets)]

	# ============
	# 训练阶段
	# ============

	@override
	def on_fit_start(self) -> None:
		self._executeCallback(VFLCallback.onFitStart)

	@override
	def on_train_epoch_start(self) -> None:
		self._executeCallback(VFLCallback.onTrainEpochStart)

	@override
	def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
		"""训练步骤。

		执行一个批次的训练，包括正向传播、损失计算和反向传播。

		Args:
			batch: 训练批次数据
			batch_idx: 批次索引
			dataloader_idx: 数据加载器索引，默认为 `0`
		"""
		v = StepVars(batch, batch_idx, dataloader_idx)
		self._executeCallback(VFLCallback.onTrainStepVars, v)

		# 1. 底部模型输入
		v.lBtmIns = v.images  # * onTrainStepVars() 解析 batch 得到了 images
		self._executeCallback(VFLCallback.onTrainBtmIns, v)

		# 2. 底部模型正向传播
		# 使用 zip(strict=True) 确保模型和输入数量匹配，防止静默错误
		v.lBtmOut = [net(x) for net, x in zip(self.lBtmNets, v.lBtmIns, strict=True)]
		self._executeCallback(VFLCallback.onTrainBtmOut, v)

		# 3. 切割层（梯度分离与梯度追踪）
		# 为顶部模型准备输入，同时保留梯度追踪用于回传给底部模型
		v.lTopIns = [ts.clone().detach().requires_grad_() for ts in v.lBtmOut]
		self._executeCallback(VFLCallback.onTrainTopIns, v)

		# 4. 顶部模型正向传播
		v.zTopOut = self.zTopNet(v.lTopIns)
		self._executeCallback(VFLCallback.onTrainTopOut, v)

		# 5. 损失计算
		v.loss = self.criterion(v.zTopOut, v.labels)
		self._executeCallback(VFLCallback.onTrainLoss, v)

		# 6. 反向传播

		# 6.1 清除梯度
		self.getTopOptim().zero_grad()
		[opt.zero_grad() for opt in self.getBtmOptims()]

		# 6.2 顶部模型反向传播
		self.manual_backward(v.loss)

		# 6.3 梯度传输（顶部模型 -> 底部模型）
		v.lTopInsGrad = [notNone(ti.grad) for ti in v.lTopIns]
		self._executeCallback(VFLCallback.onTrainTopInsGrad, v)

		# 6.4 底部模型反向传播
		for out, grad in zip(v.lBtmOut, v.lTopInsGrad, strict=True):
			self.manual_backward(out, grad)

		# 7. 优化器更新
		self.getTopOptim().step()
		[opt.step() for opt in self.getBtmOptims()]
		self._executeCallback(VFLCallback.onTrainOptimStep, v)

	@override
	def on_train_epoch_end(self) -> None:
		# 手动执行 Scheduler
		for lrs in ensureList(self.lr_schedulers()):
			lrs.step()  # pyright: ignore[reportCallIssue]
		self._executeCallback(VFLCallback.onTrainEpochEnd)

	# ============
	# 验证阶段
	# ============

	@override
	def on_validation_epoch_start(self) -> None:
		self._executeCallback(VFLCallback.onValEpochStart)

	@override
	def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
		"""验证步骤。

		执行一个批次的验证，包括正向传播和损失计算。

		Args:
			batch: 验证批次数据
			batch_idx: 批次索引
			dataloader_idx: 数据加载器索引，默认为 `0`
		"""
		d = {'Origin': StepVars(batch, batch_idx, dataloader_idx)}
		self._executeCallback(VFLCallback.onValStepVars, d)

		for v in d.values():
			v.lBtmIns = v.images
		self._executeCallback(VFLCallback.onValBtmIns, d)

		for v in d.values():
			v.lBtmOut = [net(x) for net, x in zip(self.lBtmNets, v.lBtmIns, strict=True)]
		self._executeCallback(VFLCallback.onValBtmOut, d)

		for v in d.values():
			v.lTopIns = v.lBtmOut
		self._executeCallback(VFLCallback.onValTopIns, d)

		for v in d.values():
			v.zTopOut = self.zTopNet(v.lTopIns)
		self._executeCallback(VFLCallback.onValTopOut, d)

		for v in d.values():
			v.loss = self.criterion(v.zTopOut, v.labels)
		self._executeCallback(VFLCallback.onValLoss, d)

	@override
	def on_validation_epoch_end(self) -> None:
		self._executeCallback(VFLCallback.onValEpochEnd)

	# ============
	# 测试阶段
	# ============

	@override
	def on_test_epoch_start(self) -> None:
		self._executeCallback(VFLCallback.onTestEpochStart)

	@override
	def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
		"""测试步骤。

		执行一个批次的测试，包括正向传播和损失计算。

		Args:
			batch: 测试批次数据
			batch_idx: 批次索引
			dataloader_idx: 数据加载器索引，默认为 `0`
		"""
		v = StepVars(batch, batch_idx, dataloader_idx)
		self._executeCallback(VFLCallback.onTestStepVars, v)

		v.lBtmIns = v.images
		self._executeCallback(VFLCallback.onTestBtmIns, v)

		v.lBtmOut = [net(x) for net, x in zip(self.lBtmNets, v.lBtmIns, strict=True)]
		self._executeCallback(VFLCallback.onTestBtmOut, v)

		v.lTopIns = v.lBtmOut
		self._executeCallback(VFLCallback.onTestTopIns, v)

		v.zTopOut = self.zTopNet(v.lTopIns)
		self._executeCallback(VFLCallback.onTestTopOut, v)

		v.loss = self.criterion(v.zTopOut, v.labels)
		self._executeCallback(VFLCallback.onTestLoss, v)

	@override
	def on_test_epoch_end(self) -> None:
		self._executeCallback(VFLCallback.onTestEpochEnd)
