"""数据模块，定义了基于 PyTorch Lightning 的数据加载基类。

本模块封装了 `LightningDataModule`，规范了数据集的下载 (`prepare()`)、
加载 (`setup()`) 以及各个阶段 `DataLoader` 的获取接口。
"""

from abc import ABC, abstractmethod
from typing import Any, override

from utils.common import L, tc
from utils.data import TypedDataLoader

LoaderType = TypedDataLoader[Any] | list[TypedDataLoader[Any]]
"""数据加载器的泛型类型别名"""


class DataModule(L.LightningDataModule, ABC):
	"""数据模块基类，封装了数据预处理和加载流程。

	这是一个抽象基类，子类必须实现获取 `DataLoader` 的核心逻辑。
	该类通过代理方法（如 `train_dataloader()` -> `getTrainLoader()`）
	统一了接口风格，并提供了更灵活的 Hooks。
	"""

	def __init__(self) -> None:
		"""初始化实例。"""
		super().__init__()

	# ============
	# 生命周期 Hooks
	# ============

	@override
	def prepare_data(self) -> None:
		"""【Lightning Hook】准备数据。

		该方法在单进程中执行（即在多 GPU 训练时只在主进程执行一次）。
		代理调用自定义的 `prepare()` 方法。
		"""
		self.prepare()

	def prepare(self) -> None:
		"""下载并保存数据集。

		在此处执行只需要进行一次的操作，例如：
		1. 下载数据集。
		2. Tokenize 文本并保存到磁盘。

		注意：不要在此处设置 `self` 属性（多进程中状态不共享）。
		"""

	@override
	def setup(self, stage: str) -> None:
		"""【Lightning Hook】设置数据集。

		该方法在每个 GPU 进程中都会执行。
		用于加载数据、划分训练/验证集、构建 Dataset 对象。

		Args:
			stage: 当前执行阶段，可以是 `'fit'`、`'validate'`、`'test'` 或 `'predict'`。
		"""

	# ============
	# DataLoader 获取接口 (强制子类实现)
	# ============

	@override
	def train_dataloader(self) -> LoaderType:
		return self.getTrainLoader()

	@abstractmethod
	def getTrainLoader(self) -> LoaderType:
		"""【**抽象方法**】获取训练集 `DataLoader`。

		Returns:
			训练数据的 `DataLoader` 或 `DataLoader` 列表。
		"""

	@override
	def val_dataloader(self) -> LoaderType:
		return self.getValLoader()

	@abstractmethod
	def getValLoader(self) -> LoaderType:
		"""【**抽象方法**】获取验证集 `DataLoader`。

		Returns:
			验证数据的 `DataLoader` 或 `DataLoader` 列表。
		"""

	@override
	def test_dataloader(self) -> LoaderType:
		return self.getTestLoader()

	def getTestLoader(self) -> LoaderType:
		"""获取测试集 `DataLoader`。

		默认行为是复用验证集 `DataLoader`。如果需要独立的测试集，请在子类中重写。

		Returns:
			测试数据的 `DataLoader`。
		"""
		return self.getValLoader()

	@override
	def predict_dataloader(self) -> LoaderType:
		return self.getPredLoader()

	def getPredLoader(self) -> LoaderType:
		"""获取推理（预测）集 `DataLoader`。

		如果你的模型支持推理阶段，请在子类中重写此方法。

		Raises:
			NotImplementedError: 默认抛出异常，提示未实现。
		"""
		msg = 'getPredLoader() 未实现！'
		raise NotImplementedError(msg)

	# ============
	# 数据传输 Hooks
	# ============

	@override
	def transfer_batch_to_device(self, batch: Any, device: tc.device, dataloader_idx: int) -> Any:
		"""【Lightning Hook】将 Batch 数据移动到指定设备。

		Args:
			batch: 当前批次的数据。
			device: 目标设备。
			dataloader_idx: 数据加载器的索引。

		Returns:
			移动到设备后的数据批次。
		"""
		# 优先尝试调用自定义 Hook
		custom = self.transferBatchToDevice(batch, device, dataloader_idx)
		if custom is not None:
			return custom

		# 如果子类没有实现（返回 None），则调用父类（Lightning）的默认逻辑
		return super().transfer_batch_to_device(batch, device, dataloader_idx)

	# COMPAT: 基类需要保持灵活性
	def transferBatchToDevice(self, batch: Any, device: tc.device, idxLoader: int) -> Any:  # noqa: ANN401, ARG002, PLR6301
		"""自定义数据传输逻辑。

		如果您的 `DataLoader` 返回的是自定义数据结构（如字典、对象）而非标准 `Tensor`，
		请重写此方法手动将数据移动到 `device`。

		Args:
			batch: 当前批次的数据。
			device: 目标设备（CPU/GPU）。
			idxLoader: `DataLoader` 的索引。

		Returns:
			移动到设备后的数据。如果返回 `None`，则使用 Lightning 默认传输逻辑。
		"""
		return None

	@override
	def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
		"""【Lightning Hook】数据传输后的处理。

		Args:
			batch: 已移动到设备的数据。
			dataloader_idx: 数据加载器的索引。

		Returns:
			处理后的数据批次。
		"""
		# 优先尝试调用自定义 Hook
		custom = self.onAfterBatchTransfer(batch, dataloader_idx)
		if custom is not None:
			return custom

		# 如果子类没有实现（返回 None），则调用父类（Lightning）的默认逻辑
		return super().on_after_batch_transfer(batch, dataloader_idx)

	# COMPAT: 基类需要保持灵活性
	def onAfterBatchTransfer(self, batch: Any, idxLoader: int) -> Any:  # noqa: ANN401, ARG002, PLR6301
		"""数据传输后的增强或修改逻辑。

		在此处可进行 GPU 上的数据增强（比 CPU 快）。

		Args:
			batch: 已移动到设备的数据。
			idxLoader: `DataLoader` 的索引。

		Returns:
			处理后的数据。如果返回 `None`，则不做修改。
		"""
		return None
