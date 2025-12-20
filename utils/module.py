"""数据模块工具包，提供数据处理和管理的核心组件。

该模块定义了用于数据配置、数据处理、数据加载和数据变换的核心类和协议，
主要用于构建和管理深度学习任务中的数据流程。
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, override

from main.module import DataModule

from . import define as de
from .common import Path, nn, tc
from .data import Dataset, TypedDataLoader, collate
from .vision import Transform, createTrans

if TYPE_CHECKING:
	from .misc import LoaderParams


@dataclass
class DataConfig:
	"""数据配置类，用于配置数据集和数据加载器的参数。

	Args:
		dpData: 数据集存储路径
		nBatchSize: 数据批次大小，默认值为 `32`
		nWorkers: 数据加载器工作进程数，默认值为 `4`
		fnCollate: 样本合并函数，将图像样本列表转换为批次数据，默认使用 `collate()` 函数
		enableTrans: 是否启用自动数据变换，默认值为 `True`
		tfAugment: 数据增强变换函数，应用于训练数据，_可选_
		tfNormal: 数据常规变换函数，应用于验证和测试数据，_可选_
	"""

	dpData: Path
	"""数据集存储路径"""
	nBatchSize: int = 32
	"""数据批次大小"""
	nWorkers: int = 4
	"""数据加载器工作进程数"""
	fnCollate: Callable[[list[de.TUImageSample]], de.TUImageBatch] = collate
	"""样本合并函数，将图像样本列表转换为批次数据"""
	enableTrans: bool = True
	"""是否启用自动数据变换"""
	lAugmentTrans: list[Transform] = field(default_factory=list)
	"""数据增强变换列表，应用于训练数据"""
	lNormalTrans: list[Transform] = field(default_factory=list)
	"""数据常规变换列表，应用于验证和测试数据"""


class DataHandler(Protocol):
	"""数据集处理协议，定义了数据集操作的标准接口。

	该协议规范了数据集的准备、获取、分割和数据变换等核心操作，
	确保不同数据集实现具有一致的接口，便于在数据模块中统一使用。
	"""

	def prepare(self, dpData: Path) -> None:
		"""准备数据集，如下载、解压、预处理等操作。

		Args:
			dpData: 数据集存储路径
		"""
		...

	def getTrainDataset(self, dpData: Path) -> Dataset[de.TUImageSample]:
		"""获取训练数据集。

		Args:
			dpData: 数据集存储路径

		Returns:
			训练数据集实例，包含图像样本
		"""
		...

	def getValDataset(self, dpData: Path) -> Dataset[de.TUImageSample]:
		"""获取验证数据集。

		Args:
			dpData: 数据集存储路径

		Returns:
			验证数据集实例，包含图像样本
		"""
		...

	def getSplitFn(self) -> Callable[[de.TUImages, int], list[de.TUImages]]:
		"""获取数据分割函数。

		该函数用于将图像数据分割成多个部分，通常用于联邦学习场景下的数据分配。

		Returns:
			分割函数，接收图像数据和参与方数量，返回分割后的图像列表
		"""
		...

	def getAugmentTrans(self, nParty: int = 1) -> list[Transform]:
		"""获取数据增强变换列表。

		Args:
			nParty: 参与方数量，用于调整增强策略（默认值为 `1`）

		Returns:
			数据增强变换列表，应用于训练数据
		"""
		...

	def getNormalTrans(self) -> list[Transform]:
		"""获取数据常规变换列表。

		Returns:
			数据常规变换列表，应用于验证和测试数据
		"""
		...


class BaseDataModule(DataModule):
	"""基础数据模块类，用于管理数据集和数据加载器。"""

	def __init__(self, config: DataConfig, handler: DataHandler) -> None:
		"""初始化实例。

		Args:
			config: 数据配置对象，包含数据集和数据加载器的配置参数
			handler: 数据处理器，负责具体数据集的实现逻辑
		"""
		super().__init__()
		self.cfg = config
		"""数据配置对象"""
		self.hdlr = handler
		"""数据处理器"""
		self.params: LoaderParams = {'num_workers': self.cfg.nWorkers, 'persistent_workers': True}
		"""数据加载器参数"""

		if self.cfg.lAugmentTrans:
			self.lAugmentTrans = self.cfg.lAugmentTrans
			"""数据增强变换列表，应用于训练数据"""
		else:
			self.lAugmentTrans = self.hdlr.getAugmentTrans()

		self.tfAugment = createTrans(self.lAugmentTrans)
		"""数据增强变换函数，应用于训练数据"""

		if self.cfg.lNormalTrans:
			self.lNormalTrans = self.cfg.lNormalTrans
			"""数据常规变换列表，应用于验证和测试数据"""
		else:
			self.lNormalTrans = self.hdlr.getNormalTrans()

		self.tfNormal = createTrans(self.lNormalTrans)
		"""数据常规变换函数，应用于验证和测试数据"""

	@property
	def tfCurrent(self) -> Transform:
		"""当前应使用的数据变换函数"""
		# 如果禁用了自动数据变换，返回恒等变换
		if not self.cfg.enableTrans:
			return nn.Identity()

		# 根据训练状态选择变换函数
		assert self.trainer
		if self.trainer.training:
			return self.tfAugment
		return self.tfNormal

	@override
	def prepare(self) -> None:
		self.hdlr.prepare(self.cfg.dpData)

	@override
	def setup(self, stage: str) -> None:
		self.dsTrain = self.hdlr.getTrainDataset(self.cfg.dpData)
		"""训练数据集实例"""
		self.dsVal = self.hdlr.getValDataset(self.cfg.dpData)
		"""验证数据集实例"""

	@override
	def getTrainLoader(self) -> TypedDataLoader[de.TUImageBatch]:
		return TypedDataLoader[de.TUImageBatch](
			self.dsTrain, self.cfg.nBatchSize, True, collate_fn=self.cfg.fnCollate, **self.params
		)

	@override
	def getValLoader(self) -> TypedDataLoader[de.TUImageBatch]:
		return TypedDataLoader[de.TUImageBatch](
			self.dsVal, self.cfg.nBatchSize, False, collate_fn=self.cfg.fnCollate, **self.params
		)

	@override
	def transferBatchToDevice(
		self, batch: de.TUImageBatch, device: tc.device, dataloader_idx: int
	) -> de.TUImageBatch:
		[image, label, index] = batch
		image = image.to(device)
		label = label.to(device)
		index = index.to(device)
		return de.TUImageBatch(image, label, index)


class TransDataModule(BaseDataModule):
	"""数据变换模块类，用于在批次数据转移后应用变换"""

	def __init__(self, config: DataConfig, handler: DataHandler) -> None:
		"""初始化实例。

		Args:
			config: 数据配置对象，包含数据集和数据加载器的配置参数
			handler: 数据处理器，负责具体数据集的实现逻辑
		"""
		super().__init__(config, handler)

	@override
	def onAfterBatchTransfer(self, batch: de.TUImageBatch, dataloader_idx: int) -> de.TFImageBatch:
		[image, label, index] = batch
		image = self.tfCurrent(image)
		return de.TFImageBatch(image, label, index)


class SplitDataModule(BaseDataModule):
	"""数据分割模块类，用于在联邦学习场景下分割图像数据"""

	def __init__(self, nParty: int, config: DataConfig, handler: DataHandler) -> None:
		"""初始化实例。

		Args:
			nParty: 参与方数量，指定将图像数据分割成多少部分
			config: 数据配置对象，包含数据集和数据加载器的配置参数
			handler: 数据处理器，负责具体数据集的实现逻辑
		"""
		super().__init__(config, handler)
		self.nParty = nParty
		"""参与方数量"""
		self.fnSplit = self.hdlr.getSplitFn()
		"""数据分割函数"""

		if not self.cfg.lAugmentTrans:
			self.lAugmentTrans = self.hdlr.getAugmentTrans(self.nParty)
		self.tfAugment = createTrans(self.lAugmentTrans)

	@override
	def onAfterBatchTransfer(
		self, batch: de.TUImageBatch, dataloader_idx: int
	) -> de.TSplitImageBatch:
		[image, label, index] = batch
		parts = self.fnSplit(image, self.nParty)
		parts = [self.tfCurrent(part) for part in parts]
		return de.TSplitImageBatch(parts, label, index)


__all__ = [
	'BaseDataModule',
	'DataConfig',
	'DataHandler',
	'SplitDataModule',
	'TransDataModule',
]
