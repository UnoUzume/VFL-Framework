"""数据处理工具模块。

该模块提供了与数据加载、处理和转换相关的类和函数，主要用于图像数据的批处理、
数据集定义和图像分割等操作。
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any

from beartype import beartype as typechecker
from jaxtyping import jaxtyped
from torch.utils.data import DataLoader, Dataset

from . import define as de
from .common import tc
from .vision import tf


@jaxtyped(typechecker=typechecker)
def collate(batch: list[de.TUImageSample]) -> de.TUImageBatch:
	"""将图像样本列表转换为批次数据。

	Args:
		batch: 图像样本列表

	Returns:
		批次数据，包含图像、标签和索引
	"""
	images = tc.stack([item.image for item in batch])
	labels = tc.tensor([item.label for item in batch])
	indexs = tc.tensor([item.idx for item in batch])
	return de.TUImageBatch(images, labels, indexs)


class BaseDataset(Dataset[de.TUImageSample]):
	"""基础数据集类，用于加载和处理图像数据。

	该类是所有数据集的基类，提供了基本的图像加载和处理功能。
	"""

	def __init__(self) -> None:
		"""初始化实例。"""
		self.images: Any
		"""图像数据容器"""
		self.labels: Any
		"""标签数据容器"""
		self.transform = tf.ToImage()
		"""图像转换函数"""

	@jaxtyped(typechecker=typechecker)
	def __getitem__(self, idx: int) -> de.TUImageSample:
		"""获取指定索引的图像样本。

		Args:
			idx: 样本索引

		Returns:
			图像样本，包含图像、标签和索引
		"""
		image = self.transform(self.images[idx])
		label = self.labels[idx].item()
		return de.TUImageSample(image, label, idx)

	def __len__(self) -> int:
		"""获取数据集的大小。

		Returns:
			数据集的样本数量
		"""
		return len(self.images)


class TransDataset(Dataset[de.TFImageSample]):
	"""转换数据集类，用于对图像应用转换函数。

	该类包装了一个基础数据集，并对每个图像应用指定的转换函数。
	"""

	def __init__(self, dataset: BaseDataset, transform: Callable[[de.TUImage], de.TFImage]) -> None:
		"""初始化实例。

		Args:
			dataset: 基础数据集
			transform: 应用于图像的转换函数
		"""
		self.dataset = dataset
		"""基础数据集"""
		self.transform = transform
		"""图像转换函数"""

	def __getitem__(self, idx: int) -> de.TFImageSample:
		"""获取指定索引的转换后图像样本。

		Args:
			idx: 样本索引

		Returns:
			转换后的图像样本，包含图像、标签和索引
		"""
		[image, label, _] = self.dataset[idx]
		image = self.transform(image)
		return de.TFImageSample(image, label, idx)

	def __len__(self) -> int:
		"""获取数据集的大小。

		Returns:
			数据集的样本数量
		"""
		return len(self.dataset)


class SplitDataset(Dataset[de.TFSplitImageSample], ABC):
	"""分割数据集类，用于将图像分割成多个部分。

	该类包装了一个基础数据集，将每个图像分割成指定数量的部分，并对每个部分应用转换函数。
	"""

	def __init__(
		self,
		nParty: int,
		dataset: BaseDataset,
		transform: Callable[[de.TUImage], de.TFImage],
	) -> None:
		"""初始化实例。

		Args:
			nParty: 参与方数量，决定图像分割的份数
			dataset: 基础数据集
			transform: 应用于分割后图像的转换函数
		"""
		self.nParty = nParty
		"""参与方数量"""
		self.dataset = dataset
		"""基础数据集"""
		self.transform = transform
		"""图像转换函数"""

	def __getitem__(self, idx: int) -> de.TFSplitImageSample:
		"""获取指定索引的分割后图像样本。

		Args:
			idx: 样本索引

		Returns:
			分割后的图像样本，包含分割后的图像列表、标签和索引
		"""
		[image, label, _] = self.dataset[idx]
		parts = self.splitImage(image)
		parts = [self.transform(part) for part in parts]
		return de.TFSplitImageSample(parts, label, idx)

	def __len__(self) -> int:
		"""获取数据集的大小。

		Returns:
			数据集的样本数量
		"""
		return len(self.dataset)

	@abstractmethod
	def splitImage(self, data: de.TUImage) -> list[de.TUImage]:
		"""【**抽象方法**】将图像分割成多个部分。

		该方法需要在子类中重写，实现具体的图像分割逻辑。

		Args:
			data: 要分割的图像数据

		Returns:
			分割后的图像列表
		"""


class TypedDataLoader[T](DataLoader[Any]):
	"""泛型 DataLoader 类，用于指定返回的迭代器类型。

	该类继承自 PyTorch 的 `DataLoader`，但重写了 `__iter__` 方法的类型注解，
	确保 IDE 能够正确识别迭代器返回的类型。
	"""

	def __iter__(self) -> Iterator[T]:  # type: ignore[override]
		"""返回数据集的迭代器。

		Returns:
			指定类型的迭代器
		"""
		return super().__iter__()


__all__ = [
	'BaseDataset',
	'DataLoader',
	'Dataset',
	'SplitDataset',
	'TransDataset',
	'TypedDataLoader',
	'collate',
]
