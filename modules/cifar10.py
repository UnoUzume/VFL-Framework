"""CIFAR-10 数据集处理模块。

该模块提供了 CIFAR-10 数据集的加载、预处理和分发功能，
用于联邦学习场景下的图像分类任务。
"""

from collections.abc import Callable
from typing import override

from beartype import beartype as typechecker
from jaxtyping import jaxtyped
from torchvision.datasets import CIFAR10

from utils import define as de
from utils.common import Path, np
from utils.data import Dataset, TypedDataLoader, collate
from utils.module import DataHandler
from utils.vision import Transform, tf


class BaseDataset(Dataset[de.TUImageSample]):
	"""CIFAR-10 数据集的基本加载类。

	继承自 `Dataset` 类，用于加载和预处理 CIFAR-10 数据集。
	"""

	def __init__(self, dpRoot: Path | str, isTrain: bool) -> None:
		"""初始化实例。

		Args:
				dpRoot: 数据集存储路径
				isTrain: 是否为训练集
		"""
		dataset = CIFAR10(dpRoot, isTrain)
		self.images = np.array(dataset.data)
		"""NumPy 格式的图像数据数组"""
		self.labels = np.array(dataset.targets)
		"""NumPy 格式的标签数组"""
		self.transform = tf.ToImage()
		"""图像变换函数，用于将 NumPy 图像转换为 PyTorch 张量"""

	def __getitem__(self, idx: int) -> de.TUImageSample:
		"""获取指定索引的图像样本。

		Args:
				idx: 样本索引

		Returns:
				包含图像、标签和索引的样本对象
		"""
		image = self.transform(self.images[idx])
		label = self.labels[idx].item()
		return de.TUImageSample(image, label, idx)

	def __len__(self) -> int:
		"""获取数据集的样本数量。

		Returns:
				数据集的样本数量
		"""
		return len(self.images)


@jaxtyped(typechecker=typechecker)
def splitImage(data: de.TUImages, nParty: int) -> list[de.TUImages]:
	"""将批量图像数据分割为指定数量的子数据集。

	根据参与方数量，将图像数据沿宽度或宽度和高度维度进行分割，
	用于垂直联邦学习场景下的数据分发。

	Args:
			data: 批量图像数据，形状为 `(batch, 3, height, width)`
			nParty: 参与方数量，支持 `1`、`2`、`3`、`4`

	Returns:
			分割后的图像数据列表，每个元素对应一个参与方的数据

	Raises:
			ValueError: 当参与方数量不支持时抛出
	"""
	if nParty == 1:
		return [data]
	if nParty == 2:
		return [data[..., :16], data[..., 16:]]
	if nParty == 3:
		return [data[..., :10], data[..., 10:20], data[..., 20:]]
	if nParty == 4:
		return [
			data[..., :16, :16],  # 左上角
			data[..., 16:, :16],  # 左下角
			data[..., :16, 16:],  # 右上角
			data[..., 16:, 16:],  # 右下角
		]
	msg = f'不支持的 nParty: {nParty}'
	raise ValueError(msg)


@jaxtyped(typechecker=typechecker)
def getAugmentTrans(nParty: int = 1) -> list[Transform]:
	"""创建数据增强变换列表。

	根据参与方数量，创建适合不同图像尺寸的数据增强变换，
	包括随机裁剪、水平翻转和颜色抖动。

	Args:
			nParty: 参与方数量，默认值为 `1`

	Returns:
			数据增强变换列表
	"""
	lSize = [(32, 32), (32, 16), (32, 10), (16, 16)]
	return [
		tf.RandomCrop(lSize[nParty - 1], 3, padding_mode='reflect'),
		tf.RandomHorizontalFlip(),
		tf.ColorJitter(0.2, 0.2, 0.2),
	]


class Handler(DataHandler):
	"""CIFAR-10 数据集的处理程序类。

	继承自 `DataHandler` 类，提供了 CIFAR-10 数据集的准备、加载和变换功能。
	"""

	@property
	@override
	def dpPath(self) -> Path:
		return Path('data/datasets/cifar10')

	@override
	def prepare(self, dpData: Path) -> None:
		CIFAR10(dpData, True, download=True)
		CIFAR10(dpData, False, download=True)

	@override
	def getTrainDataset(self, dpData: Path) -> Dataset[de.TUImageSample]:
		return BaseDataset(dpData, True)

	@override
	def getValDataset(self, dpData: Path) -> Dataset[de.TUImageSample]:
		return BaseDataset(dpData, False)

	@override
	def getSplitFn(self) -> Callable[[de.TUImages, int], list[de.TUImages]]:
		return splitImage

	@override
	def getAugmentTrans(self, nParty: int = 1) -> list[Transform]:
		return getAugmentTrans(nParty)

	@override
	def getNormalTrans(self) -> list[Transform]:
		return []


if __name__ == '__main__':
	dpRoot = 'data/datasets/cifar10'
	CIFAR10(dpRoot, True, download=True)

	dsBase = BaseDataset(dpRoot, True)
	dlTrans = TypedDataLoader[de.TUImageBatch](dsBase, 4, collate_fn=collate)
	for batch in dlTrans:
		print(batch)
		break

	print('运行结束！')
