"""CINIC-10 数据集处理模块。

该模块提供了 CINIC-10 数据集的加载、预处理和分发功能，
用于联邦学习场景下的图像分类任务。
"""

from collections.abc import Callable
from typing import override

from beartype import beartype as typechecker
from jaxtyping import jaxtyped
from torchvision.datasets import ImageFolder

from utils import define as de
from utils.common import Path
from utils.data import Dataset
from utils.module import DataHandler
from utils.vision import createTrans, tf


class BaseDataset(Dataset[de.TUImageSample]):
	"""CINIC-10 数据集的基本加载类。

	继承自 `Dataset` 类，用于加载和预处理 CINIC-10 数据集。
	"""

	def __init__(self, dpRoot: Path | str, isTrain: bool) -> None:
		"""初始化实例。

		Args:
				dpRoot: 数据集存储路径
				isTrain: 是否为训练集
		"""
		split = 'train' if isTrain else 'valid'
		self.dataset = ImageFolder(Path(dpRoot) / split)
		"""数据集对象，用于加载图像和标签"""
		self.transform = tf.ToImage()
		"""图像变换函数，用于将 NumPy 图像转换为 PyTorch 张量"""

	def __getitem__(self, idx: int) -> de.TUImageSample:
		"""获取指定索引的图像样本。

		Args:
				idx: 样本索引

		Returns:
				包含图像、标签和索引的样本对象
		"""
		image, label = self.dataset[idx]
		image = self.transform(image)
		return de.TUImageSample(image, label, idx)

	def __len__(self) -> int:
		"""获取数据集的样本数量。

		Returns:
				数据集的样本数量
		"""
		return len(self.dataset)


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
def getAugmentTrans(nParty: int = 1) -> Callable[[de.TUImage], de.TFImage]:
	"""创建数据增强变换函数。

	根据参与方数量，创建适合不同图像尺寸的数据增强变换，
	包括随机裁剪、水平翻转和颜色抖动。

	Args:
			nParty: 参与方数量，默认值为 `1`

	Returns:
			数据增强变换函数
	"""
	lSize = [(32, 32), (32, 16), (32, 10), (16, 16)]
	return createTrans(
		[
			tf.RandomCrop(lSize[nParty - 1], 3, padding_mode='reflect'),
			tf.RandomHorizontalFlip(),
			tf.ColorJitter(0.2, 0.2, 0.2),
		]
	)


class Handler(DataHandler):
	"""CINIC-10 数据集的处理程序类。

	继承自 `DataHandler` 类，提供了 CINIC-10 数据集的准备、加载和变换功能。
	"""

	@override
	def prepare(self, dpData: Path) -> None:
		pass

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
	def getAugmentTrans(self, nParty: int = 1) -> Callable[[de.TUImage], de.TFImage]:
		return getAugmentTrans(nParty)

	@override
	def getNormalTrans(self) -> Callable[[de.TUImage], de.TFImage]:
		return createTrans()
