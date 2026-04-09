"""CIFAR-10 数据集处理模块。

该模块提供了 CIFAR-10 数据集的加载、预处理和分发功能，
用于联邦学习场景下的图像分类任务。
"""

from collections.abc import Callable
from typing import Any, override

from beartype import beartype as typechecker
from jaxtyping import jaxtyped
from torchvision.datasets import CIFAR10

from utils import define as de
from utils.common import F, Path, np, tc
from utils.data import BaseDataset, create_collate
from utils.module import DataHandler
from utils.vision import Transform, tf


class CIFAR10Dataset(BaseDataset[de.TUImageSample]):
	"""CIFAR-10 数据集类。

	继承自泛型的 `BaseDataset`，在初始化时一次性将数据挂载为 Tensor，
	消除 DataLoader 运行时的重复格式转换开销。
	"""

	def __init__(self, dpRoot: Path | str, isTrain: bool) -> None:
		"""初始化实例。

		Args:
				dpRoot: 数据集存储路径
				isTrain: 是否为训练集
		"""
		super().__init__()
		dataset = CIFAR10(dpRoot, train=isTrain)

		# CIFAR10 原生数据形状为 (N, H, W, C)，需转置为 PyTorch 规范的 (N, C, H, W)
		images = np.array(dataset.data).transpose((0, 3, 1, 2))

		# 满足 BaseDataset 契约：强行转换为 tc.Tensor 并挂载
		self.data = tc.from_numpy(images)
		self.labels = tc.tensor(dataset.targets, dtype=tc.int64)

	@override
	def make_sample(self, data: tc.Tensor, label: int, idx: int) -> de.TUImageSample:
		"""组装强类型的图像样本实例。

		Args:
				data: 单个核心图像数据 (tc.Tensor, 格式为 uint8)。
				label: 数据标签。
				idx: 样本索引。

		Returns:
				符合 TUImageSample 定义的强类型样本。
		"""
		return de.TUImageSample(data=data, label=label, idx=idx)


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
	if nParty == 8:
		big = F.interpolate(data, scale_factor=2, mode='nearest')
		return [
			big[..., :32, :16],  # 左上角
			big[..., :32, 16:32],
			big[..., :32, 32:48],
			big[..., :32, 48:],  # 右上角
			big[..., 32:, :16],  # 左下角
			big[..., 32:, 16:32],
			big[..., 32:, 32:48],
			big[..., 32:, 48:],  # 右下角
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
	dSize = {1: (32, 32), 2: (32, 16), 3: (32, 10), 4: (16, 16), 8: (32, 16)}
	tSize = dSize[nParty]
	tPad = (max(tSize[0] // 5, 3), max(tSize[1] // 5, 3))
	return [
		tf.RandomCrop(tSize, tPad, padding_mode='reflect'),
		tf.RandomHorizontalFlip(),
		tf.ColorJitter(0.2, 0.2, 0.2),
	]


class Handler(DataHandler[de.TUImageSample]):
	"""CIFAR-10 数据集的核心处理程序。

	实现 `DataHandler` 协议，提供 CIFAR-10 数据集的路径、构造器及组装函数的实例。
	"""

	@property
	@override
	def dpPath(self) -> Path:
		return Path('data/datasets/cifar10')

	@override
	def prepare(self, dpData: Path) -> None:
		CIFAR10(dpData, train=True, download=True)
		CIFAR10(dpData, train=False, download=True)

	@override
	def getTrainDataset(self, dpData: Path) -> CIFAR10Dataset:
		return CIFAR10Dataset(dpData, isTrain=True)

	@override
	def getValDataset(self, dpData: Path) -> CIFAR10Dataset:
		return CIFAR10Dataset(dpData, isTrain=False)

	@override
	def getCollateFn(self) -> Callable[[list[Any]], de.TUImageBatch]:
		return create_collate(de.TUImageBatch)

	@override
	def getSplitFn(self) -> Callable[[tc.Tensor, int], list[tc.Tensor]]:
		return splitImage

	@override
	def getAugmentTrans(self, nParty: int = 1) -> list[Transform]:
		return getAugmentTrans(nParty)

	@override
	def getNormalTrans(self) -> list[Transform]:
		return []


if __name__ == '__main__':
	from utils.data import TypedDataLoader

	dpRoot = 'data/datasets/cifar10'
	CIFAR10(dpRoot, train=True, download=True)

	dsBase = CIFAR10Dataset(dpRoot, isTrain=True)
	fn_collate = create_collate(de.TUImageBatch)
	dlTrans = TypedDataLoader[de.TUImageBatch](dsBase, batch_size=4, collate_fn=fn_collate)

	for batch in dlTrans:
		print(f'Batch 数据形状：{batch.data.shape}, 类型：{batch.data.dtype}')
		print(f'Batch 标签：{batch.label}')
		break

	print('运行结束！')
