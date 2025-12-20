"""视觉处理工具模块，提供图像变换相关的工具函数

本模块封装了图像预处理管道的创建和配置功能，支持构图域、光影域和统计域的变换操作。
"""

from collections.abc import Callable, Iterator
from typing import Any, Self, overload

import torchvision.transforms.v2 as tf

from .common import nn, tc

type Transform = nn.Module | Callable[..., Any]
"""类型别名，表示合法的变换对象"""


class DynamicCompose(tf.Compose):
	"""增强版 `torchvision.transforms.v2.Compose`，支持动态修改变换列表

	本类继承自 `v2.Compose`，保留了所有 GPU 加速和自动微分特性，
	同时实现了 `MutableSequence` 协议，使其像 Python `list` 一样支持
	加法拼接、索引访问、切片、插入和删除等操作。
	"""

	def __init__(self, transforms: list[Transform]) -> None:
		"""初始化实例。

		Args:
			transforms: 初始的变换对象列表
		"""
		super().__init__(transforms)
		self.transforms = transforms
		"""存储变换操作的内部列表"""

	def __add__(self, other: Self | list[Transform] | Transform) -> Self:
		"""实现加法运算符 (`+`) 以支持拼接操作，返回一个新的 `DynamicCompose` 实例。

		Args:
			other: 可以是另一个 `DynamicCompose` 实例、变换列表或单个变换对象

		Returns:
			一个新的 `DynamicCompose` 实例，包含合并后的变换列表
		"""
		new_transforms: list[Transform]

		if isinstance(other, tf.Compose):
			new_transforms = list(self.transforms) + list(other.transforms)
		elif isinstance(other, list):
			new_transforms = list(self.transforms) + other
		else:
			new_transforms = [*list(self.transforms), other]

		return self.__class__(new_transforms)

	def __iadd__(self, other: Self | list[Transform] | Transform) -> Self:
		"""实现原地加法运算符 (`+=`)，直接修改当前实例。

		Args:
			other: 要追加的变换对象，可以是 `DynamicCompose` 实例、变换列表或单个变换对象

		Returns:
			修改后的当前实例 (`Self`)
		"""
		if isinstance(other, tf.Compose):
			self.transforms.extend(other.transforms)
		elif isinstance(other, list):
			self.transforms.extend(other)
		else:
			self.transforms.append(other)

		return self

	@overload
	def __getitem__(self, index: int) -> Transform: ...

	@overload
	def __getitem__(self, index: slice) -> Self: ...

	def __getitem__(self, index: int | slice) -> Transform | Self:
		"""获取指定位置的变换对象，或通过切片生成新的 `DynamicCompose` 实例。

		Args:
			index: 整数索引或切片对象 (`slice`)

		Returns:
				当 `index` 为 `int` 时，返回对应的变换对象
				当 `index` 为 `slice` 时，返回包含子列表的新 `DynamicCompose` 实例
		"""
		# 如果是切片，返回一个新的 `DynamicCompose` 实例，保持类型一致性
		if isinstance(index, slice):
			return self.__class__(self.transforms[index])
		return self.transforms[index]

	def __setitem__(self, index: int, value: Transform) -> None:
		"""修改指定位置的变换对象。

		Args:
			index: 要修改的变换对象的索引位置
			value: 新的变换对象
		"""
		self.transforms[index] = value

	def __len__(self) -> int:
		"""返回当前实例中变换对象的数量。"""
		return len(self.transforms)

	def __iter__(self) -> Iterator[Transform]:
		"""返回一个迭代器，用于遍历当前实例中的变换对象。"""
		return iter(self.transforms)

	def insert(self, index: int, transform: Transform) -> None:
		"""在指定位置插入一个变换对象。

		Args:
			index: 插入位置的索引
			transform: 要插入的变换对象
		"""
		self.transforms.insert(index, transform)

	def insertBeforeToDtype(self, transform: Transform) -> None:
		"""在 `ToDtype()` 变换之前插入一个变换对象。

		本方法会遍历变换列表，找到第一个 `ToDtype` 变换并在此位置插入新变换。

		Args:
			transform: 要插入的变换对象
		"""
		for i, t in enumerate(self.transforms):
			if isinstance(t, tf.ToDtype):
				self.transforms.insert(i, transform)
				break

	def append(self, transform: Transform) -> None:
		"""在变换列表末尾追加一个变换对象。

		Args:
			transform: 要追加的变换对象
		"""
		self.transforms.append(transform)

	def pop(self, index: int = -1) -> Transform:
		"""移除并返回指定位置的变换对象。

		Args:
			index: 要移除的变换对象的索引位置，默认为最后一个 (`-1`)

		Returns:
			被移除的变换对象
		"""
		return self.transforms.pop(index)


def _getNormLayer(config: Transform | bool | None) -> list[Transform]:
	"""根据配置参数生成 Normalize 变换层列表。

	Args:
		config: 配置参数，可以是：
			- `True`: 使用 ImageNet 标准均值和方差
			- `False` 或 `None`: 不使用 Normalize 变换
			- 自定义的 `Normalize` 对象

	Returns:
		包含 Normalize 变换的列表（可能为空）
	"""
	if config is True:
		# 默认使用 ImageNet 标准均值和方差
		return [tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
	if not config:  # 处理 False 或 None
		return []
	# 用户传入了自定义的 Normalize 对象
	return [config]


def createTrans(
	lSpatialTrans: list[Transform] | None = None,
	lSignalTrans: list[Transform] | None = None,
	tfNormalize: Transform | bool | None = True,
	lLatentTrans: list[Transform] | None = None,
) -> DynamicCompose:
	"""创建完整的图像变换管道，包括构图域、光影域和统计域三个阶段。

	Args:
		lSpatialTrans: 构图域变换列表，如 `Resize`、`RandomCrop` 等，默认为 `None`
		lSignalTrans: 光影域变换列表，如 `ColorJitter`、`GaussianNoise` 等，默认为 `None`
		tfNormalize: 归一化配置，可以是：
			- `True`: 使用 ImageNet 标准均值和方差
			- `False` 或 `None`: 不使用归一化
			- 自定义的 `Normalize` 对象

			默认为 `True`
		lLatentTrans: 统计域变换列表，如 `RandomErasing` 等，默认为 `None`

	Returns:
		配置好的 `DynamicCompose` 实例，包含完整的变换管道
	"""
	return DynamicCompose(
		[
			# 阶段 1：构图域（Structural/Spatial Domain）
			tf.ToImage(),  # 将 Tensor、NDArray 或 PIL 图像转换为 Image，此操作不缩放数值
			*(lSpatialTrans or []),
			# 阶段 2：光影域（Photometric/Signal Domain）
			tf.ToDtype(tc.float32, True),  # 转换为 float32 并归一化到 [0, 1]
			*(lSignalTrans or []),
			# 阶段 3：统计域（Statistical/Latent Domain）
			*_getNormLayer(tfNormalize),  # Normalize 需要 float 类型
			*(lLatentTrans or []),
		]
	)


def createSpatialTrans(
	lSpatialTrans: list[Transform] | None = None,
) -> DynamicCompose:
	"""创建仅包含构图域变换的管道（如 `Resize`、`Crop`、`Flip`）。

	输入输出通常仍保持图像结构（`Image`/`uint8`）。

	Args:
		lSpatialTrans: 构图域变换列表，如 `Resize`、`RandomCrop` 等，默认为 `None`

	Returns:
		配置好的 `DynamicCompose` 实例，仅包含构图域变换
	"""
	return DynamicCompose([tf.ToImage(), *(lSpatialTrans or [])])


def createSignalTrans(
	lSignalTrans: list[Transform] | None = None,
) -> DynamicCompose:
	"""创建仅包含光影域变换的管道（如 `GammaCorrection`、`ColorJitter`、`GaussianNoise`）。

	注意：会自动包含 `ToDtype(tc.float32, True)` 变换，将数据转换为 `float32` 并归一化到 `[0, 1]`。

	Args:
		lSignalTrans: 光影域变换列表，如 `ColorJitter`、`GaussianNoise` 等，默认为 `None`

	Returns:
		配置好的 `DynamicCompose` 实例，仅包含光影域变换
	"""
	return DynamicCompose([tf.ToDtype(tc.float32, True), *(lSignalTrans or [])])


def createPostTrans(
	lSignalTrans: list[Transform] | None = None,
	tfNormalize: Transform | bool | None = True,
	lLatentTrans: list[Transform] | None = None,
) -> DynamicCompose:
	"""创建包含光影域和统计域变换的管道（跳过构图域变换）。

	本管道适用于已经完成空间变换的图像，直接处理像素值和分布。

	Args:
		lSignalTrans: 光影域变换列表，如 `ColorJitter`、`RandomGrayscale` 等，默认为 `None`
		tfNormalize: 归一化配置，可以是：
			- `True`: 使用 ImageNet 标准均值和方差
			- `False` 或 `None`: 不使用归一化
			- 自定义的 `Normalize` 对象

			默认为 `True`
		lLatentTrans: 统计域变换列表，如 `RandomErasing` 等，默认为 `None`

	Returns:
		配置好的 `DynamicCompose` 实例，包含光影域和统计域变换
	"""
	return DynamicCompose(
		[
			# 进入光影域
			tf.ToDtype(tc.float32, True),
			*(lSignalTrans or []),
			# 进入统计域
			*_getNormLayer(tfNormalize),
			*(lLatentTrans or []),
		]
	)


__all__ = [
	'DynamicCompose',
	'Transform',
	'createPostTrans',
	'createSignalTrans',
	'createSpatialTrans',
	'createTrans',
	'tf',
]
