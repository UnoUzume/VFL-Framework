"""定义了项目中使用的各种类型别名、数据结构和工具函数。

该模块包含了与图像数据处理、模型训练步骤相关的类型定义，
使用了 jaxtyping 进行类型标注，确保数据类型的一致性和安全性。
"""

from dataclasses import dataclass, field
from typing import NamedTuple

from beartype import beartype as typechecker
from jaxtyping import Float32, Int64, UInt8

from .common import np, tc

NUImage = UInt8[np.ndarray, '_h _w 3']
"""NumPy 格式的图像数据，形状为 `(height, width, 3)`，数据类型为 `uint8`。"""

TUImage = UInt8[tc.Tensor, '3 _h _w']
"""PyTorch 格式的图像数据，形状为 `(3, height, width)`，数据类型为 `uint8`。"""

TFImage = Float32[tc.Tensor, '3 _h _w']
"""PyTorch 格式的图像数据，形状为 `(3, height, width)`，数据类型为 `float32`。"""

TUImages = UInt8[TUImage, 'b']
"""PyTorch 格式的批量图像数据，形状为 `(batch, 3, height, width)`，数据类型为 `uint8`。"""

TFImages = Float32[TFImage, 'b']
"""PyTorch 格式的批量图像数据，形状为 `(batch, 3, height, width)`，数据类型为 `float32`。"""


@typechecker
class TUImageSample(NamedTuple):
	"""未处理的 PyTorch 图像样本。"""

	image: TUImage
	"""图像数据"""
	label: int
	"""图像标签"""
	idx: int
	"""图像索引"""


@typechecker
class TFImageSample(NamedTuple):
	"""已处理的 PyTorch 图像样本。"""

	image: TFImage
	"""图像数据"""
	label: int
	"""图像标签"""
	idx: int
	"""图像索引"""


@typechecker
class TFSplitImageSample(NamedTuple):
	"""已分割、已处理的 PyTorch 图像样本。"""

	image: list[TFImage]
	"""分割后的图像数据列表"""
	label: int
	"""图像标签"""
	idx: int
	"""图像索引"""


@typechecker
class TUImageBatch(NamedTuple):
	"""未处理的 PyTorch 图像批次。"""

	image: TUImages
	"""图像数据批次"""
	label: Int64[tc.Tensor, ' b']
	"""图像标签批次"""
	idx: Int64[tc.Tensor, ' b']
	"""图像索引批次"""


@typechecker
class TFImageBatch(NamedTuple):
	"""已处理的 PyTorch 图像批次。"""

	image: TFImages
	"""图像数据批次"""
	label: Int64[tc.Tensor, ' b']
	"""图像标签批次"""
	idx: Int64[tc.Tensor, ' b']
	"""图像索引批次"""


@typechecker
class TSplitImageBatch(NamedTuple):
	"""已分割、已处理的 PyTorch 图像批次。"""

	image: list[TUImages | TFImages]
	"""分割后的图像数据批次列表"""
	label: Int64[tc.Tensor, ' b']
	"""图像标签批次"""
	idx: Int64[tc.Tensor, ' b']
	"""图像索引批次"""


def defaultTensor() -> tc.Tensor:
	"""创建默认的 Tensor。

	用于初始化数据类中的 Tensor 字段，返回一个零张量。

	Returns:
		一个默认的零张量
	"""
	return tc.zeros(2, 3)


@dataclass
class StepVars:
	"""模型训练步骤中的变量集合。

	Args:
		batch: 已分割批次数据
		iBatchIdx: 批次索引
		iLoaderIdx: 数据加载器索引
		images: 批量图像，默认为空列表
		labels: 批量标签，默认使用 `defaultTensor()`
		indices: 批量索引，默认使用 `defaultTensor()`
		lBtmIns: 本地模型输入，默认为空列表
		lBtmOut: 本地模型输出，默认为空列表
		lTopIns: 全局模型输入，默认为空列表
		zTopOut: 全局模型输出，默认使用 `defaultTensor()`
		loss: 损失，默认使用 `defaultTensor()`
		lTopInsGrad: 全局模型输入梯度，默认为空列表
	"""

	batch: TSplitImageBatch
	"""已分割批次数据"""
	iBatchIdx: int
	"""批次索引"""
	iLoaderIdx: int
	"""数据加载器索引"""
	images: list[TFImages] = field(default_factory=list)
	"""批量图像"""
	labels: Int64[tc.Tensor, ' b'] = field(default_factory=defaultTensor)
	"""批量标签"""
	indices: Int64[tc.Tensor, ' b'] = field(default_factory=defaultTensor)
	"""批量索引"""
	lBtmIns: list[TFImages] = field(default_factory=list)
	"""本地模型输入"""
	lBtmOut: list[tc.Tensor] = field(default_factory=list)
	"""本地模型输出"""
	lTopIns: list[tc.Tensor] = field(default_factory=list)
	"""全局模型输入"""
	zTopOut: tc.Tensor = field(default_factory=defaultTensor)
	"""全局模型输出"""
	loss: tc.Tensor = field(default_factory=defaultTensor)
	"""损失"""
	lTopInsGrad: list[tc.Tensor] = field(default_factory=list)
	"""全局模型输入梯度"""
