"""定义了项目中使用的各种类型别名、数据结构和工具函数。

该模块包含了与图像数据处理、模型训练步骤相关的类型定义，
使用了 jaxtyping 进行类型标注，确保数据类型的一致性和安全性。
"""

from dataclasses import dataclass, field
from typing import Any

from beartype import beartype as typechecker
from jaxtyping import Float32, Int64, UInt8

from .common import np, tc

# ==========================================
# 1. 基础类型别名 (使用 Python 3.12 'type' 语法)
# ==========================================

# --- 图像类 ---
type NUImage = UInt8[np.ndarray, '_h _w _c']
"""NumPy 格式的图像数据，形状为 `(height, width, channel)`，数据类型为 `uint8`。"""
type TUImage = UInt8[tc.Tensor, '_c _h _w']
"""PyTorch 格式的图像数据，形状为 `(channel, height, width)`，数据类型为 `uint8`。"""
type TFImage = Float32[tc.Tensor, '_c _h _w']
"""PyTorch 格式的图像数据，形状为 `(channel, height, width)`，数据类型为 `float32`。"""
type TUImages = UInt8[TUImage, ' b']
"""PyTorch 格式的批量图像数据，形状为 `(batch, channel, height, width)`，数据类型为 `uint8`。"""
type TFImages = Float32[TFImage, ' b']
"""PyTorch 格式的批量图像数据，形状为 `(batch, channel, height, width)`，数据类型为 `float32`。"""

# --- 特征类 ---
type NFeature = Float32[np.ndarray, ' dim']
"""NumPy 格式的 1D 特征向量，形状为 `(dim,)`。"""
type TFeature = Float32[tc.Tensor, ' dim']
"""PyTorch 格式的 1D 特征向量，形状为 `(dim,)`。"""
type TFeatures = Float32[TFeature, ' b']
"""PyTorch 格式的 1D 特征批次，形状为 `(batch, dim)`。"""


# ==========================================
# 2. 泛型数据结构 (使用 Python 3.12 泛型语法)
# ==========================================


@typechecker
@dataclass(slots=True)
class BaseSample[T]:
	"""通用的样本基类"""

	data: T
	"""样本数据"""
	label: int
	"""样本标签"""
	idx: int
	"""样本索引"""


@typechecker
@dataclass(slots=True)
class BaseBatch[T]:
	"""通用的批次基类"""

	data: T
	"""批次数据"""
	label: Int64[tc.Tensor, ' b']
	"""批次标签，形状 `(batch,)`"""
	idx: Int64[tc.Tensor, ' b']
	"""批次索引，形状 `(batch,)`"""


# ==========================================
# 3. 具体业务别名 (保持向后兼容，方便上层业务调用)
# ==========================================


# --- 图像场景 ---
@typechecker
class TUImageSample(BaseSample[TUImage]):
	"""未处理的 PyTorch 图像单样本数据类。

	继承自泛型 `BaseSample`，其核心 `data` 属性被严格约束为 `TUImage`
	（即形状为 `(c, h, w)`，数据类型为 `uint8` 的张量）。
	通常用于 Dataset 的初始加载阶段，在应用 Transform 之前。
	"""


@typechecker
class TFImageSample(BaseSample[TFImage]):
	"""已处理的 PyTorch 图像单样本数据类。

	继承自泛型 `BaseSample`，其核心 `data` 属性被严格约束为 `TFImage`
	（即形状为 `(c, h, w)`，数据类型为 `float32` 的张量）。
	通常表示已经过归一化等 Transform 增强管线，可以直接喂给神经网络的图像样本。
	"""


@typechecker
class TFSplitImageSample(BaseSample[list[TFImage]]):
	"""已分割且已处理的 PyTorch 图像单样本数据类（面向联邦学习）。

	继承自泛型 `BaseSample`，其核心 `data` 属性被严格约束为 `list[TFImage]`
	（即包含多个 `float32` 张量的列表）。
	代表一张原始图像被切割（如空间划分或通道划分）后，分属于不同参与方的数据片段集。
	"""


@typechecker
class TUImageBatch(BaseBatch[TUImages]):
	"""未处理的 PyTorch 图像批次数据类。

	继承自泛型 `BaseBatch`，其核心 `data` 属性被严格约束为 `TUImages`
	（即形状为 `(b, c, h, w)`，数据类型为 `uint8` 的张量批次）。
	通常存在于 DataLoader 刚完成 collate，但尚未应用 GPU Transform 的转移阶段。
	"""


@typechecker
class TFImageBatch(BaseBatch[TFImages]):
	"""已处理的 PyTorch 图像批次数据类。

	继承自泛型 `BaseBatch`，其核心 `data` 属性被严格约束为 `TFImages`
	（即形状为 `(b, c, h, w)`，数据类型为 `float32` 的张量批次）。
	这是标准的深度学习输入批次，代表已经完全准备好进行网络前向传播的数据。
	"""


@typechecker
class TSplitImageBatch(BaseBatch[list[TUImages | TFImages]]):
	"""已分割的 PyTorch 图像批次数据类（面向联邦学习）。

	继承自泛型 `BaseBatch`，其核心 `data` 属性被约束为 `list[TUImages | TFImages]`
	（即包含多个图像批次张量的列表）。
	代表在 Batch 级别完成切分后的多方共享数据，列表的长度通常等于联邦参与方的数量 (nParty)。
	"""


# --- 特征场景 ---
@typechecker
class TFeatureSample(BaseSample[TFeature]):
	"""一维特征单样本数据类。

	继承自泛型 `BaseSample`，其核心 `data` 属性被严格约束为 `TFeature`
	（即形状为 `(dim,)` 的 1D 浮点张量）。
	适用于处理表格数据、NLP 句向量或类似 NUS-WIDE 的多模态拼接特征。
	"""


@typechecker
class TFeatureBatch(BaseBatch[TFeatures]):
	"""一维特征批次数据类。

	继承自泛型 `BaseBatch`，其核心 `data` 属性被严格约束为 `TFeatures`
	（即形状为 `(b, dim)` 的 2D 浮点张量）。
	这是输入给全连接层 (MLP) 或特征交互网络的基础批次格式。
	"""


@typechecker
class TSplitFeatureBatch(BaseBatch[list[TFeatures]]):
	"""已分割的一维特征批次数据类（面向垂直联邦学习 VFL）。

	继承自泛型 `BaseBatch`，其核心 `data` 属性被约束为 `list[TFeatures]`
	（即包含多个 2D 特征批次张量的列表）。
	代表全局特征在维度 (Dimension) 上被切分给各个参与方，各方持有相同样本的不同特征子集。
	"""


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
		raw: 原始批次数据
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

	raw: Any
	"""原始批次数据"""
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
	"""全局模型输入，属于隐私信息"""
	zTopOut: tc.Tensor = field(default_factory=defaultTensor)
	"""全局模型输出，属于隐私信息"""
	loss: tc.Tensor = field(default_factory=defaultTensor)
	"""损失，属于隐私信息"""
	lTopInsGrad: list[tc.Tensor] = field(default_factory=list)
	"""全局模型输入梯度，属于隐私信息"""
	lBtmOutGrad: list[tc.Tensor] = field(default_factory=list)
	"""本地模型输出梯度"""
