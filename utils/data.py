"""数据处理工具模块 (Python 3.12 泛型架构)。

该模块提供了与数据加载、处理和转换相关的泛型类和高阶函数，
具备极强的多模态扩展性，可无缝支持图像、1D 特征向量及文本数据的批处理与切分。

编程知识点：
		- 泛型 (Generics)：使用 Python 3.12 的 `[T]` 语法，让一套代码可以处理多种数据类型。
		- 协议 (Protocol)：静态的鸭子类型检查，提升类型安全性。
		- 抽象基类 (ABC)：定义子类必须遵守的“契约”。
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, Protocol

from beartype import beartype as typechecker
from jaxtyping import jaxtyped
from torch.utils.data import DataLoader, Dataset

from . import define as de
from .common import tc


def create_collate[T_Batch: de.BaseBatch[Any]](
	batch_cls: Callable[[Any, tc.Tensor, tc.Tensor], T_Batch],
) -> Callable[[list[de.BaseSample[Any]]], T_Batch]:
	"""生成特定数据模态的 Collate 函数的高阶函数。

	用于替换原生 DataLoader 默认的 collate_fn，确保在组合 Batch 时
	严格返回指定的泛型批次类（如 TUImageBatch, TUFeatureBatch），以激活 IDE 补全和类型检查。

	编程思想 (高阶函数 Higher-Order Function)：
			这是一个“返回函数的函数”。因为 DataLoader 的 collate_fn 只接受一个 batch 参数，
			我们通过外层函数将 `batch_cls` 提前“闭包”绑定进去，动态生成专属的打包函数。

	Args:
			batch_cls: 目标批次类的构造函数（通常直接传入 dataclass 类名）。

	Returns:
			一个强类型的 collate 函数，可直接传递给 DataLoader 的 collate_fn 参数。
	"""

	@jaxtyped(typechecker=typechecker)
	def _collate(batch: list[de.BaseSample[Any]]) -> T_Batch:
		# 将列表中每个样本的核心数据堆叠 (stack) 为一整个 Batch 张量
		data = tc.stack([item.data for item in batch])
		# 提取标签和索引，统一转换为 int64 类型的张量，供后续损失函数计算使用
		labels = tc.tensor([item.label for item in batch], dtype=tc.int64)
		indexs = tc.tensor([item.idx for item in batch], dtype=tc.int64)
		return batch_cls(data, labels, indexs)

	return _collate


class SizedDataset[T_Sample](Protocol):
	"""带长度映射的数据集协议。

	告诉类型检查器：只要一个类同时拥有 __getitem__ 和 __len__，
	它就是一个合法的 SizedDataset。

	编程哲学 (Duck Typing 鸭子类型)：
		“如果它走起来像鸭子，叫起来像鸭子，那它就是鸭子”。
		使用 Protocol，我们不需要强迫别人的 Dataset 继承特定的父类，
		只要它实现了指定的魔法方法，类型检查器就会放行，这极大地增强了代码的兼容性。

	编程知识 (Positional-Only Arguments 仅限位置参数):
		使用 `/` 告诉类型检查器，这里的参数不支持关键字调用。
		这样就可以完美忽略子类实现时到底用的是 `idx` 还是 `index`，
		只要它是第一个位置的 int 参数，就符合协议！
	"""

	def __getitem__(self, index: int, /) -> T_Sample:
		"""获取指定索引的数据样本。

		Args:
				index: 数据的整数索引。

		Returns:
				返回泛型 T_Sample 类型的样本实例。
		"""
		...

	def __len__(self) -> int:
		"""获取数据集的总长度。

		Returns:
				数据集包含的样本总数。
		"""
		...


class BaseDataset[T_Sample: de.BaseSample[Any]](Dataset[T_Sample], ABC):
	"""泛型基础数据集抽象类 (基于 tc.Tensor 规范)。

	它是所有具体业务数据集的基类。
	强制约定底层数据的流转单元必须是 PyTorch Tensor。
	"""

	def __init__(self) -> None:
		"""初始化基础数据集实例。

		为数据和标签预留容器。子类必须在其自己的 __init__ 结束前，
		将合法的 tc.Tensor 赋值给这两个属性。
		"""
		self.data: tc.Tensor | None = None
		self.labels: tc.Tensor | None = None

	@abstractmethod
	def make_sample(self, data: tc.Tensor, label: int, idx: int) -> T_Sample:
		"""【**抽象方法**】实例化具体的样本数据类。

		编程知识 (Template Method 模板方法模式)：
				基类定义了数据流转的骨架，但将具体的实例化逻辑推迟到子类中实现。

		Args:
				data: 单个核心数据 (严格要求为 Tensor)。
				label: 数据标签。
				idx: 全局索引。

		Returns:
				强类型的样本实例 (如 de.TUFeatureSample)。
		"""
		...

	def __getitem__(self, idx: int) -> T_Sample:
		"""根据索引获取并组装为一个强类型的样本实例。

		编程哲学 (Fail-Fast 快速失败 & Type Narrowing 类型收窄)：
				在这里使用 if 检查 None，不仅在运行时杜绝了“隐式空指针”带来的诡异 Bug，
				还向静态类型检查器证明了后续代码中的 self.data 绝对是 Tensor，从而消灭了 # type: ignore。

		Args:
				idx: 样本在数据集中的索引。

		Returns:
				组装好的强类型样本实例。

		Raises:
				RuntimeError: 当子类未能在初始化时正确挂载数据或标签张量时抛出。
		"""
		# 静态类型推断：Type Narrowing 发生在此处
		if self.data is None or self.labels is None:
			msg = (
				f'数据集 {self.__class__.__name__} 尚未挂载底层数据！\n'
				f'请确保在子类的 __init__ 结束前，将 tc.Tensor 赋值给 self.data 和 self.labels。'
			)
			raise RuntimeError(msg)

		data = self.data[idx]
		label = int(self.labels[idx])
		return self.make_sample(data, label, idx)

	def __len__(self) -> int:
		"""获取数据集的总长度。

		Returns:
				如果数据已挂载则返回总数，否则返回 0。
		"""
		return len(self.data) if self.data is not None else 0


# COMPAT: 半成品工具类
class _TransDataset[T_InSample: de.BaseSample[Any], T_OutSample: de.BaseSample[Any]](  # pyright: ignore[reportUnusedClass]
	Dataset[T_OutSample]
):
	"""泛型转换数据集类。

	包装基础数据集，在不丢失索引和标签的前提下，对数据应用 Transform 管线。

	编程知识 (Decorator Pattern 装饰器模式)：
			通过包装原有的 Dataset 并增加新行为（数据增强/转换），
			避免了为每一种转换逻辑创建庞大的继承树。
	"""

	def __init__(
		self,
		dataset: SizedDataset[T_InSample],  # 依赖反转：依赖协议而非具体实现
		transform: Callable[[tc.Tensor], Any],
		out_sample_cls: Callable[[Any, int, int], T_OutSample],
	) -> None:
		"""初始化转换数据集实例。

		Args:
				dataset: 实现了 SizedDataset 协议的基础数据集实例。
				transform: 应用于核心数据张量的转换函数管线。
				out_sample_cls: 输出样本类的构造函数。
		"""
		self.dataset = dataset
		self.transform = transform
		self.out_sample_cls = out_sample_cls

	def __getitem__(self, idx: int) -> T_OutSample:
		"""获取指定索引的数据，并应用转换操作。

		Args:
				idx: 样本在数据集中的索引。

		Returns:
				经过转换函数处理后的新类型样本实例。
		"""
		sample = self.dataset[idx]
		new_data = self.transform(sample.data)
		return self.out_sample_cls(new_data, sample.label, sample.idx)

	def __len__(self) -> int:
		"""获取被包装数据集的总长度。

		Returns:
				底层数据集的样本总数。
		"""
		return len(self.dataset)


# COMPAT: 半成品工具类
class _SplitDataset[T_InSample: de.BaseSample[Any], T_OutSample: de.BaseSample[Any]](  # pyright: ignore[reportUnusedClass]
	Dataset[T_OutSample], ABC
):
	"""泛型分割数据集抽象类 (面向联邦学习)。

	用于将原始数据（如图像四等分裁剪、或 1634 维特征切片）分割成多个部分，
	并分发给垂直联邦学习 (VFL) 的各个参与方。
	"""

	def __init__(
		self,
		nParty: int,
		dataset: SizedDataset[T_InSample],
		transform: Callable[[tc.Tensor], Any],
		out_sample_cls: Callable[[list[Any], int, int], T_OutSample],
	) -> None:
		"""初始化分割数据集实例。

		Args:
				nParty: 联邦学习的参与方数量。
				dataset: 实现了 SizedDataset 协议的基础数据集实例。
				transform: 独立应用于每一个分割后数据片段的转换函数。
				out_sample_cls: 输出样本类的构造函数 (其 data 属性需接收列表)。
		"""
		self.nParty = nParty
		self.dataset = dataset
		self.transform = transform
		self.out_sample_cls = out_sample_cls

	def __getitem__(self, idx: int) -> T_OutSample:
		"""获取指定索引的数据，进行切分并分别应用转换。

		Args:
				idx: 样本在数据集中的索引。

		Returns:
				包含切分片段列表的新类型样本实例。
		"""
		sample = self.dataset[idx]

		# 核心：调用子类重写的切分逻辑，入参固定为完整的 Tensor
		parts = self.splitData(sample.data)

		# 列表推导式：对切分后的每一份数据单独执行 Transform
		parts = [self.transform(part) for part in parts]

		return self.out_sample_cls(parts, sample.label, sample.idx)

	def __len__(self) -> int:
		"""获取被包装数据集的总长度。

		Returns:
				底层数据集的样本总数。
		"""
		return len(self.dataset)

	@abstractmethod
	def splitData(self, data: tc.Tensor) -> list[tc.Tensor]:
		"""【**抽象方法**】定义具体的数据切分逻辑。

		Args:
				data: 单个原始核心数据 (严格要求为 Tensor)。

		Returns:
				分割后的张量片段列表。列表长度通常应等于 self.nParty。
		"""
		...


class TypedDataLoader[T_Batch](DataLoader[Any]):
	"""泛型 DataLoader 类。

	通过重写迭代器的类型注解，完美解决 PyTorch 原生 DataLoader
	在 IDE 中丢失 Batch 类型信息、导致无法智能补全代码的缺陷。

	编程知识 (Type Erasure 类型擦除的修复)：
			原生的 DataLoader 在运行时会将数据泛型抹除为 Any。
			通过手动声明迭代器的返回类型为 `T_Batch`，我们在不改变底层 C++ 读取逻辑的前提下，
			重新为 Python 的静态分析器（如 Pylance）建立了类型联系。
	"""

	def __iter__(self) -> Iterator[T_Batch]:  # type: ignore[override]
		"""返回携带强类型信息的批次迭代器。

		Returns:
				具有确定类型 (T_Batch) 的迭代器，支持 IDE 代码补全。
		"""
		return super().__iter__()


__all__ = [
	'BaseDataset',
	'SizedDataset',
	'TypedDataLoader',
	'create_collate',
]
