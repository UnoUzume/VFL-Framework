"""数据模块工具包，提供多模态数据处理和管理的核心组件。

该模块基于 Python 3.12 泛型与 PyTorch Lightning 生命周期设计，
定义了数据配置、处理、加载和张量变换的核心类和协议。
无论底层数据是图像、音频还是联邦学习中的 1D 异构特征，该模块都能提供强类型安全的支持。

编程哲学 (Solid Principles):
		- 单一职责原则 (SRP)：DataConfig 负责数据存储，DataHandler 负责读取逻辑，
			DataModule 负责调度 PyTorch Lightning 生命周期。
		- 依赖倒置原则 (DIP)：DataModule 不依赖具体的 CIFAR10 或 NUS-WIDE 业务类，
			而是仅仅依赖抽象的 `DataHandler` 协议。
"""

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast, overload, override

from main.module import DataModule

from . import define as de
from .common import Path, nn, tc
from .data import SizedDataset, TypedDataLoader
from .vision import Transform, createTrans

if TYPE_CHECKING:
	from torch.utils.data import Dataset

	from .misc import LoaderParams


@dataclass
class DataConfig:
	"""数据配置数据类 (DataClass)。

	用于集中管理数据集和数据加载器的各项超参数。
	使用 dataclass 可以极大地减少 boilerplate（样板代码），自动生成 __init__ 和 __repr__。

	编程知识 (Data Driven Configuration):
			将易变的超参数抽取到统一的配置类中，这使得我们在进行消融实验时，
			只需要修改外部的 YAML 文件或 Config 对象，而无需修改任何底层执行代码。
	"""

	sName: str
	"""数据集的唯一标识符名称，用于动态加载对应的 Handler 模块。"""
	nBatchSize: int = 32
	"""深度学习模型训练和推理时，每个批次包含的样本数量。"""
	nWorkers: int = 4
	"""多进程数据加载器 (DataLoader) 分配的子进程数，用于打破 Python GIL 限制加速 I/O。"""
	dpData: Path | None = None
	"""数据集存储的绝对或相对路径。若为 None，则回退使用 Handler 的默认路径。"""
	enableTrans: bool = True
	"""总开关：是否在设备转移 (Batch Transfer) 后启用自动数据变换 (Transform)。"""
	lAugmentTrans: list[Transform] | None = None
	"""数据增强变换列表（如随机裁剪、翻转等），仅在训练阶段 (Training) 启用。"""
	lNormalTrans: list[Transform] | None = None
	"""数据常规变换列表（如归一化等），在验证 (Validation) 和测试 (Testing) 阶段启用。"""

	def __post_init__(self) -> None:
		"""数据类初始化后的钩子方法 (Magic Method)。

		在 dataclass 的 `__init__` 执行完毕后自动调用，用于执行额外的属性校验
		和环境变量读取，支持通过终端环境变量动态覆盖代码默认配置。

		Args:
				无参数。

		Raises:
				ValueError: 若环境变量传入的值无法转换为整型时抛出。
		"""
		self.nBatchSize = int(os.getenv('N_BATCH_SIZE', self.nBatchSize))
		self.nWorkers = int(os.getenv('N_WORKERS', self.nWorkers))


class DataHandler[T_Sample: de.BaseSample[Any]](Protocol):
	"""泛型数据集处理协议 (Protocol)。

	该协议规范了数据集准备、获取、分割和数据变换等核心操作的契约 (Contract)。

	编程哲学 (Duck Typing 鸭子类型与接口隔离):
			借助 Protocol，外部业务模块不需要去继承一个庞大的 BaseHandler，
			只要它实现了这里的全部方法，DataModule 就会认为它是一个合法的 Handler。
	"""

	@property
	def dpPath(self) -> Path:
		"""获取当前数据集的默认持久化存储路径。"""
		...

	def prepare(self, dpData: Path) -> None:
		"""准备数据集。

		执行诸如网络下载、解压、预处理、或一维特征切片等一次性准备工作。

		Args:
				dpData: 数据集需要存储的目标路径。

		Returns:
				无返回值。

		Raises:
				RuntimeError: 如果关键依赖缺失或预处理失败。
		"""
		...

	def getTrainDataset(self, dpData: Path) -> SizedDataset[T_Sample]:
		"""获取训练数据集实例。

		Args:
				dpData: 预处理完毕的数据集路径。

		Returns:
				符合 SizedDataset 协议（拥有 __len__ 和 __getitem__）的训练数据集实例。

		Raises:
				FileNotFoundError: 如果目标路径下不存在预处理好的数据。
		"""
		...

	def getValDataset(self, dpData: Path) -> SizedDataset[T_Sample]:
		"""获取验证数据集实例。

		Args:
				dpData: 预处理完毕的数据集路径。

		Returns:
				符合 SizedDataset 协议的验证数据集实例。

		Raises:
				FileNotFoundError: 如果目标路径下不存在预处理好的数据。
		"""
		...

	def getCollateFn(self) -> Callable[[list[Any]], Any]:
		"""获取适用于当前模态的数据批次合并函数。

		编程知识 (Factory Method 工厂方法):
				将实例化具体 Batch 的逻辑下沉到具体的业务实现中。
				这使得外层框架无需关心此时打包的是 ImageBatch 还是 FeatureBatch。

		Returns:
				一个符合 PyTorch DataLoader 签名的 collate 函数。
		"""
		...

	def getSplitFn(self) -> Callable[[tc.Tensor, int], list[tc.Tensor]]:
		"""获取数据分割函数 (主要用于联邦学习的水平/垂直切分)。

		Returns:
				一个接收原始张量 (Tensor) 和参与方数量 (int)，并返回张量切片列表的函数。

		Raises:
				NotImplementedError: 如果具体的 Handler 不支持数据切分。
		"""
		...

	def getSplitBatchCls(self) -> Callable[[list[Any], tc.Tensor, tc.Tensor], Any]:
		"""获取切分后用于打包输出批次的数据类构造器。"""
		...

	def getAugmentTrans(self, nParty: int = 1) -> list[Transform]:
		"""获取适用于训练阶段的数据增强变换管线。

		Args:
				nParty: 联邦学习中的参与方数量，用于针对切片后尺寸适配增强策略（默认为 1）。

		Returns:
				包含一系列 Transform 的列表。对于 1D 特征，通常返回空列表 []。

		Raises:
				ValueError: nParty 数量非法时抛出。
		"""
		...

	def getNormalTrans(self) -> list[Transform]:
		"""获取适用于验证和推理阶段的数据常规变换管线。

		Returns:
				包含一系列常规 Transform（如 Normalize）的列表。

		Raises:
				无异常抛出。
		"""
		...


def _getHandler(name: str) -> DataHandler[Any]:
	"""通过动态反射机制加载并实例化业务数据集的 Handler。

	编程知识 (Reflection & Dynamic Import):
			这是构建高度可插拔插件系统的常用手段。系统无需在代码顶部 `import` 所有可能用到的数据集，
			仅在运行时根据配置字符串动态寻找并加载，大幅节省内存并降低模块耦合。

	Args:
			name: 业务数据集模块的文件名（位于 modules/ 目录下，如 'cifar10' 或 'nuswide'）。

	Returns:
			实例化后的具体 DataHandler 对象。

	Raises:
			ImportError: 找不到指定的模块文件时抛出。
			AttributeError: 指定的模块文件中没有定义 'Handler' 类时抛出。
	"""
	try:
		module = importlib.import_module(f'modules.{name}')
	except ImportError as err:
		msg = f'找不到数据集文件：modules/{name}.py'
		raise ImportError(msg) from err

	try:
		handler = module.Handler
	except AttributeError as err:
		msg = f"文件 {name}.py 中未定义 'Handler' 类"
		raise AttributeError(msg) from err

	return handler()


class BaseDataModule[T_Sample: de.BaseSample[Any], T_Batch: de.BaseBatch[Any]](DataModule):
	"""泛型基础数据模块类，封装 PyTorch Lightning 数据生命周期。

	负责连接 Config, Handler 和 Dataloader，将零散的数据组织为可供 GPU 吞吐的强类型 Batch。
	"""

	def __init__(self, config: DataConfig, handler: DataHandler[T_Sample] | None = None) -> None:
		"""初始化基础数据模块实例。

		Args:
				config: 全局数据配置对象，控制加载和增强行为。
				handler: 具体业务的数据处理器。若传入 None，则根据 config.sName 动态加载。

		Raises:
				ImportError: 若 handler 为空且动态加载指定的模块失败时抛出。
		"""
		super().__init__()

		self.cfg: DataConfig = config
		"""数据配置对象。"""
		self.hdlr: DataHandler[T_Sample] = handler or _getHandler(self.cfg.sName)
		"""数据处理器。负责具体多模态数据的读取与预处理逻辑。"""
		self.params: LoaderParams = {'num_workers': self.cfg.nWorkers, 'persistent_workers': True}
		"""传递给底层 DataLoader 的多进程控制参数字典。"""
		self.dpData: Path = self.cfg.dpData or self.hdlr.dpPath
		"""数据集实际采用的存储路径。"""
		self.fnCollate = self.hdlr.getCollateFn()
		"""样本合并函数。将散落的样本列表组装为统一的 Tensor Batch（通常由 create_collate 生成）。"""

		# 初始化增强变换管线
		self.lAugmentTrans: list[Transform] = (
			self.cfg.lAugmentTrans if self.cfg.lAugmentTrans is not None else self.hdlr.getAugmentTrans()
		)
		"""训练数据增强变换的步骤列表。"""

		self.tfAugment: Transform = createTrans(self.lAugmentTrans)
		"""组装完毕的统一训练增强变换管线函数。"""

		# 初始化常规变换管线
		self.lNormalTrans: list[Transform] = (
			self.cfg.lNormalTrans if self.cfg.lNormalTrans is not None else self.hdlr.getNormalTrans()
		)
		"""验证及测试数据的常规变换步骤列表。"""

		self.tfNormal: Transform = createTrans(self.lNormalTrans)
		"""组装完毕的统一验证常规变换管线函数。"""

		# 预先声明 Dataset 容器
		self.dsTrain: SizedDataset[T_Sample] | None = None
		"""训练集数据容器，在 setup 阶段赋值。"""
		self.dsVal: SizedDataset[T_Sample] | None = None
		"""验证集数据容器，在 setup 阶段赋值。"""

	@property
	def tfCurrent(self) -> Transform:
		"""动态获取当前所处阶段应当使用的数据变换函数。

		编程知识 (State-Dependent Behavior):
				通过判断 Lightning Trainer 的内部状态 (training 标志)，动态返回合适的 Transform，
				避免了手动在 train/eval 之间切换。

		Returns:
				当前应执行的数据张量变换函数 (Transform)。
		"""
		if not self.cfg.enableTrans:
			return nn.Identity()

		assert self.trainer is not None, 'Trainer 尚未绑定，无法判断生命周期状态。'

		if self.trainer.training:
			return self.tfAugment
		return self.tfNormal

	@override
	def prepare(self) -> None:
		"""数据准备钩子。

		在单节点上触发一次（不参与分布式多进程调度），常用于下载数据或提取特征。

		Args:
				无参数。

		Raises:
				RuntimeError: 如果数据底层准备或预处理失败。
		"""
		self.hdlr.prepare(self.dpData)

	@override
	def setup(self, stage: str) -> None:
		"""环境设置钩子。

		在每个 GPU/进程 上触发，负责将底层数据实例化挂载到内存中。

		Args:
				stage: 当前 Lightning 所处的生命周期阶段（如 'fit', 'test', 'predict'）。

		Raises:
				FileNotFoundError: 如果由于准备阶段失败导致数据文件丢失。
		"""
		self.dsTrain = self.hdlr.getTrainDataset(self.dpData)
		self.dsVal = self.hdlr.getValDataset(self.dpData)

	@override
	def getTrainLoader(self) -> TypedDataLoader[T_Batch]:
		"""获取训练数据加载器。

		Args:
				无参数。

		Returns:
				携带严谨泛型类型注解的 DataLoader 实例，能够触发 IDE 代码补全。

		Raises:
				RuntimeError: 若 setup 尚未执行导致 self.dsTrain 为空。
		"""
		assert self.dsTrain is not None, '必须在 setup 执行完毕后方可获取 DataLoader'

		# 编程哲学 (Type Casting 作为系统边界缓冲层):
		# 我们的系统内部使用 SizedDataset 协议获得了极高的灵活性，
		# 在调用第三方严苛 API 的边界处，显式转换类型以满足其签名要求。
		ds_train_casted = cast('Dataset[Any]', self.dsTrain)

		return TypedDataLoader[T_Batch](
			ds_train_casted, self.cfg.nBatchSize, True, collate_fn=self.fnCollate, **self.params
		)

	@override
	def getValLoader(self) -> TypedDataLoader[T_Batch]:
		"""获取验证数据加载器。

		Args:
				无参数。

		Returns:
				携带严谨泛型类型注解的验证 DataLoader 实例。

		Raises:
				RuntimeError: 若 setup 尚未执行导致 self.dsVal 为空。
		"""
		assert self.dsVal is not None, '必须在 setup 执行完毕后方可获取 DataLoader'
		ds_val_casted = cast('Dataset[Any]', self.dsVal)
		return TypedDataLoader[T_Batch](
			ds_val_casted, self.cfg.nBatchSize, False, collate_fn=self.fnCollate, **self.params
		)


class _TransDataModule[  # pyright: ignore[reportUnusedClass]
	T_Sample: de.BaseSample[Any],
	T_InBatch: de.BaseBatch[Any],
	T_OutBatch: de.BaseBatch[Any],
](BaseDataModule[T_Sample, T_InBatch]):
	"""泛型数据变换模块类。

	利用 Lightning 的 on_after_batch_transfer 钩子，将数据增强操作推迟到 GPU 上执行，
	大幅提升计算密集型 Transform (如大规模矩阵乘法或混合精度裁剪) 的吞吐效率。
	"""

	def __init__(
		self,
		config: DataConfig,
		out_batch_cls: Callable[[Any, tc.Tensor, tc.Tensor], T_OutBatch],
		handler: DataHandler[T_Sample] | None = None,
	) -> None:
		"""初始化变换数据模块实例。

		Args:
				config: 全局数据配置对象。
				out_batch_cls: 目标输出批次的强类型构造函数（如 de.TFImageBatch）。
				handler: 具体业务的数据处理器。若为空则动态加载。

		Raises:
				ImportError: 当动态加载模块失败时抛出。
		"""
		super().__init__(config, handler)

		self.out_batch_cls: Callable[[Any, tc.Tensor, tc.Tensor], T_OutBatch] = out_batch_cls
		"""输出批次数据类的工厂构造函数。"""

	@override
	def onAfterBatchTransfer(self, batch: T_InBatch, idxLoader: int) -> T_OutBatch:
		"""批次转移后的拦截钩子。

		此时的 Batch 已经处于目标设备 (如 CUDA) 上，在此处应用 tfCurrent 可以享受 GPU 加速。

		Args:
				batch: 从 DataLoader 弹出、已迁移至 GPU 的原始泛型批次。
				idxLoader: 当前 DataLoader 的序号 (支持多 DataLoader 并发)。

		Returns:
				经 Transform 变换并在类型上跃迁到 T_OutBatch (已处理批次) 的强类型对象。

		Raises:
				RuntimeError: 如果在 GPU 上的 Tensor 变换操作失败。
		"""
		new_data = self.tfCurrent(batch.data)
		return self.out_batch_cls(new_data, batch.label, batch.idx)


class SplitDataModule[
	T_Sample: de.BaseSample[Any],
	T_InBatch: de.BaseBatch[Any],
	T_OutBatch: de.BaseBatch[Any],
](BaseDataModule[T_Sample, T_InBatch]):
	"""泛型数据切分模块类 (面向联邦学习 VFL)。

	在 GPU 级别拦截 Batch 数据，对其进行张量维度的切割 (Spatial / Channel / Feature Dimension)，
	并分别对切片应用增强，以模拟不同参与方的异构数据流。
	"""

	def __init__(
		self,
		nParty: int,
		config: DataConfig,
		handler: DataHandler[T_Sample] | None = None,
	) -> None:
		"""初始化数据分割模块实例。

		Args:
				nParty: VFL 场景下的参与方总数。
				config: 全局数据配置对象。
				handler: 具体业务的数据处理器。若为空则动态加载。

		Raises:
				ImportError: 当动态加载模块失败时抛出。
		"""
		super().__init__(config, handler)

		self.nParty: int = nParty
		"""参与联邦学习的设备节点总数量。"""
		self.out_batch_cls = self.hdlr.getSplitBatchCls()
		"""用于接收分割后张量列表的批次数据类构造工厂。"""
		self.fnSplit: Callable[[tc.Tensor, int], list[tc.Tensor]] = self.hdlr.getSplitFn()
		"""将单一张量切割给不同 Party 的核心切分算法。"""

		# 如果未主动配置增强参数，根据参与方数量向下调取针对切片的专属增强策略
		if not self.cfg.lAugmentTrans:
			self.lAugmentTrans = self.hdlr.getAugmentTrans(self.nParty)

		self.tfAugment = createTrans(self.lAugmentTrans)

	def _onAfterBatchTransfer(self, batch: T_InBatch) -> T_OutBatch:
		"""切分逻辑的单批次核心处理。

		Args:
				batch: 尚未切分的原始设备端泛型批次。

		Returns:
				完成物理切分与单边 Transform 后，组装完成的多方共享批次。

		Raises:
				RuntimeError: 如果张量由于形状不兼容而切分失败。
		"""
		# 1. 对原始 Tensor 进行切割
		parts = self.fnSplit(batch.data, self.nParty)
		# 2. 对属于每个参与方的局部 Tensor 应用独立的 Transform（如不同的颜色抖动）
		parts = [self.tfCurrent(part) for part in parts]
		# 3. 将切片列表注入支持联邦切分协议的 Batch DataClass
		return self.out_batch_cls(parts, batch.label, batch.idx)

	@overload
	def onAfterBatchTransfer(self, batch: T_InBatch, idxLoader: int) -> T_OutBatch: ...

	@overload
	def onAfterBatchTransfer(self, batch: list[T_InBatch], idxLoader: int) -> list[T_OutBatch]: ...

	@override
	def onAfterBatchTransfer(
		self, batch: T_InBatch | list[T_InBatch], idxLoader: int
	) -> T_OutBatch | list[T_OutBatch]:
		"""支持单源或多源 DataLoader 组合输入的批次拦截钩子。

		Args:
				batch: 从一个或多个 DataLoader 弹出的目标批次。
				idxLoader: 当前所属 DataLoader 的序号。

		Returns:
				切分完成后的单一批次对象，或批次对象列表。

		Raises:
				RuntimeError: 切分过程或张量变换失败时抛出。
		"""
		if isinstance(batch, list):
			return [self._onAfterBatchTransfer(b) for b in batch]
		return self._onAfterBatchTransfer(batch)


__all__ = ['BaseDataModule', 'DataConfig', 'DataHandler', 'SplitDataModule']
