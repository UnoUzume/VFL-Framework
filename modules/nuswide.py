"""NUS-WIDE 数据集模块

该模块提供了 NUS-WIDE 预处理特征数据集的加载、切分和分发功能。
由于底层数据是拼接好的 1634 维多模态特征向量，此模块展示了泛型架构
在脱离“图像”领域后，处理表格/向量数据时的极致简洁与强类型安全。

编程知识 (Separation of Concerns 关注点分离):
		该文件纯粹作为具体的业务实现者 (Concrete Implementor)，它只需满足
		`DataHandler` 协议的契约，而完全不需要关心 PyTorch DataLoader 是如何运作的，
		也不需要关心联邦学习的 Trainer 是如何调度的。
"""

from collections.abc import Callable
from typing import Any, override

from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from utils import define as de
from utils.common import Path, np, tc
from utils.data import BaseDataset, create_collate
from utils.module import DataHandler
from utils.vision import Transform


class NUSWIDEDataset(BaseDataset[de.TFeatureSample]):
	"""NUS-WIDE 数据集类

	继承自泛型的 `BaseDataset`，负责将预处理好的 `.npy` 矩阵加载至内存，
	并严格转换为 PyTorch Tensor，最后封装为强类型的 `TUFeatureSample`。
	"""

	def __init__(self, dpRoot: Path | str, isTrain: bool) -> None:
		"""初始化实例。

		编程哲学 (Fail-Fast 快速失败机制):
				在初始化阶段即执行文件的存在性校验，并立即执行 numpy 到 Tensor 的转换。
				这样如果在训练配置阶段（而非漫长的前向传播阶段）发生数据路径错误或格式错误，
				程序会瞬间崩溃并抛出异常，极大节省了 Debug 成本。

		Args:
				dpRoot: 数据集存储的根路径（通常指向预处理完毕的目录）。
				isTrain: 布尔值，标识当前实例用于加载训练集 (True) 还是测试集 (False)。

		Raises:
				FileNotFoundError: 当所需的特征或标签 `.npy` 文件缺失时抛出。
		"""
		super().__init__()
		split_name = 'Train' if isTrain else 'Test'
		dp_split = Path(dpRoot) / split_name

		feat_path = dp_split / 'global_features.npy'
		label_path = dp_split / 'global_labels.npy'

		if not feat_path.exists() or not label_path.exists():
			msg = f'预处理数据缺失，请先运行预处理脚本。检查路径：{dp_split}'
			raise FileNotFoundError(msg)

		# 满足 BaseDataset 契约：将底层数据强制转为 tc.Tensor 并挂载
		self.data = tc.from_numpy(np.load(feat_path)).float()
		self.labels = tc.from_numpy(np.load(label_path)).long()

	@override
	def make_sample(self, data: tc.Tensor, label: int, idx: int) -> de.TFeatureSample:
		return de.TFeatureSample(data=data, label=label, idx=idx)


@jaxtyped(typechecker=typechecker)
def _splitFeature(data: tc.Tensor, nParty: int) -> list[tc.Tensor]:
	"""将批量 1D 特征数据动态分割为指定数量的子特征集。

	针对 NUS-WIDE 的 1634 维异构特征，将特征维度（最末维度）均匀切分给各个参与方。
	当特征总维度无法被整除时，多出的余数维度会依次补偿给排在前面的参与方。

	编程知识 (Dynamic Tensor Slicing 动态切片算法):
			相比于硬编码特定尺寸的切片，该算法通过数学除法与余数分配，
			能够完美自适应 2 方、3 方、4 方甚至 8 方的垂直联邦学习 (VFL) 切分。

	Args:
			data: 输入的张量特征数据。可以是单样本 (dim,)，也可以是批次 (batch, dim)。
			nParty: 联邦学习的参与方总数量。

	Returns:
			包含 `nParty` 个切割后张量的列表。

	Raises:
			ValueError: 如果参与方数量小于 1 时抛出。
	"""
	if nParty < 1:
		msg = f'参与方数量 nParty 必须大于等于 1，当前为：{nParty}'
		raise ValueError(msg)

	if nParty == 1:
		return [data]

	total_dim = data.shape[-1]
	base_dim = total_dim // nParty
	remainder = total_dim % nParty

	splits = []
	start_idx = 0

	for i in range(nParty):
		# 如果当前索引 i 小于余数，则分担 1 个额外维度
		current_dim = base_dim + (1 if i < remainder else 0)
		end_idx = start_idx + current_dim
		# 使用省略号 `...` 保证无论输入是 1D 还是 2D(Batch)，均沿最后维度切片
		splits.append(data[..., start_idx:end_idx])
		start_idx = end_idx

	return splits


class Handler(DataHandler[de.TFeatureSample]):
	"""NUS-WIDE 数据集的核心处理程序

	隐式实现了 `DataHandler[de.TUFeatureSample]` 协议。
	负责对外提供数据集的路径、构造器、拆分算法及组装函数的实例。

	编程哲学 (Strategy Pattern 策略模式):
			Handler 就像是提供给 DataModule 的一个策略包。只要切换了 Handler，
			整个深度学习流水线就会无缝切换到新的数据集行为上。
	"""

	@property
	@override
	def dpPath(self) -> Path:
		return Path('data/datasets/NUS-WIDE/processed')

	@override
	def prepare(self, dpData: Path) -> None:
		train_feat = dpData / 'Train' / 'global_features.npy'
		if not train_feat.exists():
			msg = f'未找到数据：{train_feat}。请先运行 prepare_centralized.py'
			raise RuntimeError(msg)

	@override
	def getTrainDataset(self, dpData: Path) -> NUSWIDEDataset:
		return NUSWIDEDataset(dpData, isTrain=True)

	@override
	def getValDataset(self, dpData: Path) -> NUSWIDEDataset:
		return NUSWIDEDataset(dpData, isTrain=False)

	@override
	def getCollateFn(self) -> Callable[[list[Any]], de.TFeatureBatch]:
		return create_collate(de.TFeatureBatch)

	@override
	def getSplitFn(self) -> Callable[[tc.Tensor, int], list[tc.Tensor]]:
		return _splitFeature

	@override
	def getSplitBatchCls(self) -> Callable[[list[Any], tc.Tensor, tc.Tensor], de.TFeatureSplitBatch]:
		return de.TFeatureSplitBatch

	@override
	def getAugmentTrans(self, nParty: int = 1) -> list[Transform]:
		return []

	@override
	def getNormalTrans(self) -> list[Transform]:
		return []
