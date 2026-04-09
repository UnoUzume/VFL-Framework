"""LFBA 标签推理模块，实现基于余弦相似度的标签推理功能"""

from typing import override

from beartype import beartype as typechecker
from jaxtyping import Float, Integer, jaxtyped

from main.arch import BaseVFLArch
from main.callback import VFLCallback
from utils.collector import TensorCollector
from utils.common import F, tc
from utils.config import rng
from utils.define import StepVars
from utils.misc import selectPerClass


def getNearPos(
	tGrads: tc.Tensor, iAncPos: int, rTgt: float, rVic: float
) -> tuple[tc.Tensor, tc.Tensor]:
	"""获取与锚点梯度具有高相似度和低相似度的样本位置索引。

	Args:
		tGrads: 梯度张量，形状为 `(nSamples, nFeatures)`
		iAncPos: 锚点样本在梯度张量中的位置索引
		rTgt: 目标类样本比例，用于确定选择多少个高相似度样本
		rVic: 受害类样本比例，用于确定选择多少个低相似度样本

	Returns:
		元组，包含两个张量：
		- `tTgtPos`: 与锚点梯度具有高相似度的样本位置索引张量
		- `tVicPos`: 与锚点梯度具有低相似度的样本位置索引张量
	"""
	tGradsL2 = tc.norm(tGrads, 2, 1)
	tAncL2 = tc.norm(tGrads[iAncPos], 2)
	# 计算相似度
	tSim = tc.div(tGrads @ tGrads[iAncPos], tGradsL2 * tAncL2)
	# 获取前 k 个最高相似度的索引
	_, tTgtPos = tSim.topk(int(rTgt * len(tSim)), 0, True)
	# 获取前 k 个最低相似度的索引
	_, tVicPos = tSim.topk(int(rVic * len(tSim)), 0, False)
	return tTgtPos, tVicPos


class LFBAInferCb(VFLCallback):
	"""LFBA 推理回调类

	该类用于在联邦学习训练过程中执行基于梯度的样本选择策略，
	通过锚样本的梯度来识别并选择目标类和非目标类样本。
	"""

	def __init__(self, iAncIdx: int, rTgt: float, rVic: float, rSel: float) -> None:
		"""初始化实例。

		Args:
			iAncIdx: 锚样本在训练数据集中的索引
			rTgt: 目标类样本比例，用于确定选择多少个与锚样本高相似度的样本
			rVic: 受害类样本比例，用于确定选择多少个与锚样本低相似度的样本
			rSel: 选择样本比例，用于确定每个批次中选择的样本数量
		"""
		super().__init__()

		self.iAncIdx = iAncIdx
		"""锚样本在训练数据集中的索引"""
		self.rTgt = rTgt
		"""目标类样本比例"""
		self.rVic = rVic
		"""受害类样本比例"""
		self.rSel = rSel
		"""选择样本比例"""

	# ============
	# 训练阶段
	# ============

	@override
	def onFitStart(self, m: BaseVFLArch) -> None:
		dsTrain = m.trainer.datamodule.dsTrain  # type: ignore[attr-defined]
		dsFullTrain = getattr(m.trainer.datamodule, 'dsFullTrain', dsTrain)  # type: ignore[attr-defined]

		# self.iAncIdx = 1096  #! 锚样本在实际训练集中的索引
		assert self.iAncIdx < len(dsTrain), '请选择正确的锚样本索引'

		m.ns.iAncId = dsTrain[self.iAncIdx].idx  # ! 锚样本在原始训练集中的索引
		m.ns.iTgtLabel = dsFullTrain[m.ns.iAncId].label  # type: ignore[index]  # 目标类标签
		m.logText.info(f'锚样本索引：{m.ns.iAncId}，目标类标签：{m.ns.iTgtLabel}')

	@override
	def onTrainEpochStart(self, m: BaseVFLArch) -> None:
		self.collector = TensorCollector()  # 用于在训练过程中收集数据

	@override
	def onTrainTopInsGrad(self, m: BaseVFLArch, v: StepVars) -> None:
		self.collector.addBatch({'grads': v.lTopInsGrad[0], 'labels': v.labels, 'ids': v.indices})

	@override
	def onTrainEpochEnd(self, m: BaseVFLArch) -> None:
		data = self.collector.read()  #: 训练过程中收集到的数据
		nSel = int(self.rSel * len(data['ids']))  #: 目标类样本的选择数量

		# 在第一个 Epoch 进行标签推理
		if m.current_epoch == 0:
			self.infer(m, data)

		# # 选择一批目标类样本

		# 方案一：根据梯度 L2 范数选择
		tMask = tc.isin(data['ids'], m.ns.tTgtIds)  #: 目标类样本的掩码
		tGradsL2 = tc.norm(data['grads'][tMask], p=2, dim=1)  #: 目标类样本的梯度 L2 范数
		_, tIndices = tc.topk(tGradsL2, k=nSel)
		m.ns.tDstIds = data['ids'][tMask][tIndices]

		# 方案二：随机选择
		# indices = tc.randperm(len(m.ns.tTgtIds))[:nSel]
		# m.ns.tDstIds = m.ns.tTgtIds[indices]

		# # 选择一批非目标类样本
		# aOtherIds = np.setdiff1d(aIds, m.ns.aTgtIds, True)
		# m.ns.aSrcIds = rng().choice(aOtherIds, nSel, False)
		# m.ns.aSrcIds = rng().choice(m.ns.aVicIds, nSel, False)

	def infer(self, m: BaseVFLArch, data: dict[str, tc.Tensor]) -> None:
		"""推理当前批次中的目标类和非目标类样本。

		基于锚样本的梯度，识别当前批次中与锚样本相似（目标类）和不相似（非目标类）的样本，
		并将这些样本的索引存储在模型的命名空间中，同时计算推理准确率并记录日志。

		Args:
			m: VFL 架构模型实例，用于访问和存储推理结果
			data: 当前批次的数据字典，包含以下键：
				- `'grads'`: 样本梯度张量
				- `'labels'`: 样本标签张量
				- `'ids'`: 样本索引张量
		"""
		# 获取当前批次中的锚样本位置
		[lAnchorPos] = tc.nonzero(data['ids'] == m.ns.iAncId, as_tuple=True)
		iAnchorPos = int(lAnchorPos.item())
		# 推理当前批次中的目标类和非目标类样本位置
		[tTgtPos, tVicPos] = getNearPos(data['grads'], iAnchorPos, self.rTgt, self.rVic)
		# 获取目标类和非目标类样本索引
		m.ns.tTgtIds = data['ids'][tTgtPos]
		m.ns.tVicIds = data['ids'][tVicPos]

		# 计算推理准确率
		fTgtRate = tc.eq(data['labels'][tTgtPos], m.ns.iTgtLabel).float().mean().item()
		fVicRate = tc.ne(data['labels'][tVicPos], m.ns.iTgtLabel).float().mean().item()
		m.logText.info(f'目标类推理准确率：{fTgtRate}, 受害类推理准确率：{fVicRate}...')


@jaxtyped(typechecker=typechecker)
def inferAllClasses(
	tUnknown: Float[tc.Tensor, 'nSamples nDims'],
	*,
	tAuxData: Float[tc.Tensor, 'nAuxs nDims'],
	tAuxLabels: Integer[tc.Tensor, ' nAuxs'],
) -> tuple[tc.Tensor, tc.Tensor]:
	"""通过余弦相似度推理全部样本的标签。

	Args:
		tUnknown: 无标签样本的嵌入或梯度，形状为 `[nSamples, nDim]`
		tAuxData: 辅助样本的嵌入或梯度，形状为 `[nAuxs, nDim]`
		tAuxLabels: 辅助样本的标签，形状为 `[nAuxs]`

	Returns:
		推理的标签张量和相似度分数张量
	"""
	# 计算类别中心
	lClasses = tAuxLabels.unique()
	lCenters = [tAuxData[tAuxLabels == i].mean(dim=0) for i in lClasses]
	tCenters = tc.stack(lCenters)

	# 对未知样本和类别中心进行 L2 归一化
	tNormUnknown = F.normalize(tUnknown, p=2, dim=1)  # * [nSamples, nDim]
	tNormCenters = F.normalize(tCenters, p=2, dim=1)  # * [nClasses, nDim]

	# 计算余弦相似度矩阵
	tSimilarityMat = tc.matmul(tNormUnknown, tNormCenters.t())  # * [nSamples, nClasses]

	# 进行标签推理
	tScores, tIndices = tc.max(tSimilarityMat, dim=1)
	tInfers = lClasses[tIndices]  # ! 类别标签可能不是数字，需要根据索引映射回原始标签
	return tInfers, tScores


class LFBAInferAllCb(VFLCallback):
	"""推理回调类，用于在 VFL 训练过程中进行标签推理"""

	def __init__(self, rSel: float) -> None:
		"""初始化实例。

		Args:
			rSel: 目标类样本占全部样本的选择比例
		"""
		super().__init__()
		self.rSel = rSel
		"""目标类样本占全部样本的选择比例"""

	@override
	def onFitStart(self, m: BaseVFLArch) -> None:
		m.ns.iTgtLabel = 1  # ! 目标类标签暂时固定为 1
		m.logText.info(f'目标类标签：{m.ns.iTgtLabel}')

		# 每个类别选择 10 个样本作为辅助样本
		[lIDs, _] = selectPerClass(lLabels=m.module.dsTrain.labels, nPerClass=10, rng=rng())
		self.lIDs = lIDs

	@override
	def onTrainEpochStart(self, m: BaseVFLArch) -> None:
		self.collector = TensorCollector()  # 用于在训练过程中收集数据

	@override
	def onTrainTopInsGrad(self, m: BaseVFLArch, v: StepVars) -> None:
		# TODO(UnoUzume): 每个参与者可以各自进行
		self.collector.addBatch(
			{'grads': tc.cat(v.lTopInsGrad[:4], dim=1), 'labels': v.labels, 'ids': v.indices}
		)

	@override
	def onTrainEpochEnd(self, m: BaseVFLArch) -> None:
		data = self.collector.read()  #: 训练过程中收集到的数据
		nSel = int(self.rSel * len(data['ids']))  #: 目标类样本的选择数量

		# 在第一个 Epoch 进行标签推理
		if m.current_epoch == 0:
			self.infer(m, data)

		# # 选择一批目标类样本

		# 方案一：根据梯度 L2 范数选择
		tMask = tc.isin(data['ids'], m.ns.tTgtIds)  #: 目标类样本的掩码
		tGradsL2 = tc.norm(data['grads'][tMask], p=2, dim=1)  #: 目标类样本的梯度 L2 范数
		_, tIndices = tc.topk(tGradsL2, k=nSel)
		m.ns.tDstIds = data['ids'][tMask][tIndices]

		# 方案二：随机选择
		# m.ns.tDstIds = rng().choice(m.ns.tTgtIds, size=nSel, replace=False)

	def infer(self, m: BaseVFLArch, data: dict[str, tc.Tensor]) -> None:
		"""进行标签推理，并存储到实例命名空间中。

		Args:
			m: VFL 架构实例
			data: 训练过程中收集到的数据
		"""
		# 生成辅助样本的掩码
		tMask = tc.isin(data['ids'], tc.as_tensor(self.lIDs).to(data['ids']))
		# 对全部样本进行推理
		tInfers, _ = inferAllClasses(
			tUnknown=data['grads'], tAuxData=data['grads'][tMask], tAuxLabels=data['labels'][tMask]
		)
		# 计算并记录推理准确率
		m.logText.info(f'准确率：{tc.eq(tInfers, data["labels"]).float().mean():.2%}')

		# 根据 ID 对推理结果进行排序
		m.ns.tIDs, tIndices = tc.sort(data['ids'])
		m.ns.tInfers = tInfers[tIndices]

		# 根据推理结果将 ID 分为目标类和受害类两组
		m.ns.tTgtIds = m.ns.tIDs[m.ns.tInfers == m.ns.iTgtLabel]
		m.ns.tVicIds = m.ns.tIDs[m.ns.tInfers != m.ns.iTgtLabel]
