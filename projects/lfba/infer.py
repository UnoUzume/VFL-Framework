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
	tInfers = lClasses[tIndices]  # 类别标签可能不是数字，需要根据索引映射回原始标签
	return tInfers, tScores


class InferCb(VFLCallback):
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
		tMask = tc.isin(data['ids'], m.ns.tTgtIdxs)  #: 目标类样本的掩码
		tGradsL2 = tc.norm(data['grads'][tMask], p=2, dim=1)  #: 目标类样本的梯度 L2 范数
		_, tIndices = tc.topk(tGradsL2, k=nSel)
		m.ns.tDstIdxs = data['ids'][tMask][tIndices]

		# 方案二：随机选择
		# m.ns.tDstIdxs = rng().choice(m.ns.tTgtIdxs, size=nSel, replace=False)

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
		m.ns.tTgtIdxs = m.ns.tIDs[m.ns.tInfers == m.ns.iTgtLabel]
		m.ns.tVicIdxs = m.ns.tIDs[m.ns.tInfers != m.ns.iTgtLabel]
