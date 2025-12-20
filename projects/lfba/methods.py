"""LFBA 方法模块

本模块实现了 LFBA 联邦学习框架中的推理回调类，主要用于基于梯度的样本选择策略。
通过锚样本的梯度来识别并选择目标类和非目标类样本，从而提升联邦学习中的标签推断能力。
"""

from typing import override

from main.arch import BaseVFLArch
from main.callback import VFLCallback
from utils.collector import TensorCollector
from utils.common import np, tc
from utils.config import rng
from utils.define import StepVars


def getNearPos(
	tGrads: tc.Tensor, iAncPos: int, rTgt: float, rNon: float
) -> tuple[tc.Tensor, tc.Tensor]:
	"""获取与锚点梯度具有高相似度和低相似度的样本位置索引。

	Args:
		tGrads: 梯度张量，形状为 `(nSample, nFeature)`
		iAncPos: 锚点样本在梯度张量中的位置索引
		rTgt: 目标样本比例，用于确定选择多少个高相似度样本
		rNon: 非目标样本比例，用于确定选择多少个低相似度样本

	Returns:
		元组，包含两个张量：
		- `tTgtPos`: 与锚点梯度具有高相似度的样本位置索引张量
		- `tNonPos`: 与锚点梯度具有低相似度的样本位置索引张量
	"""
	tGradsL2 = tc.norm(tGrads, 2, 1)
	tAncL2 = tc.norm(tGrads[iAncPos], 2)
	# 计算相似度
	tSim = tc.div(tGrads @ tGrads[iAncPos], tGradsL2 * tAncL2)
	# 获取前 k 个最高相似度的索引
	_, tTgtPos = tSim.topk(int(rTgt * len(tSim)), 0, True)
	# 获取前 k 个最低相似度的索引
	_, tNonPos = tSim.topk(int(rNon * len(tSim)), 0, False)
	return tTgtPos, tNonPos


class LFBAInferCb(VFLCallback):
	"""LFBA 推理回调类

	该类用于在联邦学习训练过程中执行基于梯度的样本选择策略，
	通过锚样本的梯度来识别并选择目标类和非目标类样本。
	"""

	def __init__(self, iAncIdx: int, rTgt: float, rNon: float, rSel: float) -> None:
		"""初始化实例。

		Args:
			iAncIdx: 锚样本在训练数据集中的索引
			rTgt: 目标样本比例，用于确定选择多少个与锚样本相似的样本
			rNon: 非目标样本比例，用于确定选择多少个与锚样本不相似的样本
			rSel: 选择样本比例，用于确定每个批次中选择的样本数量
		"""
		super().__init__()

		self.iAncIdx = iAncIdx
		"""锚样本在训练数据集中的索引"""
		self.rTgt = rTgt
		"""目标样本比例"""
		self.rNon = rNon
		"""非目标样本比例"""
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

		m.ns.iAncIdx = dsTrain[self.iAncIdx][2]  #! 锚样本在原始训练集中的索引
		m.ns.iTgtLabel = dsFullTrain[m.ns.iAncIdx][1]  # type: ignore[index]  # 目标类标签
		m.logText.info(f'锚样本索引：{m.ns.iAncIdx}，目标类标签：{m.ns.iTgtLabel}')

	@override
	def onTrainEpochStart(self, m: BaseVFLArch) -> None:
		self.collector = TensorCollector('cpu')

	@override
	def onTrainTopInsGrad(self, m: BaseVFLArch, v: StepVars) -> None:
		self.collector.addBatch({'grads': v.lTopInsGrad[0], 'labels': v.labels, 'idxs': v.indices})

	@override
	def onTrainEpochEnd(self, m: BaseVFLArch) -> None:
		data = self.collector.read()
		aIdxs = data['idxs'].numpy()
		nSel = int(self.rSel * len(aIdxs))

		if m.current_epoch == 0:
			self.infer(m, data)

		# 选择一批目标类样本
		aTgtPos = np.flatnonzero(np.isin(aIdxs, m.ns.aTgtIdxs))
		tTgtGradsL2 = tc.norm(data['grads'][aTgtPos], 2, 1)
		_, tSel = tc.topk(tTgtGradsL2, nSel)
		m.ns.aDstIdxs = aIdxs[aTgtPos[tSel]]
		# m.ns.aDstIdxs = rng().choice(m.ns.aTgtIdxs, nSel, False)

		# 选择一批非目标类样本
		aOtherIdxs = np.setdiff1d(aIdxs, m.ns.aTgtIdxs, True)
		m.ns.aSrcIdxs = rng().choice(aOtherIdxs, nSel, False)
		# m.ns.aSrcIdxs = rng().choice(m.ns.aNonIdxs, nSel, False)

	def infer(self, m: BaseVFLArch, data: dict[str, tc.Tensor]) -> None:
		"""推理当前批次中的目标类和非目标类样本。

		基于锚样本的梯度，识别当前批次中与锚样本相似（目标类）和不相似（非目标类）的样本，
		并将这些样本的索引存储在模型的命名空间中，同时计算推理准确率并记录日志。

		Args:
			m: VFL 架构模型实例，用于访问和存储推理结果
			data: 当前批次的数据字典，包含以下键：
				- `'grads'`: 样本梯度张量
				- `'labels'`: 样本标签张量
				- `'idxs'`: 样本索引张量
		"""
		aIdxs = data['idxs'].numpy()

		# 获取当前批次中的锚样本位置
		iAnchorPos = np.flatnonzero(aIdxs == m.ns.iAncIdx).item()
		# 推理当前批次中的目标类和非目标类样本位置
		[tTgtPos, tNonPos] = getNearPos(data['grads'], iAnchorPos, self.rTgt, self.rNon)
		# 获取目标类和非目标类样本索引
		m.ns.aTgtIdxs = aIdxs[tTgtPos]
		m.ns.aNonIdxs = aIdxs[tNonPos]

		# 计算推理准确率
		fTgtRate = tc.eq(data['labels'][tTgtPos], m.ns.iTgtLabel).float().mean().item()
		fNonRate = tc.ne(data['labels'][tNonPos], m.ns.iTgtLabel).float().mean().item()
		m.logText.info(f'目标类推理准确率：{fTgtRate}, 非目标类推理准确率：{fNonRate}...')
