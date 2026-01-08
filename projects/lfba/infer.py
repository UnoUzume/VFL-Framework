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
	tAuxiliary: Float[tc.Tensor, 'nAuxs nDims'],
	lAuxLabels: Integer[tc.Tensor, ' nAuxs'],
	method: str = 'cos',
) -> tuple[tc.Tensor, tc.Tensor]:
	"""使用原型网络 (Prototypical) 思想，推断所有无标签样本的类别。

	适用于特征 (Embeddings) 和梯度 (Gradients)。

	Args:
		tUnknown: 无标签样本的特征或梯度 (N_samples, hidden_dim)
		tAuxiliary: 辅助样本的特征或梯度 (N_aux, hidden_dim)
		lAuxLabels: 辅助样本的标签 (N_aux, )
		method: 计算相似度的方法，可选 'cos' 或 'dot'
	"""
	# --- 1. 计算每个类别的中心/原型 (Class Prototypes) ---
	# 获取所有唯一的类别
	unique_classes = tc.unique(lAuxLabels).sort()[0]

	# 存储每个类别的中心向量
	# 形状：(n_classes, hidden_dim)
	prototypes = []

	for cls in unique_classes:
		# 筛选出属于当前类 cls 的辅助样本
		mask = lAuxLabels == cls
		cls_data = tAuxiliary[mask]

		# 计算均值作为该类的“中心” (Anchor)
		# 这里的 mean(0) 就是论文中提到的 z_agt (Eq. 5)
		cls_center = cls_data.mean(dim=0)
		prototypes.append(cls_center)

	tPrototypes = tc.stack(prototypes)

	# --- 2. 预处理 (归一化) ---
	if method == 'cos':
		# 对未知样本和原型都做 L2 归一化
		tUnknown_norm = F.normalize(tUnknown, p=2, dim=1)
		tPrototypes_norm = F.normalize(tPrototypes, p=2, dim=1)
	else:
		# 如果是纯点积，则不归一化 (但通常不推荐，不稳定)
		tUnknown_norm = tUnknown
		tPrototypes_norm = tPrototypes

	# --- 3. 计算相似度矩阵 ---
	# 矩阵乘法：(N_samples, dim) @ (dim, n_classes) -> (N_samples, n_classes)
	# result[i][j] 表示第 i 个样本与第 j 个类别的相似度
	similarity_matrix = tc.matmul(tUnknown_norm, tPrototypes_norm.t())

	# --- 4. 推断类别 (Argmax) ---
	# 在类别维度 (dim=1) 上找最大值的索引
	# values: 每个样本的最大相似度分数 (置信度)
	# indices: 推断出的类别索引 (0 ~ n_classes-1)
	confidence_scores, predicted_indices = tc.max(similarity_matrix, dim=1)

	# 如果类别标签不是 0,1,2... 而是具体的 label 值，需要映射回来
	predicted_labels = unique_classes[predicted_indices]

	return predicted_labels, confidence_scores


class InferCb(VFLCallback):
	def __init__(self, rTgt: float, rVic: float, rSel: float, fPrec: float = 0.9) -> None:
		super().__init__()

		self.rTgt = rTgt
		"""目标类样本占该类比例"""
		self.rVic = rVic
		"""受害类样本占该类比例"""
		self.rSel = rSel
		"""选择样本比例"""

		assert 0.0 <= fPrec <= 1.0, '期望准确率必须在 [0, 1] 范围内'
		self.fPrec = fPrec

	@override
	def onFitStart(self, m: BaseVFLArch) -> None:
		m.ns.iTgtLabel = 1
		m.ns.iVicLabel = 2
		m.logText.info(f'目标类标签：{m.ns.iTgtLabel}，受害类标签：{m.ns.iVicLabel}')
		m.logText.info(f'期望准确率：{self.fPrec:.2%}')

		[lIDs, _] = selectPerClass(m.module.dsTrain.labels, 10, rng())
		self.lIDs = lIDs

	@override
	def onTrainEpochStart(self, m: BaseVFLArch) -> None:
		self.collector = TensorCollector()

	@override
	def onTrainTopInsGrad(self, m: BaseVFLArch, v: StepVars) -> None:
		self.collector.addBatch(
			{'grads': tc.cat(v.lTopInsGrad[:4], 1), 'labels': v.labels, 'ids': v.indices}
		)

	@override
	def onTrainEpochEnd(self, m: BaseVFLArch) -> None:
		data = self.collector.read()
		nSel = int(self.rSel * len(data['ids']))

		if m.current_epoch == 0:
			self.infer(m, data)

		# 选择一批目标类样本
		tMask = tc.isin(data['ids'], m.ns.tTgtIdxs)
		tTgtGradsL2 = tc.norm(data['grads'][tMask], 2, 1)
		_, tSel = tc.topk(tTgtGradsL2, nSel)
		m.ns.tDstIdxs = data['ids'][tMask][tSel]
		# m.ns.aDstIdxs = rng().choice(m.ns.aTgtIdxs, nSel, False)

	def infer(self, m: BaseVFLArch, data: dict[str, tc.Tensor]) -> None:
		"""执行受控精度的推理逻辑"""
		tMask = tc.isin(data['ids'], tc.as_tensor(self.lIDs).to(data['ids']))
		tPreds, _ = inferAllClasses(data['grads'], data['grads'][tMask], data['labels'][tMask])
		m.logText.info(f'准确率：{tc.eq(tPreds, data["labels"]).float().mean():.2%}')

		tIDs_, tIndices = tc.sort(data['ids'])
		tPreds_ = tPreds[tIndices]

		m.ns.tIDs = tIDs_
		m.ns.tPreds = tPreds_

		m.ns.tTgtIdxs = m.ns.tIDs[m.ns.tPreds == m.ns.iTgtLabel]
		m.ns.tVicIdxs = m.ns.tIDs[m.ns.tPreds != m.ns.iTgtLabel]
