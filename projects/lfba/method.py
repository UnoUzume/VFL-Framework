"""LFBA 方法模块

本模块实现了 LFBA 联邦学习框架中的推理回调类，主要用于基于梯度的样本选择策略。
通过锚样本的梯度来识别并选择目标类和非目标类样本，从而提升联邦学习中的标签推断能力。
"""

from typing import Any, override

from main.arch import BaseVFLArch
from main.callback import VFLCallback
from utils.collector import TensorCollector
from utils.common import np, tc
from utils.config import rng
from utils.define import StepVars
from utils.vision import tf


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
			self._verify_inference_integrity(m)

		# 选择一批目标类样本
		aTgtPos = np.flatnonzero(np.isin(aIdxs, m.ns.aTgtIdxs))
		tTgtGradsL2 = tc.norm(data['grads'][aTgtPos], 2, 1)
		_, tSel = tc.topk(tTgtGradsL2, nSel)
		m.ns.aDstIdxs = aIdxs[aTgtPos[tSel]]
		# m.ns.aDstIdxs = rng().choice(m.ns.aTgtIdxs, nSel, False)

		# 选择一批非目标类样本
		# aOtherIdxs = np.setdiff1d(aIdxs, m.ns.aTgtIdxs, True)
		# m.ns.aSrcIdxs = rng().choice(aOtherIdxs, nSel, False)
		m.ns.aSrcIdxs = rng().choice(m.ns.aVicIdxs, nSel, False)

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
		[tTgtPos, tVicPos] = getNearPos(data['grads'], iAnchorPos, self.rTgt, self.rVic)
		# 获取目标类和非目标类样本索引
		m.ns.aTgtIdxs = aIdxs[tTgtPos]
		m.ns.aVicIdxs = aIdxs[tVicPos]

		# 计算推理准确率
		fTgtRate = tc.eq(data['labels'][tTgtPos], m.ns.iTgtLabel).float().mean().item()
		fVicRate = tc.ne(data['labels'][tVicPos], m.ns.iTgtLabel).float().mean().item()
		m.logText.info(f'目标类推理准确率：{fTgtRate}, 受害类推理准确率：{fVicRate}...')

	def _verify_inference_integrity(self, m: BaseVFLArch) -> None:
		"""验证推理结果的集合完整性和统计正确性。

		Args:
				m: VFL 模型实例
		"""
		aTgtIdxs = m.ns.aTgtIdxs
		aVicIdxs = m.ns.aVicIdxs

		# [无重复] 检查单个集合内部是否存在重复索引
		assert len(aTgtIdxs) == len(np.unique(aTgtIdxs)), '错误：目标类集合包含重复索引！'
		assert len(aVicIdxs) == len(np.unique(aVicIdxs)), '错误：受害类集合包含重复索引！'

		# [无交集] 检查目标类和受害类是否有重叠
		intersection = np.intersect1d(aTgtIdxs, aVicIdxs, True)
		assert len(intersection) == 0, '错误：目标类和受害类存在重叠！'


class CheatInferCb(LFBAInferCb):
	"""可控精度的作弊推理回调类 (Controlled Oracle)

	该类允许用户指定推理的准确率（Precision）。
	它通过在真实的“目标类”和“非目标类”样本之间进行受控的混淆（Mixing），
	来模拟特定性能的攻击效果。

	主要用于：
	1. 评估攻击方法在不同推理质量下的鲁棒性。
	2. 模拟弱分类器或强分类器的行为。
	"""

	def __init__(
		self, iAncIdx: int, rTgt: float, rVic: float, rSel: float, fTgtPrec: float = 0.9
	) -> None:
		super().__init__(iAncIdx, rTgt, rVic, rSel)
		assert 0.0 <= fTgtPrec <= 1.0, '目标类期望准确率必须在 [0, 1] 范围内'
		self.fTgtPrec = fTgtPrec

	@override
	def onFitStart(self, m: BaseVFLArch) -> None:
		super().onFitStart(m)
		m.logText.info(f'[ControlCheat] 目标类期望准确率：{self.fTgtPrec:.2%}')

	@override
	def infer(self, m: BaseVFLArch, data: dict[str, tc.Tensor]) -> None:
		"""执行受控精度的推理逻辑"""
		aIdxs = data['idxs'].numpy()
		aLabels = data['labels'].numpy()
		iTgtLabel = m.ns.iTgtLabel

		# 1. 获取真实的样本分类 (Ground Truth)
		# aTrueTgt: 所有真实属于目标类的索引
		aTrueTgtIdxs = aIdxs[aLabels == iTgtLabel]
		# aTrueVic: 所有真实属于受害类的索引
		aTrueVicIdxs = aIdxs[aLabels != iTgtLabel]

		nTgtTotal = len(aTrueTgtIdxs)
		nVicTotal = len(aTrueVicIdxs)

		# 2. 构建目标类集合 (m.ns.aTgtIdxs)
		# 计算需要多少个“真阳性”样本 (True Positive)
		nTP = int(nTgtTotal * self.fTgtPrec)
		# 计算需要多少个“假阳性”样本 (False Positive)
		nFP = nTgtTotal - nTP
		assert nFP <= nVicTotal, '假阳性样本数不能超过受害类样本数'

		# 随机抽取
		# 从真目标中选 nTP 个
		sel_TP = rng().choice(aTrueTgtIdxs, nTP, replace=False)
		# 从真非目标中选 nFP 个 (作为噪声)
		sel_FP = rng().choice(aTrueVicIdxs, nFP, replace=False)

		temp = np.concatenate([sel_TP, sel_FP])
		rng().shuffle(temp)
		m.ns.aTgtIdxs = rng().choice(temp, int(len(aIdxs) * self.rTgt), replace=False)

		# 3. 构建受害类集合 (m.ns.aVicIdxs)
		temp = np.setdiff1d(aIdxs, m.ns.aTgtIdxs, True)
		rng().shuffle(temp)
		m.ns.aVicIdxs = rng().choice(temp, int(len(aIdxs) * self.rVic), replace=False)

		# 4. 验证并记录实际准确率
		mask_tgt_pred = np.isin(aIdxs, m.ns.aTgtIdxs)
		tPredTgtLabels = aLabels[mask_tgt_pred]
		fRealTgtPrec = np.mean(tPredTgtLabels == iTgtLabel).item()

		# 验证受害类
		mask_vic_pred = np.isin(aIdxs, m.ns.aVicIdxs)
		tPredVicLabels = aLabels[mask_vic_pred]
		fRealVicPrec = np.mean(tPredVicLabels != iTgtLabel).item()

		m.logText.info('[ControlCheat] 构造完成。')
		m.logText.info(f'    - 目标类集合大小：{len(m.ns.aTgtIdxs)} (真值总数：{nTgtTotal})')
		m.logText.info(f'    - 样本选择比例：{self.rTgt:.2%} -> {len(m.ns.aTgtIdxs) / len(aIdxs):.2%}')
		m.logText.info(f'    - 目标类期望准确率：{self.fTgtPrec:.2%} -> {fRealTgtPrec:.2%}')
		m.logText.info(f'    - 受害类集合大小：{len(m.ns.aVicIdxs)} (真值总数：{nVicTotal})')
		m.logText.info(f'    - 样本选择比例：{self.rVic:.2%} -> {len(m.ns.aVicIdxs) / len(aIdxs):.2%}')
		m.logText.info(f'    - 受害类期望准确率：{1 - self.fTgtPrec:.2%} -> {fRealVicPrec:.2%}')


class NonOverlappingSampler:
	def __init__(self, all_ids: np.ndarray, all_labels: np.ndarray) -> None:
		self.all_ids = all_ids
		self.all_labels = all_labels
		# 创建一个布尔掩码，记录哪些 ID 已经被使用了
		# False 代表可用，True 代表已用
		self.used_mask = np.zeros(len(all_ids), dtype=bool)

	def _get_available_pool(self, condition_indices: np.ndarray) -> np.ndarray:
		"""从满足特定条件 (condition_indices) 的样本中，筛选出目前还未被使用的样本"""
		# 既满足特定条件（如属于某类），且 mask 为 False（未被占用）
		# 这里的 condition_indices 是下标
		available = []
		for idx in condition_indices:
			if not self.used_mask[idx]:
				available.append(idx)
		return np.array(available)

	def generate_list(
		self, target_class_label: int, fRatioOfAll: float, accuracy: float
	) -> np.ndarray:
		"""生成单个列表，并锁定相关资源"""
		# 1. 定义目标类和非目标类的 全局下标
		target_indices = np.where(self.all_labels == target_class_label)[0]
		noise_indices = np.where(self.all_labels != target_class_label)[0]

		# 2. 计算目标数量
		list_len = int(len(self.all_labels) * fRatioOfAll)

		if list_len == 0:
			return np.array([])

		n_correct = int(list_len * accuracy)  # 需要抽取的 TP
		n_wrong = list_len - n_correct  # 需要抽取的 FP

		# 3. 获取当前“剩余可用”的资源池
		# 可用的正样本池 (Available TP Pool)
		avail_true_pool_idx = self._get_available_pool(target_indices)
		# 可用的负样本池 (Available FP Pool)
		avail_noise_pool_idx = self._get_available_pool(noise_indices)

		# 4. 资源充足性检查
		if len(avail_true_pool_idx) < n_correct:
			msg = f'资源耗尽：类 [{target_class_label}] 需要 {n_correct} 个正样本，但只剩 {len(avail_true_pool_idx)} 个可用。'
			raise ValueError(msg)

		if len(avail_noise_pool_idx) < n_wrong:
			msg = f'资源耗尽：类 [{target_class_label}] 需要 {n_wrong} 个噪声样本，但只剩 {len(avail_noise_pool_idx)} 个可用。'
			raise ValueError(msg)

		# 5. 执行采样（抽取的是下标 indices）
		selected_tp_idx = np.random.choice(avail_true_pool_idx, n_correct, replace=False)
		selected_fp_idx = np.random.choice(avail_noise_pool_idx, n_wrong, replace=False)

		# 6. 更新全局占用掩码
		self.used_mask[selected_tp_idx] = True
		self.used_mask[selected_fp_idx] = True

		# 7. 转换回 ID 并打乱
		result_indices = np.concatenate([selected_tp_idx, selected_fp_idx])
		result_ids = self.all_ids[result_indices]
		np.random.shuffle(result_ids)

		return result_ids


class ControlCheatInferCb(VFLCallback):
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
			self._verify(m)

		# 选择一批目标类样本
		aTgtPos = np.flatnonzero(np.isin(aIdxs, m.ns.aTgtIdxs))
		tTgtGradsL2 = tc.norm(data['grads'][aTgtPos], 2, 1)
		_, tSel = tc.topk(tTgtGradsL2, nSel)
		m.ns.aDstIdxs = aIdxs[aTgtPos[tSel]]
		# m.ns.aDstIdxs = rng().choice(m.ns.aTgtIdxs, nSel, False)

	def infer(self, m: BaseVFLArch, data: dict[str, tc.Tensor]) -> None:
		"""执行受控精度的推理逻辑"""
		aIdxs = data['idxs'].numpy()
		aLabels = data['labels'].numpy()
		sampler = NonOverlappingSampler(aIdxs, aLabels)

		m.ns.aTgtIdxs = sampler.generate_list(m.ns.iTgtLabel, self.rTgt, self.fPrec)
		m.ns.aVicIdxs = sampler.generate_list(m.ns.iVicLabel, self.rVic, self.fPrec)

		# 4. 验证并记录实际准确率
		tPredTgtLabels = aLabels[np.isin(aIdxs, m.ns.aTgtIdxs)]
		fRealTgtPrec = np.mean(tPredTgtLabels == m.ns.iTgtLabel).item()
		tPredVicLabels = aLabels[np.isin(aIdxs, m.ns.aVicIdxs)]
		fRealVicPrec = np.mean(tPredVicLabels == m.ns.iVicLabel).item()

		m.logText.info('[ControlCheat] 构造完成。')
		m.logText.info(f'    - 目标类集合大小：{len(m.ns.aTgtIdxs)}')
		m.logText.info(f'    - 样本选择比例：{self.rTgt:.2%} -> {len(m.ns.aTgtIdxs) / len(aIdxs):.2%}')
		m.logText.info(f'    - 目标类期望准确率：{self.fPrec:.2%} -> {fRealTgtPrec:.2%}')
		m.logText.info(f'    - 受害类集合大小：{len(m.ns.aVicIdxs)}')
		m.logText.info(f'    - 样本选择比例：{self.rVic:.2%} -> {len(m.ns.aVicIdxs) / len(aIdxs):.2%}')
		m.logText.info(f'    - 受害类期望准确率：{self.fPrec:.2%} -> {fRealVicPrec:.2%}')

	def _verify(self, m: BaseVFLArch) -> None:
		"""验证推理结果的集合完整性和统计正确性。

		Args:
				m: VFL 模型实例
		"""
		aTgtIdxs = m.ns.aTgtIdxs
		aVicIdxs = m.ns.aVicIdxs

		# [无重复] 检查单个集合内部是否存在重复索引
		assert len(aTgtIdxs) == len(np.unique(aTgtIdxs)), '错误：目标类集合包含重复索引！'
		assert len(aVicIdxs) == len(np.unique(aVicIdxs)), '错误：受害类集合包含重复索引！'

		# [无交集] 检查目标类和受害类是否有重叠
		intersection = np.intersect1d(aTgtIdxs, aVicIdxs, True)
		assert len(intersection) == 0, '错误：目标类和受害类存在重叠！'


class AddTrigger(tf.Transform):
	"""添加后门攻击触发器的转换类

	在图像的左上角区域添加特定的像素模式，用于实现后门攻击。
	"""

	@override
	def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
		"""对输入应用触发器转换。

		在输入张量的左上角区域添加特定的像素模式，用于后门攻击。
		如果输入不是张量，则返回 `None`。

		Args:
			inpt: 输入数据，可以是张量或其他类型
			params: 转换参数（当前未使用）

		Returns:
			添加触发器后的张量，如果输入不是张量则返回 `None`
		"""
		if isinstance(inpt, tc.Tensor):
			# 计算触发器区域大小：尺寸最小值的 1/8，但不小于 3
			size = max(min(inpt.shape[-2:]) // 8, 3)
			# 创建输入的副本以避免修改原始数据
			out = inpt.clone()
			# 将左上角区域设置为黑色
			out[..., :size, :size] = 0
			# 将特定位置设置为白色
			out[..., 3, 1] = 255
			out[..., 1, 3] = 255
			out[..., 2, 2] = 255
			out[..., 1, 1] = 255
			return out
		return None
