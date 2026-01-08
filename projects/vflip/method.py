"""VFLIP 方法模块"""

from itertools import permutations
from typing import Any, override

from torch.optim import Adam, Optimizer

from main.arch import BaseVFLArch
from main.callback import VFLCallback
from models.fcn import FCN
from utils.common import Path, copy, nn, np, tc
from utils.config import rng
from utils.define import StepVars
from utils.misc import notNone


class MAE(nn.Module):
	"""掩码自动编码器，用于学习嵌入的潜在表示并重构输入

	该类实现了一个简单的掩码自动编码器，通过编码器将输入嵌入压缩到潜在空间，
	再通过解码器重构原始输入，用于学习嵌入之间的关系和检测异常嵌入。
	"""

	def __init__(self, nDim: int) -> None:
		"""初始化掩码自动编码器实例。

		Args:
			nDim: 输入嵌入的维度
		"""
		super().__init__()

		self.encoder = FCN([nDim, int(nDim * 0.8), int(nDim * 0.6)], False)
		"""编码器网络，将输入嵌入压缩到低维潜在空间"""
		self.decoder = FCN([int(nDim * 0.6), int(nDim * 0.8), nDim], False)
		"""解码器网络，将潜在表示重构为原始输入维度"""

	@override
	def forward(self, ins: tc.Tensor) -> tc.Tensor:
		latent = self.encoder(ins)  # 编码为潜在表示
		out = self.decoder(latent)  # 解码重构输入
		return out


class VFLIP:
	"""VFL 异常检测与净化模型，用于垂直联邦学习中的参与方嵌入异常检测与修复

	该类实现了垂直联邦学习中的异常检测与净化功能，通过掩码自动编码器（MAE）
	学习参与方嵌入之间的关系，实现嵌入异常的检测和修复。
	"""

	def __init__(self, lPartyDims: list[int]) -> None:
		"""初始化 VFLIP 模型实例。

		Args:
			lPartyDims: 每个参与方的嵌入维度列表
		"""
		self.lPartyDims = lPartyDims
		"""每个参与方的嵌入维度列表"""
		self.nParty = len(lPartyDims)
		"""参与方数量"""
		self.nDim = sum(lPartyDims)
		"""总嵌入维度"""

		self.mae = MAE(self.nDim)
		"""掩码自动编码器"""
		self.tMasks = self._createPartyMasks()  # * [nParty, nDim]
		"""每个参与方的掩码"""

		self._mean: tc.Tensor | None = None
		"""训练集的均值"""
		self._std: tc.Tensor | None = None
		"""训练集的标准差"""
		self._tThres: tc.Tensor | None = None
		"""每个参与方的分数阈值"""

	@property
	def mean(self) -> tc.Tensor:
		"""获取训练集的均值。

		Returns:
			训练集的均值张量，形状为 `[nDim]`

		Raises:
			ValueError: 当均值未被赋值时
		"""
		if self._mean is None:
			msg = '均值未被赋值！'
			raise ValueError(msg)
		return self._mean

	@mean.setter
	def mean(self, value: tc.Tensor) -> None:
		"""设置训练集的均值。

		Args:
			value: 训练集的均值张量，形状必须为 `[nDim]`
		"""
		if value.size() != (self.nDim,):
			msg = f'均值必须具有形状 [{self.nDim}], 但得到 {value.size()}'
			raise ValueError(msg)
		self._mean = value

	@property
	def std(self) -> tc.Tensor:
		"""获取训练集的标准差。

		Returns:
			训练集的标准差张量，形状为 `[nDim]`

		Raises:
			ValueError: 当标准差未被赋值时
		"""
		if self._std is None:
			msg = '标准差未被赋值！'
			raise ValueError(msg)
		return self._std

	@std.setter
	def std(self, value: tc.Tensor) -> None:
		"""设置训练集的标准差。

		Args:
			value: 训练集的标准差张量，形状必须为 `[nDim]`
		"""
		if value.size() != (self.nDim,):
			msg = f'标准差必须具有形状 [{self.nDim}], 但得到 {value.size()}'
			raise ValueError(msg)
		self._std = value

	@property
	def tThres(self) -> tc.Tensor:
		"""获取每个参与方的分数阈值。

		Returns:
			每个参与方的分数阈值张量，形状为 `[nParty]`

		Raises:
			ValueError: 当阈值未被赋值时
		"""
		if self._tThres is None:
			msg = '分数阈值未被赋值！'
			raise ValueError(msg)
		return self._tThres

	@tThres.setter
	def tThres(self, value: tc.Tensor) -> None:
		"""设置每个参与方的分数阈值。

		Args:
			value: 每个参与方的分数阈值张量，形状必须为 `[nParty]`
		"""
		if value.size() != (self.nParty,):
			msg = f'分数阈值必须具有形状 [{self.nParty}], 但得到 {value.size()}'
			raise ValueError(msg)
		self._tThres = value

	def _createPartyMasks(self) -> tc.Tensor:
		"""创建每个参与方的掩码张量。

		为每个参与方生成一个维度掩码，用于标识该参与方在总嵌入向量中的对应维度位置。
		掩码中，参与方对应的维度位置为 `1.0`，其他位置为 `0.0`。

		Returns:
			形状为 `[nParty, nDim]` 的张量，每行对应一个参与方的维度掩码
		"""
		# 计算各参与方维度的累积和，用于确定每个参与方的维度范围
		aCumsum = np.cumsum(self.lPartyDims)

		lMasks: list[tc.Tensor] = []
		for i in range(self.nParty):
			# 初始化全零掩码
			mask = tc.zeros(self.nDim)
			# 将第 i 个参与方对应的维度范围设置为 1.0
			mask[aCumsum[i] - self.lPartyDims[i] : aCumsum[i]] = 1.0
			# 将第 i 个参与方的掩码添加到列表中
			lMasks.append(mask)

		# 将所有参与方的掩码堆叠成张量
		tMasks = tc.stack(lMasks, 0)  # * [nParty, nDim]
		return tMasks

	# ============
	# 训练损失
	# ============

	def computeN1Loss(self, embeds: tc.Tensor) -> tc.Tensor:
		"""计算“N-1 to 1”策略的损失。

		“N-1 to 1”策略是一种模型训练策略，对于每个样本随机选择一个参与方，
		使用其他所有参与方的嵌入信息来重构该参与方的嵌入，以学习参与方之间的相关性。

		Args:
			embeds: 所有参与方的嵌入向量，形状为 `[nBatchSize, nDim]`

		Returns:
			计算得到的批次平均损失值
		"""
		nBatchSize = embeds.size(0)  #: 批次大小

		# 针对批次中的每个样本，随机选择一个参与方
		tIdxsI = tc.randint(self.nParty, (nBatchSize,))  # * [nBatchSize]

		# 批量生成掩码
		tMasks = self.tMasks.to(embeds.device)  # 将掩码转移到与输入相同的设备上
		tMasksI = tMasks[tIdxsI]  # 选择参与方对应的掩码 # * [nBatchSize, nDim]
		tMasksI_ = 1 - tMasksI  # 反转掩码，用于选择除目标参与方外的所有参与方 # * [nBatchSize, nDim]

		# 使用其他参与方的嵌入重构整个嵌入
		recon = self.mae(tMasksI_ * embeds)  # * [nBatchSize, nDim]
		# 计算目标参与方嵌入的重构误差，求批次平均
		#! 如果参与方的维度 (lPartyDims) 差异很大（例如 Party A 有 100 维，Party B 有 5 维），
		#! Party A 的 L2 误差天然会比 Party B 大。
		#! 这会导致模型过度关注维度高的参与方，或在阈值计算时产生偏差。
		# TODO(UnoUzume): 建议使用均方误差 (MSE) 或除以维度的平方根进行归一化，使误差对维度不敏感。
		loss = tc.norm(tMasksI * (embeds - recon), 2, 1).mean()
		return loss

	def compute11Loss(self, embeds: tc.Tensor) -> tc.Tensor:
		"""计算“1 to 1”策略的损失。

		“1 to 1”策略是一种模型训练策略，对于每个样本随机选择一对不同的参与方 (i,j)，
		使用参与方 j 的嵌入信息来重构参与方 i 的嵌入，以学习参与方之间的一对一相关性。

		Args:
			embeds: 所有参与方的嵌入向量，形状为 `[nBatchSize, nDim]`

		Returns:
			计算得到的批次平均损失值
		"""
		nBatchSize = embeds.size(0)  #: 批次大小

		# 生成参与方之间的有序排列对 (i,j)，其中 i≠j
		aPairs = np.array(list(permutations(range(self.nParty), 2)))  # * ['排列对的总数', 2]

		# 随机抽取当前批次的各对参与方，每个样本对应一对不同的参与方
		aBatchPairs = rng().choice(aPairs, nBatchSize, True)  # * [nBatchSize, 2]

		# 获取对应的参与方掩码
		tMasks = self.tMasks.to(embeds.device)  # 将掩码转移到与输入相同的设备上
		tMasksI = tMasks[aBatchPairs[:, 0]]  # 选择第一个参与方 i 对应的掩码 # * [nBatchSize, nDim]
		tMasksJ = tMasks[aBatchPairs[:, 1]]  # 选择第二个参与方 j 对应的掩码 # * [nBatchSize, nDim]

		# 使用参与方 j 的嵌入重构整个嵌入
		recon = self.mae(tMasksJ * embeds)  # * [nBatchSize, nDim]
		# 计算参与方 i 嵌入的重构误差，求批次平均
		# TODO(UnoUzume): 参考 computeN1Loss() 中的建议，使用 MSE 或归一化 L2 损失。
		loss = tc.norm(tMasksI * (embeds - recon), 2, 1).mean()
		return loss

	# ============
	# 计算分数阈值
	# ============

	def computeBatchScores(self, embeds: tc.Tensor) -> tc.Tensor:
		"""计算批次中每个样本的 S 分数矩阵。

		S 分数矩阵用于衡量使用一个参与方的嵌入信息重构另一个参与方嵌入的质量，
		矩阵维度为 `[nBatchSize, nSrcParty, nDstParty]`，其中 `score[b, j, i]`
		表示在第 `b` 个样本中，使用参与方 `j` 的嵌入重构参与方 `i` 嵌入的误差分数。

		Args:
			embeds: 所有参与方的嵌入向量，形状为 `[nBatchSize, nDim]`

		Returns:
			S 分数矩阵，形状为 `[nBatchSize, nParty, nParty]`
		"""
		nBatchSize = embeds.size(0)  #: 批次大小

		tMasks = self.tMasks.to(embeds.device)  # 将掩码转移到与输入相同的设备上
		score = tc.zeros([nBatchSize, self.nParty, self.nParty], device=embeds.device)  # 初始化分数矩阵

		for j in range(self.nParty):
			# 使用参与方 j 的掩码提取其嵌入信息，并进行重构
			recon = self.mae(tMasks[j] * embeds)  # * [nBatchSize, nDim]

			for i in range(self.nParty):
				if j == i:
					continue  # 跳过使用相同参与方进行重构的情况
				# 计算使用参与方 j 的嵌入重构参与方 i 的嵌入的误差，并将结果存储到分数矩阵中
				score[:, j, i] = tc.norm(tMasks[i] * (embeds - recon), 2, 1)  # 使用 L2 范数计算误差

		return score

	def startThres(self) -> None:
		"""初始化阈值计算过程。

		创建一个空列表用于存储后续收集的所有分数，为阈值计算做准备。
		"""
		self.lAllScores: list[tc.Tensor] = []  # 初始化空列表，用于存储各个批次的分数

	def updateThres(self, tRawEmbeds: tc.Tensor) -> None:
		"""更新阈值计算所需的分数信息。

		对输入的原始嵌入进行标准化处理，计算批次中每个样本的分数矩阵，
		并将计算得到的目标参与方平均分数添加到分数列表中。

		Args:
			tRawEmbeds: 原始嵌入向量，形状为 `[nBatchSize, nDim]`
		"""
		# 对原始嵌入进行标准化处理
		tStandEmbeds = (tRawEmbeds - self.mean) / self.std  # * [nBatchSize, nDim]
		# 计算批次中每个样本的分数矩阵
		tScoreMat = self.computeBatchScores(tStandEmbeds)  # * [nBatchSize, nSrcParty, nDstParty]
		# 计算每个目标参与方的平均分数（排除自身重构的情况）
		tDstScore = tc.sum(tScoreMat, 1) / (tScoreMat.size(1) - 1)  # * [nBatchSize, nDstParty]
		# 将当前批次的分数添加到总列表中
		self.lAllScores.append(tDstScore)

	def calcThres(self) -> None:
		"""计算最终的异常检测阈值。

		将所有收集到的分数合并，然后为每个参与方计算异常检测阈值，
		阈值计算公式为：`均值 + 3 * 标准差`，用于识别异常嵌入。
		"""
		# 将所有批次的分数合并为一个张量
		tScores = tc.cat(self.lAllScores, 0)  # * [nTrainSize, nDstParty]
		# 计算每个参与方的异常检测阈值（均值 + 3 倍标准差）
		self.tThres = tc.mean(tScores, 0) + 3 * tc.std(tScores, 0)  # * [nDstParty]

	# ============
	# 识别与净化
	# ============

	def detect(self, tRawEmbeds: tc.Tensor) -> tc.Tensor:
		"""检测嵌入中的异常样本。

		该方法实现了基于投票机制的异常检测策略，通过计算各参与方嵌入间的重构分数，
		并基于阈值判断每个样本在各参与方嵌入上是否为异常。

		Args:
			tRawEmbeds: 原始嵌入张量，形状为 `[nBatchSize, nDim]`

		Returns:
			异常检测结果张量，形状为 `[nBatchSize, nParty]`，每个元素为布尔值，
			表示对应样本在对应参与方嵌入上是否为异常
		"""
		# 标准化输入嵌入
		tStandEmbeds = (tRawEmbeds - self.mean) / self.std

		# 计算批次分数矩阵
		tScoreMat = self.computeBatchScores(tStandEmbeds)  # * [nBatchSize, nSrcParty, nDstParty]
		# 统计超过阈值的投票数
		tVoteMat = tc.sum((tScoreMat > self.tThres), 1)  # * [nBatchSize, nDstParty]
		# 基于多数投票判断异常：投票数 >= 参与方数量的一半
		tIsAnomaly = tVoteMat >= self.nParty / 2  # * [nBatchSize, nDstParty]
		return tIsAnomaly

	def purify(self, tRawEmbeds: tc.Tensor, tIsAnomaly: tc.Tensor) -> tc.Tensor:
		"""净化嵌入中的异常样本。

		该方法利用 MAE 模型重构异常嵌入，通过掩码机制保留正常嵌入部分，
		仅重构被检测为异常的嵌入部分，实现嵌入净化。

		Args:
			tRawEmbeds: 原始嵌入张量，形状为 `[nBatchSize, nDim]`
			tIsAnomaly: 异常检测结果张量，形状为 `[nBatchSize, nParty]`，
				每个元素为布尔值，表示对应样本在对应参与方嵌入上是否为异常

		Returns:
			净化后的嵌入张量，形状为 `[nBatchSize, nDim]`
		"""
		# 标准化输入嵌入
		tStandEmbeds = (tRawEmbeds - self.mean) / self.std

		# 将掩码移动到与输入相同的设备
		tMasks = self.tMasks.to(tStandEmbeds.device)
		# 初始化清理掩码，默认为全 1（保留所有嵌入）
		tCleanMask = tc.ones_like(tStandEmbeds)
		# 为每个样本生成清理掩码：减去异常对应的掩码之和
		for i in range(tStandEmbeds.size(0)):
			tCleanMask[i] -= tMasks[tIsAnomaly[i]].sum(0)

		# 使用清理后的嵌入重构异常部分
		recon = self.mae(tCleanMask * tStandEmbeds)
		# 将重构结果恢复到原始嵌入空间
		restore = recon * self.std + self.mean
		# 组合原始正常嵌入和重构的异常嵌入
		purify = tCleanMask * tRawEmbeds + (1 - tCleanMask) * restore
		return purify

	def single(self, tRawEmbeds: tc.Tensor) -> list[tc.Tensor]:
		"""为每个参与方生成净化嵌入。

		该方法逐一假设参与方异常，分别生成净化嵌入。

		Args:
			tRawEmbeds: 原始嵌入张量，形状为 `[nBatchSize, nDim]`

		Returns:
			净化嵌入列表（长度为 `nParty`），每个元素为一个净化嵌入张量，形状为 `[nBatchSize, nDim]`
		"""
		# 标准化输入嵌入
		tStandEmbeds = (tRawEmbeds - self.mean) / self.std

		# 初始化净化嵌入列表
		lPurify = []
		# 将掩码移动到与输入相同的设备
		tMasks = self.tMasks.to(tStandEmbeds.device)
		# 为每个掩码生成净化嵌入
		for i in range(tMasks.size(0)):
			# 获取当前掩码
			tCleanMask = tMasks[i]
			# 使用当前掩码重构嵌入
			recon = self.mae(tCleanMask * tStandEmbeds)
			# 将重构结果恢复到原始嵌入空间
			restore = recon * self.std + self.mean
			# 组合原始嵌入和重构结果
			purify = tCleanMask * tRawEmbeds + (1 - tCleanMask) * restore
			# 将净化结果添加到列表
			lPurify.append(purify)

		return lPurify


class VFLIPCb(VFLCallback):
	"""VFLIP 模型回调类，实现垂直联邦学习中的异常检测与净化功能

	该类继承自 `VFLCallback`，用于在 VFL 训练过程中集成 VFLIP 模型，
	实现参与方嵌入的异常检测、净化以及相关指标的记录。
	"""

	def __init__(self, dpRoot: Path | str, lPartyDims: list[int], M: float, N: float) -> None:
		"""初始化 VFLIP 回调类。

		Args:
			dpRoot: 根路径，用于保存中间结果
			lPartyDims: 各参与方嵌入维度列表
			M: 熵阈值计算参数 M
			N: 熵阈值计算参数 N
		"""
		super().__init__()
		self.dpRoot = Path(dpRoot)
		"""根路径，用于保存中间结果"""
		self.lPartyDims = lPartyDims
		"""各参与方嵌入维度列表"""
		self.M = M
		"""熵阈值参数 M"""
		self.N = N
		"""熵阈值参数 N"""
		self.vflip = VFLIP(lPartyDims)
		"""VFLIP 模型实例"""
		self.lTrainEmbeds: list[tc.Tensor] = []
		"""训练嵌入列表"""

	@override
	def onInitModule(self, m: BaseVFLArch) -> None:
		m.add_module('mae_VFLIP', self.vflip.mae)  # 将 MAE 模块添加到 VFL 架构
		# ? m.register_buffer('tMasks', self.vflip.tMasks)  # 注册掩码为缓冲区（注释状态）

	@override
	def onConfigOptims(self) -> list[Optimizer]:
		optN1 = Adam(self.vflip.mae.parameters(), 1e-4)  #: N1 损失优化器，学习率 1e-4
		opt11 = Adam(self.vflip.mae.parameters(), 5e-4)  #: 11 损失优化器，学习率 5e-4
		return [optN1, opt11]

	@override
	def onSaveCheckpoint(self, m: BaseVFLArch, ckpt: dict[str, Any]) -> None:
		assert 'VFLIPCb' not in ckpt  # 确保检查点中不存在 VFLIPCb 键
		ckpt['VFLIPCb'] = {'lTrainEmbeds': self.lTrainEmbeds}  # 保存训练嵌入列表

	@override
	def onLoadCheckpoint(self, m: BaseVFLArch, ckpt: dict[str, Any]) -> None:
		dState = ckpt['VFLIPCb']  # 获取 VFLIPCb 状态
		# 恢复训练嵌入并移至正确设备
		self.lTrainEmbeds = [t.to(m.device) for t in dState['lTrainEmbeds']]

	# ============
	# 训练阶段
	# ============

	@override
	def onTrainEpochStart(self, m: BaseVFLArch) -> None:
		self.lTrainEmbeds = []  # 清空训练嵌入列表

	@override
	def onTrainTopIns(self, m: BaseVFLArch, v: StepVars) -> None:
		# 拼接并分离顶层输入，获取原始嵌入
		tRawEmbeds = tc.cat(v.lTopIns, 1).detach()  # * [nBatchSize, nDim]
		self.lTrainEmbeds.append(tRawEmbeds)  # 添加到训练嵌入列表

		# 计算当前批次嵌入的均值和标准差
		#! 如果在验证时 batch size 较小或分布偏移，可能出现问题
		# TODO(UnoUzume): 建议在 MAE 内部的第一层添加 nn.BatchNorm1d 或 nn.LayerNorm，而不是手动计算
		mean = tc.mean(tRawEmbeds, 0)
		std = tc.std(tRawEmbeds, 0) + 1e-8  # 避免除以零
		tStandEmbeds = (tRawEmbeds - mean) / std  # 标准化嵌入

		# 训练 N1 损失
		optN1 = m.getOptim(self.iOpt)
		optN1.zero_grad()
		lossN1 = self.vflip.computeN1Loss(tStandEmbeds)  # 计算 N1 损失
		m.manual_backward(lossN1)
		optN1.step()

		# 训练 11 损失
		opt11 = m.getOptim(self.iOpt + 1)
		opt11.zero_grad()
		loss11 = self.vflip.compute11Loss(tStandEmbeds)  # 计算 11 损失
		m.manual_backward(loss11)
		opt11.step()

		# 记录损失
		m.logDict({'loss/TrainingN1': lossN1, 'loss/Training11': loss11})

	# ============
	# 验证阶段
	# ============

	@override
	def onValEpochStart(self, m: BaseVFLArch) -> None:
		if m.trainer.sanity_checking:
			return

		tTrainEmbeds = tc.cat(self.lTrainEmbeds, 0)  # 拼接所有训练嵌入

		# 计算训练嵌入的均值和标准差
		self.vflip.mean = tc.mean(tTrainEmbeds, 0)
		self.vflip.std = tc.std(tTrainEmbeds, 0) + 1e-8

		# 初始化并计算阈值
		self.vflip.startThres()  # 初始化阈值列表
		for tRawEmbeds in self.lTrainEmbeds:
			self.vflip.updateThres(tRawEmbeds)  # 更新阈值
		self.vflip.calcThres()  # 计算最终阈值

		# 计算熵阈值
		self.calcEntropyThres(m)

		self.lEntropy: list[tc.Tensor] = []  # 初始化熵列表

	def calcEntropyThres(self, m: BaseVFLArch) -> None:
		"""计算熵阈值，用于检测低熵样本。

		Args:
			m: VFL 架构实例
		"""
		lAllEntropy = []  # 所有熵值列表
		for tRawEmbeds in self.lTrainEmbeds:  # * [nBatchSize, nDim] | nBatch
			# 获取每个掩码的净化嵌入
			lPurify = self.vflip.single(tRawEmbeds)  # * [nParty, [nBatchSize, nDim]]
			lEntropy = []
			for purify in lPurify:  # * [nBatchSize, nDim] | nParty
				# 将净化嵌入分割为各参与方嵌入
				lTopIns = list(tc.split(purify, self.lPartyDims, 1))
				# 通过顶层网络获取输出
				zTopOut = m.zTopNet(lTopIns)  # * [nBatchSize, nClass]
				# 计算类别概率
				probs = tc.nn.functional.softmax(zTopOut, dim=1)  # * [nBatchSize, nClass]
				# 计算熵值 #* [nBatchSize]
				entropy = tc.distributions.Categorical(probs).entropy()  # type: ignore[no-untyped-call]
				lEntropy.append(entropy)  # * [nParty, [nBatchSize]]

			tEntropy = tc.stack(lEntropy, 1)  # 拼接熵值 #* [nBatchSize, nParty]
			lAllEntropy.append(tEntropy)  # * [nBatch, [nBatchSize, nParty]]
		tAllEntropy = tc.cat(lAllEntropy, 0)  # 拼接所有批次的熵值 #* [nTrainSize, nParty]
		self.tEntropyMean = tc.mean(tAllEntropy, 0)  # 计算熵值均值 #* [nParty]
		# 最后一个 epoch 保存熵值均值
		if m.current_epoch == notNone(m.trainer.max_epochs) - 1:
			tc.save(self.tEntropyMean, Path(notNone(m.trainer.log_dir)) / 'tEntropyMean.pt')

	@override
	def onValTopIns(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		if m.trainer.sanity_checking:
			return

		# 初始化占位变量
		vOD = d['Origin']  # 原始数据净化（异常检测）
		vOE = d['Origin']  # 原始数据净化（熵值检测）
		vAD = d['Attack']  # 攻击数据净化（异常检测）
		vAE = d['Attack']  # 攻击数据净化（熵值检测）
		# 获取各参与方嵌入维度
		lTopInsDims = [t.size(1) for t in vOD.lTopIns]

		for k, v in d.items():  # 针对原始样本和攻击样本
			# 拼接顶层输入获取原始嵌入
			tRawEmbeds = tc.cat(v.lTopIns, 1)  # * [nBatchSize, nDim]

			# region 异常分数检测

			# 检测异常样本
			tIsAnomaly = self.vflip.detect(tRawEmbeds)  # * [nBatchSize, nParty]

			# 记录各参与方异常检测准确率
			acc = tIsAnomaly.float().mean(0)  # * [nParty]
			m.logDict({f'det/Loader{k}/Party{i}': acc[i] for i in range(acc.size(0))})

			# 记录全异常样本比例
			tIsBad = tIsAnomaly.all(1)  # * [nBatchSize]
			m.logDict({f'det/Loader{k}/Bad': tIsBad.float().mean()})

			# endregion

			# region 概率熵值检测

			# 获取每个掩码的净化嵌入
			lPurify = self.vflip.single(tRawEmbeds)  # * [nParty, [nBatchSize, nDim]]
			lEntropy = []
			for purify in lPurify:  # * [nBatchSize, nDim] | nParty
				# 将净化嵌入分割为各参与方嵌入
				lTopIns = list(tc.split(purify, lTopInsDims, 1))
				# 通过顶层网络获取输出
				zTopOut = m.zTopNet(lTopIns)  # * [nBatchSize, nClass]
				# 计算类别概率
				probs = tc.nn.functional.softmax(zTopOut, dim=1)  # * [nBatchSize, nClass]
				# 计算熵值 #* [nBatchSize]
				entropy = tc.distributions.Categorical(probs).entropy()  # type: ignore[no-untyped-call]
				lEntropy.append(entropy)  # * [nParty, [nBatchSize]]

			tEntropy = tc.stack(lEntropy, 1)  # 拼接熵值 #* [nBatchSize, nParty]
			if k == 'Attack':  # 仅保存攻击样本的熵值
				self.lEntropy.append(tEntropy)
			# 检测低熵样本：熵值小于 M*均值 且 熵值小于 N*最大熵
			temp = tEntropy < self.M * self.tEntropyMean
			tIsLowEntropy = tc.logical_and(temp, tEntropy < (self.N * tEntropy.max(1)[0][:, None]))

			# 记录各参与方低熵检测准确率
			acc = tIsLowEntropy.float().mean(0)  # * [nParty]
			m.logDict({f'ent/Loader{k}/Party{i}': acc[i] for i in range(acc.size(0))})

			# 记录全低熵样本比例
			tIsBad = tIsLowEntropy.all(1)  # * [nBatchSize]
			m.logDict({f'ent/Loader{k}/Bad': tIsBad.float().mean()})

			# endregion

			# 净化原始数据
			if k == 'Origin':
				vOD = copy(v)  # 复制原始步骤变量
				purify = self.vflip.purify(tRawEmbeds, tIsAnomaly)  # 基于异常检测净化
				vOD.lTopIns = list(tc.split(purify, lTopInsDims, 1))  # 更新顶层输入

				vOE = copy(v)  # 复制原始步骤变量
				purify = self.vflip.purify(tRawEmbeds, tIsLowEntropy)  # 基于熵值检测净化
				vOE.lTopIns = list(tc.split(purify, lTopInsDims, 1))  # 更新顶层输入

			# 净化攻击数据
			if k == 'Attack':
				vAD = copy(v)  # 复制攻击步骤变量
				purify = self.vflip.purify(tRawEmbeds, tIsAnomaly)  # 基于异常检测净化
				vAD.lTopIns = list(tc.split(purify, lTopInsDims, 1))  # 更新顶层输入

				vAE = copy(v)  # 复制攻击步骤变量
				purify = self.vflip.purify(tRawEmbeds, tIsLowEntropy)  # 基于熵值检测净化
				vAE.lTopIns = list(tc.split(purify, lTopInsDims, 1))  # 更新顶层输入

		# 将净化后的步骤变量添加到字典
		d['OriginD'] = vOD  # 原始数据净化（异常检测）
		d['OriginE'] = vOE  # 原始数据净化（熵值检测）
		d['AttackD'] = vAD  # 攻击数据净化（异常检测）
		d['AttackE'] = vAE  # 攻击数据净化（熵值检测）

	@override
	def onValEpochEnd(self, m: BaseVFLArch) -> None:
		if m.trainer.sanity_checking:
			return

		# 拼接并移至 CPU
		temp = tc.cat(self.lEntropy, 0).cpu()  # * [nTrainSize, nParty]
		# 最后一个 epoch 保存熵值
		if m.current_epoch == notNone(m.trainer.max_epochs) - 1:
			tc.save(temp, Path(notNone(m.trainer.log_dir)) / 'tEntropy.pt')
