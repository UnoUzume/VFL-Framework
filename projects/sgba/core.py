"""SGBA 攻击实现模块"""

from dataclasses import dataclass
from typing import override

from torch.optim import AdamW, lr_scheduler as lrs

from main.arch import BaseVFLArch
from main.callback import OPT_TYPE, VFLCallback
from models.fcn import FCN
from projects.vfl.config import AppConfig
from utils.common import F, copy, tc
from utils.define import StepVars
from utils.misc import accuracy, segment


def calcVecLoss(a: tc.Tensor, b: tc.Tensor, method: str = 'MSE') -> tc.Tensor:
	"""计算批次损失，支持多种损失函数。

	Args:
		a: 预测向量，形状为 `(nBatch, nDim)`
		b: 目标向量，形状为 `(nBatch, nDim)`
		method: 损失函数，可选 `'SSE'`、`'SSE/N'`、`'MSE'`、`'L2.ND'`、`'L2.D/N'`

	Returns:
		批次损失

	Raises:
		ValueError: 损失函数未知
	"""
	if method == 'SSE':
		loss = F.mse_loss(a, b, reduction='sum')
	elif method == 'SSE/N':  # 样本 SSE，批量平均
		loss = F.mse_loss(a, b, reduction='sum') / len(a)
	elif method == 'MSE':
		loss = F.mse_loss(a, b, reduction='mean')
	elif method == 'Huber/N':  # 样本 Huber 损失，批量平均
		loss = F.huber_loss(a, b, reduction='sum', delta=1.0) / len(a)
	elif method == 'Huber':
		loss = F.huber_loss(a, b, reduction='mean', delta=1.0)
	elif method == 'L2/N':  # 样本 L2 距离，批量平均
		loss = tc.norm(a - b, p=2, dim=1).mean()
	elif method == 'L2':
		loss = tc.norm(a - b, p=2)
	else:
		msg = f'损失函数未知：{method}！'
		raise ValueError(msg)
	return loss


@dataclass
class MethodArgs:
	"""SGBA 攻击方法参数配置类

	Args:
		fRecLr: 生成网络的学习率
		fTrainAlpha: 训练时生成样本的混合比例
		fValAlpha: 验证时生成样本的混合比例
		lLossScales: 生成网络的损失缩放比例
		lGradScales: 投毒目标的梯度缩放比例
	"""

	fRecLr: float
	"""生成网络的学习率"""
	fTrainAlpha: float
	"""训练时生成样本的混合比例"""
	fValAlpha: float
	"""验证时生成样本的混合比例"""
	lLossScales: tuple[float, float]
	"""生成网络的损失缩放比例"""
	lGradScales: tuple[float, float]
	"""投毒目标的梯度缩放比例"""


class SGBACb(VFLCallback):
	"""SGBA 攻击回调类"""

	def __init__(self, args: MethodArgs, config: AppConfig) -> None:
		"""初始化实例。

		Args:
			args: SGBA 攻击方法参数配置
			config: 应用配置对象
		"""
		self.args = args
		"""SGBA 攻击方法参数配置"""
		self.cfg = config
		"""应用配置"""

		self.iRP = 0
		"""攻击节点的索引"""

		nDim = self.cfg.model.lPartyDims[self.iRP]
		self.zRecNet = FCN(lDims=[nDim, int(nDim * 0.8), int(nDim * 0.8), nDim], hasBN=False)  # ! 可变
		"""攻击者用于生成后门触发器的网络"""

	@override
	def onInitModule(self, m: BaseVFLArch) -> None:
		m.add_module('zRecNet', self.zRecNet)

	@override
	def onConfigOptims(self) -> OPT_TYPE:
		optRec = AdamW(self.zRecNet.parameters(), lr=self.args.fRecLr)
		lrsRec = lrs.LinearLR(optRec, start_factor=0.1, total_iters=5)
		return [optRec], [lrsRec]

	def getRecon(self, tEmbed: tc.Tensor, alpha: float = 1.0) -> tuple[tc.Tensor, tc.Tensor]:
		"""根据输入嵌入生成重构输出。

		Args:
			tEmbed: 输入嵌入
			alpha: 重构输出的混合比例，默认为 `1.0`

		Returns:
			重构输出
			重构损失值
		"""
		# 生成重构输出
		tRecOut = self.zRecNet(tEmbed)
		# 混合重构输出
		tRecon = alpha * tRecOut + (1 - alpha) * tEmbed
		# 计算重构损失
		vLoss = calcVecLoss(tRecOut, tEmbed, 'Huber/N')

		return tRecon, vLoss

	# ============
	# 训练阶段
	# ============

	@override
	def onTrainBtmIns(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch > 0:
			self.switch(m, v)

	def switch(self, m: BaseVFLArch, v: StepVars) -> None:
		"""执行样本切换。"""
		# 获取投毒目的样本（属于目标类样本）的掩码
		self.tDstMask = tc.isin(v.indices, m.ns.tDstIds)
		# 获取受害类样本的掩码
		self.tVicMask = tc.isin(v.indices, m.ns.tVicIds)

		# # 在批次内部执行样本切换
		if self.tDstMask.any() and self.tVicMask.any():
			# 获取投毒目的样本（属于目标类样本）的位置（索引的索引）
			[tDstPos] = tc.nonzero(self.tDstMask, as_tuple=True)
			# 获取受害类样本的位置（索引的索引）
			[tVicPos] = tc.nonzero(self.tVicMask, as_tuple=True)

			# 方案一：有放回抽取
			tSelect = tc.randint(high=len(tVicPos), size=(len(tDstPos),), device=m.device)
			# 方案二：无放回抽取，但是来源样本可能少于目的样本
			# tSelect = tc.randperm(len(tVicPos), device=m.device)[: len(tDstPos)]

			# 批量样本切换
			v.lBtmIns[self.iRP][tDstPos] = v.lBtmIns[self.iRP][tVicPos[tSelect]]

		# 整个训练集样本切换
		# if len(aDstPos) > 0:  # 如果找到投毒目标
		# 	# 寻找投毒目标对应的 self.aDstIdxs 的元素位置，同时作为 self.aSrcIdxs 的元素位置
		# 	aSrcIdxs_Pos = _aDstIdxs_Pos = np.where(aBatchIdxs[aDstPos, None] == m.ns.aDstIdxs)[1]
		# 	# 获取投毒来源的元素
		# 	aSrcIdxsSubset = m.ns.aSrcIdxs[aSrcIdxs_Pos]

		# 	for pos, src_idx in zip(self.aDstPos, aSrcIdxsSubset, strict=True):
		# 		image = m.module.dsTrain[src_idx.item()][0]  # pyright: ignore[reportAttributeAccessIssue]

		# 		parts = m.module.fnSplit(image.unsqueeze(0), m.module.nParty)
		# 		parts = [m.module.tfCurrent(part) for part in parts]
		# 		v.lBtmIns[0][pos] = parts[0]

	@override
	def onTrainBtmOut(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch > 0:
			self.poison(m, v)

	def poison(self, m: BaseVFLArch, v: StepVars) -> None:
		"""执行生成式投毒。"""
		# ! 不使用 detach()，让底层模型也更新，降低触发器生成网络的训练难度
		tEmbed = v.lBtmOut[self.iRP]
		_, self.vReconLoss = self.getRecon(tEmbed)
		m.logDict({'loss/ReconTrain': self.vReconLoss})

		# # 投毒操作
		v.lBtmOut[self.iRP] = v.lBtmOut[self.iRP].clone()  # 避免 In-place 操作错误

		# # 向目标类投毒（目的样本特征改成来源样本特征，建立目标类与来源样本的联系）
		if self.tDstMask.any():
			# ! 使用 detach() 避免投毒样本的梯度传播至底层模型，产生意外影响
			tEmbed = v.lBtmOut[self.iRP][self.tDstMask].detach()
			tRecon, _ = self.getRecon(tEmbed, self.args.fTrainAlpha)
			v.lBtmOut[self.iRP][self.tDstMask] = tRecon

	@override
	def onTrainLoss(self, m: BaseVFLArch, v: StepVars) -> None:
		optRec = m.getOptim(self.iOpt)
		optRec.zero_grad()

	@override
	def onTrainBtmOutGrad(self, m: BaseVFLArch, v: StepVars) -> None:
		if m.current_epoch > 0:
			self.doSGBA(m, v)

	def doSGBA(self, m: BaseVFLArch, v: StepVars) -> None:
		"""执行 SGBA 攻击。"""
		# 重构损失的缩放与反向传播
		p = segment(m.current_epoch, ins=(10, 30), out=self.args.lLossScales)
		m.logDict({'value/LossScale': p})
		m.manual_backward(self.vReconLoss * p, retain_graph=True)

		# 目的样本梯度的缩放
		value = segment(m.current_epoch, ins=(10, 30), out=self.args.lGradScales)
		m.logDict({'value/GradScale': value})
		if self.tDstMask.any():
			v.lTopInsGrad[self.iRP][self.tDstMask] *= value

	@override
	def onTrainOptimStep(self, m: BaseVFLArch, v: StepVars) -> None:
		optRec = m.getOptim(self.iOpt)
		optRec.step()

	# ============
	# 验证阶段
	# ============

	@override
	def onValStepVars(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		d['Attack'] = copy(d['Origin'])

	@override
	def onValBtmOut(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		v = d['Attack']

		# 计算重构损失
		tEmbed = v.lBtmOut[self.iRP]
		tRecon, vReconLoss = self.getRecon(tEmbed, self.args.fValAlpha)
		v.lBtmOut[self.iRP] = tRecon

		m.logDict({'loss/ReconVal': vReconLoss})

	@override
	def onValLoss(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		for k, v in d.items():
			probs = F.softmax(v.zTopOut, 1)
			entropy = tc.distributions.Categorical(probs).entropy().mean().item()
			m.logDict({f'entropy/Val{k}': entropy})

			if m.current_epoch > 0:
				self.logVal(m, k, v)

	@staticmethod
	def logVal(m: BaseVFLArch, k: str, v: StepVars) -> None:
		"""记录验证阶段的损失。"""
		logk = f'Val{v.iLoaderIdx}_{k}'
		# 获取非目标类样本的掩码
		mask = v.labels != m.ns.iTgtLabel
		if mask.any():
			tTgtLogits = v.zTopOut[mask]
			tTgtLabels = tc.full_like(v.labels[mask], m.ns.iTgtLabel)

			loss = m.criterion(tTgtLogits, tTgtLabels)
			[acc1, acc3] = accuracy(lprobs=tTgtLogits, target=tTgtLabels, topk=(1, 3))
			m.logDict({f'lossTgt/{logk}': loss, f'accTgt/{logk}/Top1': acc1, f'accTgt/{logk}/Top3': acc3})
