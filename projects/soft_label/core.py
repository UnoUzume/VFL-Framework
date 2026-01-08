from typing import override

from torch.optim import AdamW

from main.arch import BaseVFLArch
from main.callback import OPT_TYPE, VFLCallback
from models.fcn import FCN
from projects.vfl.config import AppConfig
from utils.common import F, nn, tc
from utils.define import StepVars, TSplitImageBatch
from utils.misc import accuracy


class SoftLabelCb(VFLCallback):
	def __init__(self, config: AppConfig) -> None:
		self.cfg = config

		self.zSurNet = FCN([self.cfg.model.lPartyDims[0], 256, 10])
		self.criSur = nn.CrossEntropyLoss()

		self.fSurLr = 1e-2

		# 温度参数调度
		self.T_max = 10
		self.T_min = 10
		self.current_T = self.T_max

	@override
	def onInitModule(self, m: BaseVFLArch) -> None:
		m.logText.info(f'{self.__class__.__name__}.onInitModule()')

		m.add_module('zSurNet', self.zSurNet)
		m.add_module('criSur', self.criSur)

	@override
	def onConfigOptims(self) -> OPT_TYPE:
		optSur = AdamW(self.zSurNet.parameters(), self.fSurLr)
		# lrsSur = createLRS(optSur, [5, 20, 30])
		return [optSur]

	# ============
	# 训练阶段
	# ============

	@override
	def onTrainStepVars(self, m: BaseVFLArch, v: StepVars) -> None:
		uAuxBatch: TSplitImageBatch = v.raw[1]  #! 来自辅助数据集
		self.tAuxLabels = uAuxBatch.label
		v.images[0] = tc.cat([v.images[0], uAuxBatch.image[0]], 0)

	@override
	def onTrainBtmOut(self, m: BaseVFLArch, v: StepVars) -> None:
		tEmbeds = v.lBtmOut[0]

		self.tTrainEmbeds = tEmbeds[: len(v.labels)]
		self.tAuxEmbeds = tEmbeds[len(v.labels) :]

		v.lBtmOut[0] = self.tTrainEmbeds

	@override
	def onTrainLoss(self, m: BaseVFLArch, v: StepVars) -> None:
		optSur = m.getOptim(self.iOpt)
		optSur.zero_grad()

	@override
	def onTrainTopInsGrad(self, m: BaseVFLArch, v: StepVars) -> None:
		with tc.no_grad():
			# 生成代理特征
			tUniques = tc.unique(self.tAuxLabels)
			assert len(tUniques) == 10, '标签数量错误'
			lAgentFeatures = [self.tAuxEmbeds[self.tAuxLabels == label].mean(0) for label in tUniques]
			tAgentFeatures = tc.stack(lAgentFeatures)

			# 1. 对训练特征和代理特征都做 L2 归一化
			tTrainNorm = F.normalize(self.tTrainEmbeds, p=2, dim=1)
			tAgentNorm = F.normalize(tAgentFeatures, p=2, dim=1)

			# 2. 计算余弦相似度 (范围 -1 到 1)
			cosine_sim = tc.mm(tTrainNorm, tAgentNorm.t())

			# 3. 此时通常需要一个较大的 Inverse Temperature (如 scale=20) 来拉开差距
			# 否则分布会太平坦
			scale = 10.0
			tSoftLabels = tc.softmax(cosine_sim * scale, 1)

			# 生成软标签
			#! 公式中右上角的 T 表示转置，分母的 T 表示温度参数
			# similarity = tc.mm(self.tTrainEmbeds, tAgentFeatures.t())
			# tSoftLabels = tc.softmax(similarity / self.current_T, 1)

			[vSoftAcc] = accuracy(tSoftLabels, v.labels)
			m.logDict({'acc/soft': vSoftAcc})

		# 模型损失
		tSurIns = self.tTrainEmbeds.detach().requires_grad_()
		tSurOut = self.zSurNet(tSurIns)
		vModelLoss = self.criSur(tSurOut, tSoftLabels)

		# 梯度损失
		[tSurGrad] = tc.autograd.grad(vModelLoss, tSurIns, create_graph=True)
		tRawGrad = v.lTopInsGrad[0]
		vGradLoss = F.mse_loss(tSurGrad, tRawGrad)

		# 总损失
		vSurLoss = vModelLoss + 2e5 * vGradLoss  # 调整权重
		m.manual_backward(vSurLoss)

		# tSurInsGrad = notNone(tSurIns.grad) * 1e-2
		# m.manual_backward(self.tTrainEmbeds, tSurInsGrad, retain_graph=True)

		# 更新温度参数
		if m.trainer.current_epoch < 20:
			progress = m.trainer.current_epoch / 20
			self.current_T = self.T_max - (self.T_max - self.T_min) * progress
		else:
			self.current_T = self.T_min

		m.logDict(
			{
				'loss/attack': vSurLoss,
				'T': self.current_T,
				'loss/grad': vGradLoss,
				'loss/model': vModelLoss,
			}
		)

	@override
	def onTrainOptimStep(self, m: BaseVFLArch, v: StepVars) -> None:
		optSur = m.getOptim(self.iOpt)
		optSur.step()

	# ============
	# 验证阶段
	# ============

	@override
	def onValLoss(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		v = d['Origin']
		lossTop = v.loss
		[accTop] = accuracy(v.zTopOut, v.labels)

		insSur = v.lBtmOut[0]
		outSur = self.zSurNet(insSur)
		lossSur = self.criSur(outSur, v.labels)
		[accSur] = accuracy(outSur, v.labels)

		if v.iLoaderIdx == 0:  # 验证集
			m.logDict(
				{
					'loss/val/top': lossTop,
					'acc/val/top': accTop,
					'loss/val/sur': lossSur,
					'acc/val/sur': accSur,
				},
			)
		elif v.iLoaderIdx == 1:  # 训练集
			m.logDict(
				{
					'loss/train/top': lossTop,
					'acc/train/top': accTop,
					'loss/train/sur': lossSur,
					'acc/train/sur': accSur,
				},
			)
