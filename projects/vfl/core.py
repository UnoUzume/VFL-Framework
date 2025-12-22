"""纵向联邦学习（VFL）架构实现模块。

本模块实现了纵向联邦学习的具体架构，包括底层模型和顶层模型的定义、
优化器配置以及训练、验证和测试过程中的回调处理。
"""

from typing import override

from torch.optim import Adam, Optimizer, lr_scheduler as lr

from main.arch import BaseVFLArch
from main.callback import VFLCallback
from utils.define import StepVars
from utils.misc import accuracy

from .config import AppConfig


def createLRScheduler(optimizer: Optimizer) -> lr.LRScheduler:
	"""创建学习率调度器链。

	组合使用线性学习率预热和多步学习率衰减策略。

	Args:
		optimizer: 需要应用学习率调度的优化器

	Returns:
		组合后的学习率调度器
	"""
	scheduler1 = lr.LinearLR(optimizer, 0.1, total_iters=10)
	scheduler2 = lr.MultiStepLR(optimizer, [20, 80], 0.2)
	return lr.ChainedScheduler([scheduler1, scheduler2], optimizer)


class VFLArch(BaseVFLArch):
	"""纵向联邦学习架构实现类，定义了 VFL 的具体模型结构和训练流程

	本类实现了纵向联邦学习的核心逻辑，包括底层模型和顶层模型的协同训练。
	"""

	def __init__(self, config: AppConfig, lCallbacks: list[VFLCallback] | None = None) -> None:
		"""初始化实例。

		Args:
				config: 应用配置对象，包含模型和训练参数
				lCallbacks: 回调函数列表，用于在训练过程中执行自定义逻辑
		"""
		super().__init__(config.dpRoot, lCallbacks)
		self.logText.info(f'{self.__class__.__name__}.__init__()')
		self.cfg = config

		self.lBtmNets = config.model.getBtmNets()
		self.zTopNet = config.model.getTopNet()

	@override
	def onConfigOptims(self, iOpt: int, iLRS: int) -> tuple[list[Optimizer], list[lr.LRScheduler]]:
		optBtms = [Adam(net.parameters(), self.cfg.run.lr) for net in self.lBtmNets]
		optTop = Adam(self.zTopNet.parameters(), self.cfg.run.lr)

		lrsBtms = [createLRScheduler(opt) for opt in optBtms]
		lrsTop = createLRScheduler(optTop)
		return [*optBtms, optTop], [*lrsBtms, lrsTop]

	# ============
	# 训练阶段
	# ============

	@override
	def onTrainStepVars(self, m: BaseVFLArch, v: StepVars) -> None:
		[v.images, v.labels, v.indices] = v.batch

	@override
	def onTrainLoss(self, m: BaseVFLArch, v: StepVars) -> None:
		self.logDict({'loss/Training': v.loss})

	# ============
	# 验证阶段
	# ============

	@override
	def onValStepVars(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		for v in d.values():  # * Origin
			[v.images, v.labels, v.indices] = v.batch

	@override
	def onValLoss(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		for k, v in d.items():  # * Origin
			[acc1, acc3] = accuracy(v.zTopOut, v.labels, (1, 3))
			self.logDict({f'loss/Val{k}': v.loss, f'acc/Val{k}/Top1': acc1, f'acc/Val{k}/Top3': acc3})

	# ============
	# 测试阶段
	# ============

	@override
	def onTestStepVars(self, m: BaseVFLArch, v: StepVars) -> None:
		self.onTrainStepVars(m, v)

	@override
	def onTestLoss(self, m: BaseVFLArch, v: StepVars) -> None:
		[acc1, acc3] = accuracy(v.zTopOut, v.labels, (1, 3))
		self.logText.info(f'onTestLoss: Accuracy {acc1}, {acc3}')
