"""垂直联邦学习（VFL）架构实现模块。

本模块实现了垂直联邦学习的具体架构，包括底层模型和顶层模型的定义、
优化器配置以及训练、验证和测试过程中的回调处理。
"""

from typing import override

from torch.optim import Adam, Optimizer, lr_scheduler as lr

from main.arch import BaseVFLArch
from main.callback import VFLCallback
from utils.common import Path, nn
from utils.define import StepVars
from utils.misc import accuracy

from .model import BottomModel, TopModel


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
	"""垂直联邦学习架构实现类。

	继承自 `BaseVFLArch`，实现了具体的垂直联邦学习模型结构和训练逻辑。

	Args:
		dpRoot: 模型保存和日志记录的根目录路径
		lPartyDims: 各参与方底层模型的输出维度列表
		lCallbacks: 额外的回调函数列表，_可选_
	"""

	def __init__(
		self, dpRoot: Path | str, lPartyDims: list[int], lCallbacks: list[VFLCallback] | None = None
	) -> None:
		"""初始化 VFL 架构实例。

		Args:
			dpRoot: 模型保存和日志记录的根目录路径
			lPartyDims: 各参与方底层模型的输出维度列表
			lCallbacks: 额外的回调函数列表，_可选_
		"""
		super().__init__(dpRoot, lCallbacks)
		self.logText.info('VFLArch.__init__()')

		self.lPartyDims = lPartyDims
		"""各参与方底层模型的输出维度列表"""

		self.lBtmNets = nn.ModuleList([BottomModel(dim) for dim in lPartyDims])  #! 可变
		self.zTopNet = TopModel()

	@override
	def onConfigOptims(self, iOpt: int, iLRS: int) -> tuple[list[Optimizer], list[lr.LRScheduler]]:
		LR = 1e-3
		optBtms = [Adam(net.parameters(), LR) for net in self.lBtmNets]
		optTop = Adam(self.zTopNet.parameters(), LR)

		lrsBtms = [createLRScheduler(opt) for opt in optBtms]
		lrsTop = createLRScheduler(optTop)
		return [*optBtms, optTop], [*lrsBtms, lrsTop]

	# ============
	# 训练阶段
	# ============

	@override
	def onTrainStepVars(self, m: BaseVFLArch, v: StepVars) -> None:
		[v.images, v.labels, _] = v.batch

	@override
	def onTrainLoss(self, m: BaseVFLArch, v: StepVars) -> None:
		self.logDict({'loss/Training': v.loss})

	# ============
	# 验证阶段
	# ============

	@override
	def onValStepVars(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		for v in d.values():  # * Origin
			[v.images, v.labels, _] = v.batch

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
