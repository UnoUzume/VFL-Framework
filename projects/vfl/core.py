"""纵向联邦学习（VFL）架构实现模块。

本模块实现了纵向联邦学习的具体架构，包括底层模型和顶层模型的定义、
优化器配置以及训练、验证和测试过程中的回调处理。
"""

from typing import override

from main.arch import BaseVFLArch
from main.callback import OPT_TYPE, VFLCallback
from utils.define import StepVars
from utils.misc import accuracy

from .config import AppConfig


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
	def onConfigOptims(self) -> OPT_TYPE:
		return self.cfg.run.configOptims(self)

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
