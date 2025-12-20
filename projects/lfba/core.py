"""LFBA 架构模块

本模块实现了基于标签翻转的后门攻击方法，包括本地模型、学习率调度器、
触发器转换以及完整的 VFL 架构实现。
"""

from typing import Any, override

from torch.optim import Adam, Optimizer, lr_scheduler as lr
from torchvision import models

from main.arch import BaseVFLArch
from main.callback import VFLCallback
from models.fcn import FCN
from utils.common import Path, copy, nn, np, tc
from utils.config import rng
from utils.define import StepVars
from utils.misc import accuracy
from utils.module import BaseDataModule
from utils.vision import createPostTrans, createSpatialTrans, tf


class LocalModel(nn.Module):
	"""本地模型类，用于垂直联邦学习中的各参与方

	该类基于 ResNet-18 构建本地模型网络，负责处理本地数据并提取特征。
	"""

	def __init__(self, nOutDim: int) -> None:
		"""初始化本地模型。

		Args:
			nOutDim: 输出维度，即特征向量的长度
		"""
		super().__init__()
		self.net = models.resnet18(num_classes=nOutDim)

	@override
	def forward(self, ins: tc.Tensor) -> tc.Tensor:
		return self.net(ins)


def createLRScheduler(optimizer: Optimizer) -> lr.LRScheduler:
	"""创建学习率调度器链。

	组合使用线性学习率预热和多步学习率衰减策略。

	Args:
		optimizer: 需要应用学习率调度的优化器

	Returns:
		组合后的学习率调度器
	"""
	scheduler1 = lr.LinearLR(optimizer, 0.1, total_iters=5)
	scheduler2 = lr.MultiStepLR(optimizer, [20, 30], 0.2)
	return lr.ChainedScheduler([scheduler1, scheduler2], optimizer)


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


class LFBAArch(BaseVFLArch):
	"""LFBA 架构实现类

	实现 LFBA 攻击的垂直联邦学习架构，包括模型训练、验证和攻击逻辑。
	"""

	def __init__(
		self,
		module: BaseDataModule,
		dpRoot: Path | str,
		lPartyDims: list[int],
		lTopDims: list[int],
		lCallbacks: list[VFLCallback] | None = None,
	) -> None:
		"""初始化实例。

		Args:
			module: 数据模块，提供训练和验证数据
			dpRoot: 日志和检查点保存根目录
			lPartyDims: 各参与方的输出维度列表
			lTopDims: 顶层网络的维度列表
			lCallbacks: 回调函数列表，_可选_
		"""
		super().__init__(dpRoot, lCallbacks)
		self.logText.info('LFBAArch.__init__()')

		self.lPartyDims = lPartyDims
		self.lTopDims = lTopDims
		self.module = module

		# 必要属性
		self.lBtmNets = nn.ModuleList([LocalModel(dim) for dim in lPartyDims])  #! 可变
		self.zTopNet = FCN(lTopDims, True, 'relu')

		# 数据转换
		self.tfNormal = createSpatialTrans(module.lNormalTrans)
		self.tfAugment = createSpatialTrans(module.lAugmentTrans)
		self.tfTrigger = AddTrigger()
		self.tfPost = createPostTrans()

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
	def onFitStart(self, m: BaseVFLArch) -> None:
		self.logText.info(self.trainer.log_dir)

	@override
	def onTrainStepVars(self, m: BaseVFLArch, v: StepVars) -> None:
		[v.images, v.labels, v.indices] = v.batch

	@override
	def onTrainBtmIns(self, m: 'BaseVFLArch', v: StepVars) -> None:
		v.lBtmIns = [self.tfAugment(images) for images in v.lBtmIns]

		if self.current_epoch >= 1:
			# 获取当前批次中投毒目的样本、非目标类样本的位置（索引的索引）
			aBatchIdxs = v.indices.cpu().numpy()
			aDstPos = np.flatnonzero(np.isin(aBatchIdxs, self.ns.aDstIdxs))  #: aBatchIdxs_DstPos
			aNonPos = np.flatnonzero(np.isin(aBatchIdxs, self.ns.aNonIdxs))  #: aBatchIdxs_NonPos

			if len(aDstPos) > 0 and len(aNonPos) > 0:  # 如果找到投毒目标
				aSrcPos = rng().choice(aNonPos, len(aDstPos), len(aNonPos) < len(aDstPos))
				for dst, src in zip(aDstPos, aSrcPos, strict=True):
					v.lBtmIns[0][dst] = self.tfTrigger(v.lBtmIns[0][src])

		v.lBtmIns = [self.tfPost(images) for images in v.lBtmIns]

	@override
	def onTrainLoss(self, m: BaseVFLArch, v: StepVars) -> None:
		self.logDict({'loss/Training': v.loss})

	# ============
	# 验证阶段
	# ============

	@override
	def onValStepVars(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		v = d['Origin']
		[v.images, v.labels, _] = v.batch

		d['Attack'] = copy(d['Origin'])

	@override
	def onValBtmIns(self, m: 'BaseVFLArch', d: dict[str, StepVars]) -> None:
		v = d['Origin']
		v.lBtmIns = [self.tfNormal(images) for images in v.lBtmIns]
		v.lBtmIns = [self.tfPost(images) for images in v.lBtmIns]

		v = d['Attack']
		v.lBtmIns = [self.tfNormal(images) for images in v.lBtmIns]
		v.lBtmIns[0] = self.tfTrigger(v.lBtmIns[0])
		v.lBtmIns = [self.tfPost(images) for images in v.lBtmIns]

	@override
	def onValLoss(self, m: BaseVFLArch, d: dict[str, StepVars]) -> None:
		for k, v in d.items():  # * Origin, Attack, Defend
			[acc1, acc3] = accuracy(v.zTopOut, v.labels, (1, 3))
			self.logDict({f'loss/Val{k}': v.loss, f'acc/Val{k}/Top1': acc1, f'acc/Val{k}/Top3': acc3})

			# probs = tc.nn.functional.softmax(v.zTopOut, 1)
			# entropy = tc.distributions.Categorical(probs).entropy().mean().item()
			# self.logDict({f'entropy/Val{k}': entropy})

			mask = v.labels != self.ns.iTgtLabel
			if not mask.any():
				continue

			labels = v.labels[mask]
			zTopOut = v.zTopOut[mask]

			tTgtLabels = tc.full_like(labels, self.ns.iTgtLabel)
			[accT1, accT3] = accuracy(zTopOut, tTgtLabels, (1, 3))
			lossT = self.criterion(zTopOut, tTgtLabels)

			sName = f'Val{k}/Tgt'
			self.logDict({f'loss/{sName}': lossT, f'acc/{sName}/Top1': accT1, f'acc/{sName}/Top3': accT3})
