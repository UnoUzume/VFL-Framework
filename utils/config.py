# pyright: reportConstantRedefinition=false


"""配置工具模块。

该模块提供了随机种子初始化、日志过滤和 PyTorch Lightning 回调配置等功能，用于统一管理实验配置。
"""

import logging
import types

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.callback import Callback

from .common import L, np, tc

SEED: int | None = None
"""全局随机种子"""
RNG: np.random.Generator | None = None
"""全局随机数生成器"""


class IgnorePLFilter(logging.Filter):
	"""忽略 PyTorch Lightning 特定日志信息的过滤器。

	用于过滤掉 PyTorch Lightning 输出的一些冗余日志信息，如 CUDA 可用性、LOCAL_RANK 等。

	参考：<https://github.com/Lightning-AI/pytorch-lightning/issues/3431>
	"""

	def filter(self, record: logging.LogRecord) -> bool:
		"""过滤日志记录。

		Args:
			record: 日志记录对象

		Returns:
			是否保留该日志记录，`True` 表示保留，`False` 表示过滤掉
		"""
		keywords = ['available:', 'CUDA', 'LOCAL_RANK:']
		return not any(keyword in record.getMessage() for keyword in keywords)


def init(seed: int = 0x0721) -> None:
	"""初始化随机种子和配置环境。

	设置全局随机种子，配置 PyTorch 浮点计算精度，并应用日志过滤器。

	Args:
		seed: 随机种子值，默认为 `0x0721`
	"""
	global SEED, RNG  # noqa: PLW0603

	# 设置随机种子
	SEED = seed  # * 备选项：0x0042 0x0721 0x1096 0x1314 0x4869
	RNG = np.random.default_rng(SEED)
	L.seed_everything(SEED, True, False)
	tc.set_float32_matmul_precision('high')
	print(f'随机种子设置为：{SEED:#06x}')

	# 添加白名单
	tc.serialization.add_safe_globals(
		[
			types.SimpleNamespace,
			np.ndarray,
			np.dtype,
			np.dtypes.Int64DType,
			np._core.multiarray._reconstruct,  # type: ignore[attr-defined]  # noqa: SLF001
		]
	)

	# 配置日志过滤
	logging.getLogger('lightning.pytorch.utilities.rank_zero').addFilter(IgnorePLFilter())
	logging.getLogger('lightning.pytorch.accelerators.cuda').addFilter(IgnorePLFilter())


def rng() -> np.random.Generator:
	"""获取全局随机数生成器。

	Returns:
		全局随机数生成器实例

	Raises:
		AssertionError: 当未调用 `init()` 函数初始化随机种子时抛出
	"""
	assert RNG is not None, '请先调用 `init()` 函数初始化随机种子'
	return RNG


def getCallbacks() -> list[Callback]:
	"""获取 PyTorch Lightning 训练回调列表。

	Returns:
		包含回调对象的列表。
	"""
	cbCkptAcc = ModelCheckpoint(
		filename='epoch={epoch}-acc_val={acc/ValOrigin/Top1:.4f}',
		monitor='acc/ValOrigin/Top1',
		save_top_k=1,
		mode='max',
		auto_insert_metric_name=False,
	)
	cbCkptEpoch = ModelCheckpoint(every_n_epochs=5, save_top_k=-1, save_last='link')
	cbLRMonitor = LearningRateMonitor()

	#! 注意：`validate(ckpt_path="best")` 仅考虑第一个 `ModelCheckpoint`
	return [cbCkptAcc, cbCkptEpoch, cbLRMonitor]


__all__ = ['getCallbacks', 'init', 'rng']
