"""LFBA 实验主程序模块

本模块实现了 LFBA 实验流程，包括数据加载、模型训练和验证等功能。
"""

# from projects.vflip.vflip import VFLIPCb
import time

from modules.cifar10 import Handler
from utils.common import L, Path
from utils.config import getCallbacks, init
from utils.module import DataConfig, SplitDataModule

from .core import LFBAArch
from .methods import LFBAInferCb

lPartyDims = [128, 128]
dpRoot = Path(f'data/logs/lfba_P{len(lPartyDims)}L2')
dpData = Path('data/datasets/cifar10')
fpCkpt: str | None = None
# fpCkpt = ''

# 数据模块
config = DataConfig(dpData, 128, 8, enableTrans=False)
module = SplitDataModule(len(lPartyDims), config, Handler())


def main(lTopDims: list[int]) -> None:
	"""运行 LFBA 实验主函数

	Args:
		lTopDims: 顶层网络结构维度列表
	"""
	arch = LFBAArch(
		module,
		dpRoot,
		lPartyDims,
		lTopDims,
		[
			LFBAInferCb(1096, 0.08, 0.70, 0.03),
			# VFLIPCb(dpRoot, lPartyDims, 0.05, 0.05),
		],
	)

	trainer = L.Trainer(
		deterministic=True,
		max_epochs=40,
		default_root_dir=dpRoot,
		callbacks=getCallbacks(),
	)
	trainer.fit(arch, datamodule=module, ckpt_path=fpCkpt)
	trainer.validate(arch, datamodule=module, ckpt_path='best')


if __name__ == '__main__':
	lSeed = [int(time.time())]
	llDims = [
		# [256, 10],
		[256, 256, 10],
		# [256, 256, 256, 10],
		# [256, 256, 256, 256, 10],
	]

	for lDims in llDims:
		for seed in lSeed:
			init(seed)
			main(lDims)

	print('运行结束！')
