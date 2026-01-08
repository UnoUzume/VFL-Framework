"""软标签推理攻击

参与者数量为 1：Train 95%/97%，Val 83%/84%
参与者数量为 2：Train 74%/93%，Val 64%/75%
"""

import time

from projects.vfl.config import AppConfig, ModelConfig, RunConfig
from projects.vfl.core import VFLArch
from utils.common import L
from utils.config import getCallbacks, init
from utils.module import DataConfig

from .core import SoftLabelCb
from .module import CustomDataModule


def main(lTopDims: list[int]) -> None:
	"""运行 LFBA 实验主函数

	Args:
		lTopDims: 顶层网络结构维度列表
	"""
	# 配置
	data = DataConfig('cifar10', 128, 4)
	model = ModelConfig([128, 128], lTopDims)
	run = RunConfig(0.001, 40)
	app = AppConfig(data, model, run, fpCkpt=None)

	# 模型架构
	arch = VFLArch(
		app,
		[SoftLabelCb(app)],
	)

	# 数据模块
	module = CustomDataModule(len(model.lPartyDims), data)

	trainer = L.Trainer(
		deterministic=True,
		max_epochs=arch.cfg.run.epochs,
		default_root_dir=arch.cfg.dpRoot,
		callbacks=getCallbacks(),
	)
	trainer.fit(arch, datamodule=module, ckpt_path=app.fpCkpt)
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
