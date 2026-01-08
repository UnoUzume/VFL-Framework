"""SGBA 攻击实验主模块"""

import time

from projects.lfba.method import LFBAInferCb
from projects.vfl.config import AppConfig, ModelConfig, RunConfig
from projects.vfl.core import VFLArch
from projects.vflip.method import VFLIPCb
from utils.common import L
from utils.config import getCallbacks, init
from utils.module import DataConfig, SplitDataModule

from .core import MethodArgs, SGBACb


def main(lTopDims: list[int]) -> None:
	"""运行 SGBA 实验主函数

	Args:
		lTopDims: 顶层网络结构维度列表
	"""
	# 配置
	data = DataConfig('imagenette', 256, 10)
	model = ModelConfig([64] * 4, lTopDims)
	run = RunConfig(0.001, 40)
	app = AppConfig(data, model, run, fpCkpt=None)

	# 方法参数
	args = MethodArgs(
		fRecLr=5e-4, fTrainAlpha=0.3, fValAlpha=0.8, lLossScales=(1e-4, 10e-3), lGradScales=(15.0, 5.0)
	)

	# 模型架构
	arch = VFLArch(
		app,
		[
			LFBAInferCb(1096, 0.09, 0.80, 0.03),
			SGBACb(args, app),
			VFLIPCb(app.dpRoot, model.lPartyDims, 0.01, 0.01),
		],
	)

	# 数据模块
	module = SplitDataModule(len(model.lPartyDims), data)

	trainer = L.Trainer(
		deterministic=True,
		max_epochs=arch.cfg.run.epochs,
		default_root_dir=arch.cfg.dpRoot,
		log_every_n_steps=30,
		callbacks=getCallbacks(),
	)
	trainer.fit(arch, datamodule=module, ckpt_path=app.fpCkpt)
	trainer.validate(arch, datamodule=module, ckpt_path='best')


if __name__ == '__main__':
	lSeed = [int(time.time()) + 0x1000 * x for x in range(7)]
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
