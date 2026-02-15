"""SGBA 攻击实验主模块"""

import time

from projects.lfba.infer import InferCb
from projects.vfl.config import AppConfig, ModelConfig, RunConfig
from projects.vfl.core import VFLArch
from projects.vflip.method import VFLIPCb
from utils.common import L
from utils.config import getCallbacks, init
from utils.module import DataConfig, SplitDataModule

from .core import MethodArgs, SGBACb


def main() -> None:
	"""进行 SGBA 实验。"""
	# 实验配置
	data = DataConfig(sName='imagenette', nBatchSize=128, nWorkers=4)
	model = ModelConfig(lPartyDims=[32] * 8, lTopDims=[256, 256, 10])
	run = RunConfig(lr=0.001, epochs=40)
	app = AppConfig(data, model, run, fpCkpt=None)

	# 方法参数
	args = MethodArgs(
		fRecLr=5e-4,
		fSurLr=1e-3,
		fTrainAlpha=0.3,
		fValAlpha=0.8,
		lLossScales=(1e-4, 10e-3),
		lGradScales=(15.0, 5.0),
	)

	# 模型架构
	arch = VFLArch(
		app,
		[
			InferCb(0.03),
			SGBACb(args, app),
			VFLIPCb(app.dpRoot, model.lPartyDims, 0.01, 0.01),
		],
	)

	# 数据模块
	module = SplitDataModule(len(model.lPartyDims), data)

	# 训练器
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
	for seed in lSeed:
		init(seed)
		main()

	print('运行结束！')
