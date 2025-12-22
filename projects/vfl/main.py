"""纵向联邦学习（VFL）模型训练主程序。

本模块实现了基于 CINIC-10 数据集的纵向联邦学习模型训练流程，包括数据加载、
模型初始化、训练器配置以及模型的训练和验证过程。
"""

import time

from utils.common import L
from utils.config import getCallbacks, init
from utils.module import DataConfig, SplitDataModule

from .config import AppConfig, ModelConfig, RunConfig
from .core import VFLArch


def main() -> None:
	"""训练 VFL 模型。"""
	# 配置
	data = DataConfig('cifar10', 128, 4)
	model = ModelConfig([128, 128], [256, 256, 128, 10])
	run = RunConfig(0.001, 40)
	app = AppConfig(data, model, run, fpCkpt=None)

	# 模型架构
	arch = VFLArch(app, [])

	# 数据模块
	module = SplitDataModule(len(model.lPartyDims), data)

	trainer = L.Trainer(
		deterministic=True,
		max_epochs=arch.cfg.run.epochs,
		default_root_dir=arch.cfg.dpRoot,
		callbacks=getCallbacks(),
	)
	trainer.fit(arch, datamodule=module, ckpt_path=app.fpCkpt)
	trainer.validate(arch, datamodule=module, ckpt_path='best')


if __name__ == '__main__':
	init(int(time.time()))
	main()
	print('运行结束！')
