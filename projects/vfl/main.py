"""垂直联邦学习（VFL）模型训练主程序。

本模块实现了基于 CINIC-10 数据集的垂直联邦学习模型训练流程，包括数据加载、
模型初始化、训练器配置以及模型的训练和验证过程。
"""

import time

from modules.cifar10 import Handler
from utils.common import L, Path
from utils.config import getCallbacks, init
from utils.module import DataConfig, SplitDataModule

from .core import VFLArch


def main() -> None:
	"""垂直联邦学习模型训练主函数。

	执行垂直联邦学习模型的完整训练流程：
	1. 初始化训练配置和日志记录
	2. 创建 VFL 架构实例
	3. 配置数据模块和数据加载器
	4. 初始化训练器
	5. 执行模型训练
	6. 执行模型验证（使用最佳模型权重）
	"""
	dpRoot = Path('data/logs/vfl_cifar10')
	dpData = Path('data/datasets/cifar10')
	fpCkpt = None
	# fpCkpt = ''
	lPartyDims = [128, 128]

	# 模型架构
	arch = VFLArch(dpRoot, lPartyDims)

	# 数据模块
	config = DataConfig(dpData, 128, 8)
	module = SplitDataModule(len(lPartyDims), config, Handler())

	trainer = L.Trainer(
		deterministic=True,
		max_epochs=40,
		default_root_dir=dpRoot,
		callbacks=getCallbacks(),
	)
	trainer.fit(arch, datamodule=module, ckpt_path=fpCkpt)
	trainer.validate(arch, datamodule=module, ckpt_path='best')


if __name__ == '__main__':
	init(int(time.time()))
	main()
	print('运行结束！')
