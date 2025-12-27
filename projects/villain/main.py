"""Villain 攻击实验主模块"""

import time

from torch.optim import Adam

from main.arch import BaseVFLArch
from main.callback import OPT_TYPE
from projects.lfba.method import LFBAInferCb
from projects.vfl.config import AppConfig, ModelConfig, RunConfig, createLRS
from projects.vfl.core import VFLArch
from utils.common import L
from utils.config import getCallbacks, init
from utils.module import DataConfig, SplitDataModule

from .core import VillainCb


def configOptims(m: BaseVFLArch, lr: float) -> OPT_TYPE:
	"""配置优化器和学习率调度器。

	Args:
		m: 模型架构对象，包含底端网络和顶端网络
		lr: 默认学习率
	Returns:
		包含优化器和学习率调度器的元组
	"""
	optBtms = [Adam(m.lBtmNets[0].parameters(), lr * 2)]
	optBtms.extend([Adam(net.parameters(), lr) for net in m.lBtmNets[1:]])
	optTop = Adam(m.zTopNet.parameters(), lr)

	lrsBtms = [createLRS(optBtms[0], [5, 7, 20], 0.5)]
	lrsBtms.extend([createLRS(opt, [5, 20, 30]) for opt in optBtms[1:]])
	lrsTop = createLRS(optTop, [5, 20, 30])
	return [*optBtms, optTop], [*lrsBtms, lrsTop]


def main(lTopDims: list[int]) -> None:
	"""运行 Villain 实验主函数

	Args:
		lTopDims: 顶层网络结构维度列表
	"""
	# 配置
	data = DataConfig('cifar10', 128, 4)
	model = ModelConfig([128, 128], lTopDims)
	run = RunConfig(0.001, 40, configOptims)
	app = AppConfig(data, model, run, fpCkpt=None)

	# 模型架构
	arch = VFLArch(
		app,
		[
			LFBAInferCb(1096, 0.08, 0.70, 0.03),
			VillainCb(),
		],
	)

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
