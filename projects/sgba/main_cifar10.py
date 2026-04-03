"""SGBA 攻击实验主模块"""

import time

from torch.optim import AdamW

from main.arch import BaseVFLArch
from main.callback import OPT_TYPE
from projects.lfba.infer import LFBAInferCb
from projects.vfl.config import AppConfig, ModelConfig, RunConfig, createLRS
from projects.vfl.core import VFLArch
from projects.vflip.method import VFLIPCb
from utils.common import L
from utils.config import getCallbacks, init
from utils.module import DataConfig, SplitDataModule

from .core import MethodArgs, SGBACb

# 方法参数
args = MethodArgs(
	fRecLr=5e-4,
	fTrainAlpha=0.3,
	fValAlpha=0.8,
	lLossScales=(1e-4, 1e-3),
	lGradScales=(10.0, 5.0),
)


def configOptims(m: BaseVFLArch, lr: float) -> OPT_TYPE:
	"""配置优化器和学习率调度器。

	Returns:
		优化器和学习率调度器列表
	"""
	optBtms = [AdamW(net.parameters(), lr) for net in m.lBtmNets]
	optTop = AdamW(m.zTopNet.parameters(), lr)

	lrsBtms = [createLRS(opt, milestones=[5, 20, 30], gamma=0.2) for opt in optBtms]
	lrsTop = createLRS(optTop, milestones=[5, 20, 30], gamma=0.2)
	return [*optBtms, optTop], [*lrsBtms, lrsTop]


def main(lTopDims: list[int]) -> None:
	"""运行 SGBA 实验主函数

	Args:
		lTopDims: 顶层网络结构维度列表
	"""
	# 实验配置
	data = DataConfig(sName='cifar10', nBatchSize=1024, nWorkers=16)
	model = ModelConfig(lPartyDims=[64] * 4, lTopDims=lTopDims)
	run = RunConfig(lr=0.001, epochs=40, _configOptims=configOptims)
	app = AppConfig(data, model, run, fpCkpt=None)

	# 模型架构
	arch = VFLArch(
		app,
		[
			LFBAInferCb(1096, 0.08, 0.70, 0.03),
			SGBACb(args, app),
			VFLIPCb(app.dpRoot, model.lPartyDims, 0.1, 0.1),
		],
	)

	# 数据模块
	module = SplitDataModule(len(model.lPartyDims), data)

	# 训练器
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
	iSeed = int(time.time())
	lSeed = [iSeed + i for i in range(3)]
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
