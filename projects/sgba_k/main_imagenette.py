"""SGBA 攻击实验主模块"""

import time

from torch.optim import Adam

from main.arch import BaseVFLArch
from main.callback import OPT_TYPE
from projects.lfba.infer import InferCb
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
	fSurLr=5e-4,
	fTrainAlpha=0.3,
	fValAlpha=0.75,
	lLossScales=(2e-4, 5e-3),
	lGradScales=(15.0, 5.0),
	milestones=[5, 15, 25, 35],
	gamma=0.8,
	lVicScales=(10, 10),
	lVicEntropyScales=(100, 100),
	lVicPartScales=(100, 100),
)


def configOptims(m: BaseVFLArch, lr: float) -> OPT_TYPE:
	"""配置优化器和学习率调度器。"""
	optBtms = [Adam(net.parameters(), lr) for net in m.lBtmNets]
	optTop = Adam(m.zTopNet.parameters(), lr)

	lrsBtms = [createLRS(opt, milestones=args.milestones, gamma=args.gamma) for opt in optBtms]
	lrsTop = createLRS(optTop, milestones=args.milestones, gamma=args.gamma)
	return [*optBtms, optTop], [*lrsBtms, lrsTop]


def main() -> None:
	"""进行 SGBA 实验。"""
	# 实验配置
	data = DataConfig(sName='imagenette', nBatchSize=128, nWorkers=4)
	model = ModelConfig(lPartyDims=[32] * 8, lTopDims=[256, 256, 10])
	run = RunConfig(lr=0.001, epochs=40, _configOptims=configOptims)
	app = AppConfig(data, model, run, fpCkpt=None)

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
	iSeed = int(time.time())
	lSeed = [iSeed + i for i in range(5)]
	for seed in lSeed:
		init(seed)
		main()

	print('运行结束！')
