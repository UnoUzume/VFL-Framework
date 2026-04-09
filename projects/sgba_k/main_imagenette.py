"""SGBA 攻击实验主模块"""

import time

from torch.optim import AdamW

from main.arch import BaseVFLArch
from main.callback import OPT_TYPE
from projects.lfba.infer import LFBAInferAllCb
from projects.vfl.config import AppConfig, ModelConfig, RunConfig, createLRS
from projects.vfl.core import VFLArch
from projects.vflip.method import VFLIPCb
from utils.common import L
from utils.config import getCallbacks, init
from utils.module import DataConfig, SplitDataModule

from .core import MethodArgs, SGBACb


def configOptims(m: BaseVFLArch, lr: float) -> OPT_TYPE:
	"""配置优化器和学习率调度器。

	Returns:
		优化器和学习率调度器列表
	"""
	optBtms = [AdamW(net.parameters(), lr=lr) for net in m.lBtmNets]
	optTop = AdamW(m.zTopNet.parameters(), lr=lr)

	lrsBtms = [createLRS(opt, milestones=args.milestones, gamma=args.gamma) for opt in optBtms]
	lrsTop = createLRS(optTop, milestones=args.milestones, gamma=args.gamma)
	return [*optBtms, optTop], [*lrsBtms, lrsTop]


def main(app: AppConfig, args: MethodArgs) -> None:
	"""进行实验。"""
	# 模型架构
	arch = VFLArch(
		config=app,
		lCallbacks=[
			LFBAInferAllCb(rSel=0.03),
			SGBACb(args=args, config=app),
			VFLIPCb(dpRoot=app.dpRoot, lPartyDims=app.model.lPartyDims, M=0.01, N=0.01),
		],
	)

	# 数据模块
	module = SplitDataModule(nParty=len(app.model.lPartyDims), config=app.data)

	# 训练器
	trainer = L.Trainer(
		deterministic=True,
		max_epochs=arch.cfg.run.epochs,
		default_root_dir=arch.cfg.dpRoot,
		log_every_n_steps=30,
		callbacks=getCallbacks(),
	)
	trainer.fit(model=arch, datamodule=module, ckpt_path=app.fpCkpt)
	trainer.validate(model=arch, datamodule=module, ckpt_path='best')


if __name__ == '__main__':
	iSeed = int(time.time())
	lSeeds = [iSeed + i for i in range(3)]
	for seed in lSeeds:
		# 设置随机种子
		init(seed)

		# 设置方法参数
		args = MethodArgs(
			fRecLr=2e-4,
			fSurLr=2e-4,
			fTrainAlpha=0.3,
			fValAlpha=0.7,
			lGradScales=(15.0, 5.0),
			lLossScales=(2e-4, 2e-3),
			milestones=[5, 50],
			gamma=0.8,
			lVicScales=(150, 150),  # 150
			lVicEntropyScales=(300, 300),  # 150
			lVicPartScales=(1000, 1000),  # 1000
			fSurLambda1=1,
			fSurLambda2=1e4,  # 1e6
		)

		# 设置实验参数
		data = DataConfig(sName='imagenette', nBatchSize=256, nWorkers=8)
		model = ModelConfig(lPartyDims=[32] * 8, lTopDims=[256, 256, 10])
		run = RunConfig(lr=1e-3, epochs=40, _configOptims=configOptims)
		app = AppConfig(data, model, run, fpCkpt=None)

		# 开始实验
		main(app=app, args=args)

	print('运行结束！')
