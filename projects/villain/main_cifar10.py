"""Villain 攻击实验主模块"""

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

from .core import VillainCb


def configOptims(m: BaseVFLArch, lr: float) -> OPT_TYPE:
	"""配置优化器和学习率调度器。

	Returns:
		优化器和学习率调度器列表
	"""
	optBtms = [AdamW(m.lBtmNets[0].parameters(), lr=lr * 2)]
	optBtms.extend([AdamW(net.parameters(), lr=lr) for net in m.lBtmNets[1:]])
	optTop = AdamW(m.zTopNet.parameters(), lr=lr)

	lrsBtms = [createLRS(optBtms[0], milestones=[5, 10, 20, 30], gamma=0.4)]
	lrsBtms.extend([createLRS(opt, milestones=[5, 10, 20, 30], gamma=0.4) for opt in optBtms[1:]])
	lrsTop = createLRS(optTop, milestones=[5, 10, 20, 30], gamma=0.4)
	return [*optBtms, optTop], [*lrsBtms, lrsTop]


def main(app: AppConfig) -> None:
	"""进行实验。"""
	# 模型架构
	arch = VFLArch(
		config=app,
		lCallbacks=[
			LFBAInferCb(iAncIdx=1096, rTgt=0.08, rVic=0.70, rSel=0.03),
			VillainCb(),
			VFLIPCb(dpRoot=app.dpRoot, lPartyDims=app.model.lPartyDims, M=0.03, N=0.02),
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

		# 设置实验参数
		data = DataConfig(sName='cifar10', nBatchSize=1024, nWorkers=16)
		model = ModelConfig(lPartyDims=[64] * 4, lTopDims=[256, 256, 10])
		run = RunConfig(lr=1e-3, epochs=40, _configOptims=configOptims)
		app = AppConfig(data, model, run, fpCkpt=None)

		# 开始实验
		main(app)

	print('运行结束！')
