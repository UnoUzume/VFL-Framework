"""Villain 攻击实验主模块"""

import time

from torch.optim import AdamW

from main.arch import BaseVFLArch
from main.callback import OPT_TYPE
from models.fcn import FCN
from projects.lfba.infer import LFBAInferCb
from projects.vfl.config import AppConfig, ModelConfig, RunConfig, createLRS
from projects.vfl.core import VFLArch
from projects.vflip.method import VFLIPCb
from utils import define as de
from utils.common import L, nn
from utils.config import getCallbacks, init
from utils.module import DataConfig, SplitDataModule

from .core import VillainCb


def configOptims(m: BaseVFLArch, lr: float) -> OPT_TYPE:
	"""配置优化器和学习率调度器。

	Returns:
		优化器和学习率调度器列表
	"""
	optBtms = [AdamW(net.parameters(), lr=lr) for net in m.lBtmNets]
	optTop = AdamW(m.zTopNet.parameters(), lr=lr)

	lrsBtms = [createLRS(opt, milestones=[5, 20], gamma=0.4) for opt in optBtms]
	lrsTop = createLRS(optTop, milestones=[5,  20], gamma=0.4)
	return [*optBtms, optTop], [*lrsBtms, lrsTop]


def get_vfl_dims(total: int, n: int) -> list[int]:
	"""获取 VFL 每一方的维度分布。

	Returns:
		维度分布
	"""
	base = total // n
	rem = total % n
	return [base + (1 if i < rem else 0) for i in range(n)]


def getBtmNets() -> nn.ModuleList:
	"""获取各参与方的底端网络列表。

	Returns:
		各参与方的底端网络模块列表
	"""
	lInputDims = get_vfl_dims(1634, 4)
	return nn.ModuleList([FCN([dim, 256,256, 64]) for dim in lInputDims])


def main(app: AppConfig) -> None:
	"""进行实验。"""
	# 模型架构
	arch = VFLArch(
		config=app,
		lCallbacks=[
			LFBAInferCb(iAncIdx=1096, rTgt=0.08, rVic=0.70, rSel=0.03),
			VillainCb(),
			VFLIPCb(dpRoot=app.dpRoot, lPartyDims=app.model.lPartyDims, M=0.05, N=0.05),
		],
	)

	# 数据模块
	module = SplitDataModule[de.TFeatureSample, de.TFeatureBatch, de.TSplitFeatureBatch](
		nParty=len(app.model.lPartyDims), config=app.data
	)

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
		data = DataConfig(sName='nuswide', nBatchSize=1024, nWorkers=16, enableTrans=False)
		model = ModelConfig(lPartyDims=[64] * 4, lTopDims=[256, 256, 10],_getBtmNets=getBtmNets)
		run = RunConfig(lr=1e-3, epochs=40, _configOptims=configOptims)
		app = AppConfig(data, model, run, fpCkpt=None)

		# 开始实验
		main(app=app)

	print('运行结束！')
