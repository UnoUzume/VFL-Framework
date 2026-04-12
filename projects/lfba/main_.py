"""LFBA 攻击实验主模块"""

import time

from models.fcn import FCN
from projects.vfl.config import AppConfig, ModelConfig, RunConfig
from utils.common import nn
from utils.config import init
from utils.module import DataConfig

from .main import configOptims, main


def getBtmNets() -> nn.ModuleList:
	"""获取各参与方的底端网络列表。

	Returns:
		各参与方的底端网络模块列表
	"""
	total = 1634
	n = 4
	base = total // n
	rem = total % n
	lDims = [base + (1 if i < rem else 0) for i in range(n)]
	print(lDims)
	return nn.ModuleList([FCN([dim, 256, 256, 64]) for dim in lDims])


if __name__ == '__main__':
	iSeed = int(time.time())
	lSeeds = [iSeed + i for i in range(3)]
	for seed in lSeeds:
		# 设置随机种子
		init(seed)

		# 设置实验参数
		data = DataConfig(sName='nuswide', nBatchSize=1024, nWorkers=16, enableTrans=False)
		model = ModelConfig(lPartyDims=[64] * 4, lTopDims=[256, 256, 10], _getBtmNets=getBtmNets)
		run = RunConfig(lr=1e-3, epochs=40, _configOptims=configOptims)
		app = AppConfig(data, model, run, fpCkpt=None)

		# 开始实验
		main(app)

	print('运行结束！')
