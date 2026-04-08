import pathlib

import numpy as np
import torch as tc
from scipy.stats import gaussian_kde


def get_kde(fpData: str, iEpoch: int, iTargetClass: int = 0, optimal_m1: float = 0.02) -> None:
	print(f'Loading data from: {fpData}')
	all_data = tc.load(fpData)

	# 1. 提取指定 Epoch 的数据
	data = all_data[iEpoch]

	# 2. 数据清洗：屏蔽目标类别
	tMask = data['labels'] != iTargetClass
	tIsAttack = data['is_attack'][tMask]  # * (N)
	tEntropy = data['entropy'][tMask]  # * (N, nParty)
	tMeanEntropy = data['mean_entropy']  # * (nParty)

	# 3. 锁定攻击方 (Party 0) 并划分干净样本与投毒样本
	aEntropy = tEntropy[:, 0].numpy()
	aIsAttack = tIsAttack.numpy()
	aCleanEntropy = aEntropy[aIsAttack == 0] + 1e-8
	aPpoisonedEntropy = aEntropy[aIsAttack == 1] + 1e-8

	# 4. 计算真实环境下的绝对物理阈值
	absolute_threshold = optimal_m1 * tMeanEntropy[0]
	print(f'Global Entropy Mean: {tMeanEntropy[0]:.4f}')
	print(f'Absolute Cut-off Threshold (m1={optimal_m1}): {absolute_threshold:.4f}')

	kdeClean = gaussian_kde(aCleanEntropy)
	Xc = np.linspace(0, 2.5, 501)
	Yc = kdeClean(Xc)
	np.savetxt('kde_clean.txt', np.column_stack((Xc, Yc)), fmt='%.5f', header='x y')

	kdePoisoned = gaussian_kde(aPpoisonedEntropy)
	Xp = np.linspace(0, 1.00, 501)
	Yp = kdePoisoned(Xp)
	np.savetxt('kde_poisoned.txt', np.column_stack((Xp, Yp)), fmt='%.5f', header='x y')

	print('KDE coordinates saved for TikZ!')


if __name__ == '__main__':
	FILE = 'data/logs/sgba_cifar10/lightning_logs/version_81/dEpochRecords.pt'
	get_kde(FILE, iEpoch=39, iTargetClass=5, optimal_m1=0.02)
