import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def analyze_all_epochs(file_path: str, epoch_to_eval: int, target_class: int = 0) -> None:
	# 1. 加载包含所有数据的字典
	all_data = torch.load(file_path)

	if epoch_to_eval not in all_data:
		print(f'Epoch {epoch_to_eval} not found in {file_path}')
		return

	# 2. 提取指定 Epoch 的数据
	data = all_data[epoch_to_eval]

	valid_mask = data['labels'] != target_class
	tEntropy = data['entropy'][valid_mask]  # * (N, nParty)
	tIsAttack = data['is_attack'][valid_mask]  # * (N)
	tMeanEntropy = data['mean_entropy']  # * (nParty)

	# 3. 构造全局 Y_true
	Yt = torch.zeros_like(tEntropy, dtype=torch.bool)
	Yt[:, 0] = tIsAttack == 1

	# --- [网格搜索逻辑与之前一致] ---
	m1_list = np.arange(0.005, 0.105, 0.005)
	m2_list = np.arange(0.005, 0.105, 0.005)
	f1_matrix = np.zeros((len(m1_list), len(m2_list)))

	print('Starting Grid Search...')
	# 4. 执行网格搜索
	for i, m1 in enumerate(m1_list):
		for j, m2 in enumerate(m2_list):
			temp1 = tEntropy < m1 * tMeanEntropy
			temp2 = tEntropy < m2 * tEntropy.amax(dim=1, keepdim=True)
			Yp = temp1 & temp2  # * (N, nParty)

			# 全局矩阵计算
			TP = (Yp & Yt).sum().item()
			FP = (Yp & ~Yt).sum().item()
			FN = (~Yp & Yt).sum().item()

			precision = TP / (TP + FP + 1e-8)
			recall = TP / (TP + FN + 1e-8)
			f1 = 2 * precision * recall / (precision + recall + 1e-8)

			f1_matrix[i, j] = f1

	# 5. 打印并在终端输出结果矩阵
	print('\n[ F1-Score Matrix ]')
	print(np.round(f1_matrix, 4))
	print(f1_matrix.shape)

	# 6. 使用 Seaborn 绘制热力图并保存 (面向对象风格)

	# 显式创建 Figure (画布) 和 Axes (坐标系) 对象
	fig, ax = plt.subplots(figsize=(10, 8))

	# 坐标轴标签保留两位小数
	x_labels = [f'{v:.3f}' for v in m2_list]
	y_labels = [f'{v:.3f}' for v in m1_list]

	# 将 seaborn 的绘制目标显式指定为刚刚创建的 ax
	sns.heatmap(
		f1_matrix,
		annot=False,
		cmap='YlGnBu',
		xticklabels=x_labels,
		yticklabels=y_labels,
		linewidths=0.5,
		linecolor='white',
		ax=ax,  # <--- 核心：绑定到特定坐标系
	)

	# 使用 ax 的 set_ 方法统一设置标题和轴标签
	ax.set_title(f'Defense F1-Score Heatmap (Epoch {epoch_to_eval})', fontsize=14, pad=15)
	ax.set_xlabel('Mutual-Anomaly Tolerance ($m_2$)', fontsize=12)
	ax.set_ylabel('Self-Anomaly Tolerance ($m_1$)', fontsize=12)

	# 优化刻度标签的角度
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
	ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

	# 翻转 Y 轴
	ax.invert_yaxis()

	# 调整布局，防止标签被截断
	fig.tight_layout()

	# 使用 fig 对象直接保存
	save_img_path = file_path.replace('.pt', f'_heatmap{epoch_to_eval}.png')
	fig.savefig(save_img_path, dpi=300)
	print(f'\nHeatmap image saved to: {save_img_path}')

	# 显式关闭画布，释放内存 (在循环画图时极其重要！)
	plt.close(fig)

	with pathlib.Path('heatmap_tikz.txt').open('w', encoding='utf-8') as f:
		# 写入表头，TikZ 会根据这些名字读取列
		f.write('x y c\n')

		# 注意遍历顺序：外层 Y，内层 X。这决定了 pgfplots 扫描网格的方向
		for i, m1 in enumerate(m1_list):
			for j, m2 in enumerate(m2_list):
				# x 对应 m2，y 对应 m1，c 对应 F1 分数
				f.write(f'{m2:.3f} {m1:.3f} {f1_matrix[i, j]:.4f}\n')

	print('Heatmap data exported to heatmap_tikz.txt for TikZ!')


if __name__ == '__main__':
	FILE = 'data/logs/sgba_cifar10/lightning_logs/version_81_o/dEpochRecords.pt'
	for ep in [36]:
		analyze_all_epochs(FILE, ep, 5)
