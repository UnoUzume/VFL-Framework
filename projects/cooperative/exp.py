# mypy: ignore-errors

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_metric_learning import distances, losses, miners, samplers
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


# ==========================================
# 1. 配置参数
# ==========================================
class Config:
	SEED = 42
	BATCH_SIZE = 64
	LR = 1e-3
	EPOCHS = 50
	LATENT_DIM = 64  # [cite: 423] MNIST Latent Dim = 64
	KNOWN_RATIO = 0.01  # [cite: 423] 1% labels known
	TRIPLET_MARGIN = 0.25  # [cite: 423] Margin alpha
	LAMBDA_TRIPLET = 1.0  # 论文公式 (4) 中的 lambda_hat
	LAMBDA_KL = 0.001  # KL 散度权重
	BETA = 0.999  # [cite: 423] Confidence threshold
	TARGET_LABEL = 1  # 想要推断的目标类别
	SAMPLES_PER_CLASS = 4  # 配合 Triplet Loss，每个 Batch 每类至少采样的个数


pl.seed_everything(Config.SEED)


# ==========================================
# 2. 数据模块 (LightningDataModule)
# ==========================================
class AttackerDataModule(pl.LightningDataModule):
	def __init__(self, data_dir='./data', batch_size=64):
		super().__init__()
		self.data_dir = data_dir
		self.batch_size = batch_size

	def setup(self, stage=None):
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
		mnist_full = datasets.MNIST(self.data_dir, train=True, download=True, transform=transform)

		# 1. 模拟“超级攻击者”：获取部分特征
		# 假设 4 方参与，2 方合谋，拥有前 50% 的特征 (392 维) [cite: 173]
		X_full = mnist_full.data.float().view(-1, 784) / 255.0
		X_attacker = X_full[:, :392]
		y_full = mnist_full.targets

		# 2. 划分 Known (1%) 和 Unknown (99%)
		num_total = len(y_full)
		num_known = int(num_total * Config.KNOWN_RATIO)

		perm = torch.randperm(num_total)
		self.known_idx = perm[:num_known]
		self.unknown_idx = perm[num_known:]

		self.X_known = X_attacker[self.known_idx]
		self.y_known = y_full[self.known_idx]

		self.X_unknown = X_attacker[self.unknown_idx]
		self.y_unknown_truth = y_full[self.unknown_idx]  # 仅用于验证，训练不用

		# 3. 构建 Known Dataset
		self.known_dataset = TensorDataset(self.known_idx, self.X_known, self.y_known)

		# 4. 构建 Unknown Dataset (用于推断阶段)
		# 传入 unknown_idx 是为了推断后能映射回原始数据集
		self.unknown_dataset = TensorDataset(self.unknown_idx, self.X_unknown, self.y_unknown_truth)

	def train_dataloader(self):
		# 关键：使用 MPerClassSampler 确保 Triplet Mining 有效
		sampler = samplers.MPerClassSampler(
			labels=self.y_known,
			m=Config.SAMPLES_PER_CLASS,
			batch_size=self.batch_size,
			length_before_new_iter=len(self.known_dataset),
		)
		# 注意：使用 sampler 时 shuffle 必须为 False
		return DataLoader(self.known_dataset, batch_size=self.batch_size, sampler=sampler)

	def predict_dataloader(self):
		return DataLoader(self.unknown_dataset, batch_size=self.batch_size, shuffle=False)


# ==========================================
# 3. 攻击模型 (LightningModule - Manual Mode)
# ==========================================
class AttackerModel(pl.LightningModule):
	def __init__(self, input_dim=392):
		super().__init__()

		# --- A. 开启手动优化 ---
		self.automatic_optimization = False

		# --- B. 模型架构 ---
		# 1. VAE Encoder [cite: 223]
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()
		)
		self.fc_mu = nn.Linear(128, Config.LATENT_DIM)
		self.fc_logvar = nn.Linear(128, Config.LATENT_DIM)

		# 2. VAE Decoder [cite: 221]
		self.decoder = nn.Sequential(
			nn.Linear(Config.LATENT_DIM, 128),
			nn.ReLU(),
			nn.Linear(128, 256),
			nn.ReLU(),
			nn.Linear(256, input_dim),
			nn.Sigmoid(),
		)

		# 3. 辅助分类器 [cite: 245]
		self.aux_clf = nn.Linear(Config.LATENT_DIM, 10)

		# --- C. 损失函数组件 (使用 PML 库) ---
		# 论文使用欧氏距离和 Batch-Hard Mining [cite: 238]
		self.distance = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
		self.miner = miners.TripletMarginMiner(
			margin=Config.TRIPLET_MARGIN, distance=self.distance, type_of_triplets='hard'
		)
		self.triplet_loss_fn = losses.TripletMarginLoss(
			margin=Config.TRIPLET_MARGIN, distance=self.distance
		)
		self.ce_loss_fn = nn.CrossEntropyLoss()

	def forward(self, x):
		h = self.encoder(x)
		mu = self.fc_mu(h)
		logvar = self.fc_logvar(h)

		# Reparameterization Trick [cite: 223]
		if self.training:
			std = torch.exp(0.5 * logvar)
			eps = torch.randn_like(std)
			z = mu + eps * std
		else:
			z = mu

		recon_x = self.decoder(z)
		aux_logits = self.aux_clf(mu)  # 辅助分类器仅使用 mu [cite: 245]
		return recon_x, mu, logvar, aux_logits

	def training_step(self, batch, batch_idx):
		# 1. 获取优化器
		opt = self.optimizers()

		# 2. 解包数据 (idx, features, labels)
		_, x, y = batch

		# 3. 前向传播
		recon_x, mu, logvar, aux_logits = self(x)

		# 4. 计算各个 Loss

		# (a) VAE Reconstruction Loss (Sum over features, Mean over batch)
		# 手动处理：先对特征求和，再除以 Batch Size，避免 loss 过小
		recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

		# (b) KL Divergence [cite: 227]
		# 公式：-0.5 * sum(1 + logvar - mu^2 - exp(logvar))
		kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

		# (c) Triplet Loss (Metric Learning) [cite: 233, 241]
		# 使用 Miner 挖掘难例
		hard_pairs = self.miner(mu, y)
		triplet_loss = self.triplet_loss_fn(mu, y, hard_pairs)

		# (d) Auxiliary Classifier Loss
		cls_loss = self.ce_loss_fn(aux_logits, y)

		# (e) Total Loss [cite: 241]
		total_loss = (
			recon_loss + Config.LAMBDA_KL * kl_loss + Config.LAMBDA_TRIPLET * triplet_loss + cls_loss
		)

		# 5. 手动反向传播与更新
		opt.zero_grad()
		self.manual_backward(total_loss)
		opt.step()

		# 6. Logging
		self.log_dict(
			{
				'train_loss': total_loss,
				'recon_loss': recon_loss,
				'triplet_loss': triplet_loss,
				'cls_loss': cls_loss,
				'active_triplets': self.miner.num_triplets,
			},
			prog_bar=True,
		)

		return total_loss

	def predict_step(self, batch, batch_idx):
		"""推断阶段：在无标签数据上运行"""
		global_indices, x, true_y = batch  # global_indices 用于追踪原始样本位置

		# 仅使用 mu 进行推断
		_, mu, _, aux_logits = self(x)
		probs = F.softmax(aux_logits, dim=1)
		max_probs, preds = torch.max(probs, dim=1)

		# 筛选逻辑：预测为目标类 且 置信度 > Beta
		mask = (preds == Config.TARGET_LABEL) & (max_probs >= Config.BETA)

		# 返回筛选出的样本信息
		return {
			'indices': global_indices[mask],
			'true_labels': true_y[mask],  # 仅用于验证准确率
			'preds': preds[mask],
			'confs': max_probs[mask],
		}

	def configure_optimizers(self):
		return optim.Adam(self.parameters(), lr=Config.LR)


# ==========================================
# 4. 执行流程
# ==========================================
if __name__ == '__main__':
	# 1. 准备数据
	dm = AttackerDataModule()

	# 2. 初始化模型
	model = AttackerModel()

	# 3. 训练 (Phase 1 Training)
	trainer = pl.Trainer(
		max_epochs=Config.EPOCHS,
		accelerator='auto',
		devices=1,
		enable_checkpointing=False,
		logger=False,  # 简化输出
	)
	print('--- Starting Attack Training (Manual Optimization) ---')
	trainer.fit(model, datamodule=dm)

	# 4. 推断 (Phase 1 Inference)
	print(f'\n--- Starting Label Inference on Unknown Data (Target: {Config.TARGET_LABEL}) ---')
	predictions = trainer.predict(model, datamodule=dm)

	# 5. 结果汇总与评估
	total_inferred = 0
	correct_inferred = 0

	inferred_global_indices = []

	for batch_res in predictions:
		if batch_res is None or len(batch_res['indices']) == 0:
			continue

		indices = batch_res['indices'].cpu().numpy()
		true_labels = batch_res['true_labels'].cpu().numpy()

		inferred_global_indices.extend(indices)
		total_inferred += len(indices)
		# 验证推断是否正确 (攻击者在实际场景中无法得知，这里用于评估)
		correct_inferred += (true_labels == Config.TARGET_LABEL).sum()

	print('-' * 50)
	print(f'Total Unknown Samples Scanned: {len(dm.unknown_idx)}')
	print(f'Samples Inferred as Target Class {Config.TARGET_LABEL}: {total_inferred}')

	if total_inferred > 0:
		precision = 100.0 * correct_inferred / total_inferred
		print(f'Inference Precision: {precision:.2f}%')

		# 计算 Recall
		total_targets_truth = (dm.y_unknown_truth == Config.TARGET_LABEL).sum().item()
		recall = 100.0 * correct_inferred / total_targets_truth
		print(f'Inference Recall: {recall:.2f}%')

		print("\n[Next Step] Use 'inferred_global_indices' to inject triggers in VFL training.")
	else:
		print('No samples met the confidence threshold. Try lowering Config.BETA or training longer.')
