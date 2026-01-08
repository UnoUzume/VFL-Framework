from typing import cast, override

from torch.optim import Adam, Optimizer, lr_scheduler as lr

from main.arch import BaseVFLArch, LightningArch
from main.callback import VFLCallback
from utils.common import F, L, Path, nn, tc
from utils.define import StepVars
from utils.misc import accuracy


# ==========================================
# 1. 配置与超参数
# ==========================================
class Config:
	SEED = 42
	BATCH_SIZE = 64
	LR = 1e-3
	EPOCHS = 50  # VAE 训练轮数
	LATENT_DIM = 64  # 论文 MNIST 设置为 64 [cite: 423]
	KNOWN_RATIO = 0.01  # 1% 已知标签 [cite: 423]
	TRIPLET_MARGIN = 0.25  # 论文 MNIST 设置为 0.25 [cite: 423]
	LAMBDA_TRIPLET = 1.0  # Triplet Loss 权重 (lambda_hat)
	LAMBDA_KL = 0.001  # KL 散度权重 (通常需要较小以平衡重构误差)
	BETA = 0.999  # 辅助分类器置信度阈值 [cite: 423]
	TARGET_LABEL = 1  # 我们想找出的目标 (例如数字 1)
	DEVICE = tc.device('cuda' if tc.cuda.is_available() else 'cpu')


# ==========================================
# 2. 数据准备 (模拟攻击者视角)
# ==========================================
def get_attacker_data() -> tuple[
	tuple[tc.Tensor, tc.Tensor], tuple[tc.Tensor, tc.Tensor, tc.Tensor]
]:
	"""模拟纵向联邦学习数据切分，并返回攻击者视角的拼接特征。"""
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
	dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

	# 原始特征 [60000, 784]
	X_full = dataset.data.float().view(-1, 784) / 255.0
	y_full = dataset.targets

	# 模拟切分：假设 4 个参与方，前 2 个是攻击者
	# 攻击者 0: features [0:196]
	# 攻击者 1: features [196:392]
	# 攻击者将两者拼接 -> features [0:392]
	feature_split_idx = 392
	X_attacker = X_full[:, :feature_split_idx]  # [cite: 173] X_m = union(x_m')

	# 划分 Known Set (1%) 和 Unknown Set (99%)
	num_total = len(y_full)
	num_known = int(num_total * Config.KNOWN_RATIO)

	# 随机选择已知索引
	perm = tc.randperm(num_total)
	known_idx = perm[:num_known]
	unknown_idx = perm[num_known:]

	# 构造 Known Dataset (有标签)
	X_known = X_attacker[known_idx]
	y_known = y_full[known_idx]

	# 构造 Unknown Dataset (无标签，用于推断)
	X_unknown = X_attacker[unknown_idx]
	# 注意：真实世界中攻击者不知道 y_unknown，这里保留是为了后续评估准确率
	y_unknown_truth = y_full[unknown_idx]

	return (X_known, y_known), (X_unknown, y_unknown_truth, unknown_idx)


# ==========================================
# 3. 模型定义：VAE + Auxiliary Classifier
# ==========================================
class AttackerVAE(nn.Module):
	def __init__(self, input_dim: int, latent_dim: int = 64) -> None:
		super().__init__()

		# Encoder [cite: 431] Fully connected for MNIST
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()
		)

		# Latent Space Projections
		self.fc_mu = nn.Linear(128, latent_dim)
		self.fc_logvar = nn.Linear(128, latent_dim)

		# Decoder
		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 256),
			nn.ReLU(),
			nn.Linear(256, input_dim),
			nn.Sigmoid(),  # 对应归一化后的数据范围
		)

	def reparameterize(self, mu: tc.Tensor, logvar: tc.Tensor) -> tc.Tensor:
		"""重参数化技巧 [cite: 223]"""
		if self.training:
			std = tc.exp(0.5 * logvar)
			eps = tc.randn_like(std)
			return mu + eps * std
		return mu  # 推理时直接使用 mu

	def forward(self, x: tc.Tensor) -> tuple[tc.Tensor, tc.Tensor, tc.Tensor]:
		h = self.encoder(x)
		mu = self.fc_mu(h)
		logvar = self.fc_logvar(h)
		z = self.reparameterize(mu, logvar)
		recon_x = self.decoder(z)
		return recon_x, mu, logvar


class AuxClassifier(nn.Module):
	"""辅助分类器 phi_mu [cite: 245]"""

	def __init__(self, latent_dim: int = 64, num_classes: int = 10) -> None:
		super().__init__()
		# 简单的线性层或浅层网络
		self.fc = nn.Linear(latent_dim, num_classes)

	def forward(self, mu: tc.Tensor) -> tc.Tensor:
		return self.fc(mu)


class InferArch(LightningArch):
	def __init__(self, dpRoot: Path | str, lPartyDims: list[int]) -> None:
		super().__init__(dpRoot)

		print(f'--- Setting up Attacker with {Config.KNOWN_RATIO * 100}% known labels ---')
		(X_known, y_known), (X_unknown, y_unknown_truth, original_indices) = get_attacker_data()

	@override
	def configOptims(self) -> tuple[list[Optimizer], list[lr.LRScheduler]]:
		optN1 = Adam(self.vflip.mae.parameters(), 1e-4)
		opt11 = Adam(self.vflip.mae.parameters(), 5e-4)
		return [optN1, opt11], []

	# ============
	# 训练阶段
	# ============

	@override
	def on_train_start(self) -> None:
		self.vflip.mean = self.trainer.datamodule.mean.to(self.device)  # pyright: ignore[reportAttributeAccessIssue]
		self.vflip.std = self.trainer.datamodule.std.to(self.device)  # pyright: ignore[reportAttributeAccessIssue]

	@override
	def training_step(
		self, batch: tuple[tc.Tensor, tc.Tensor], batch_idx: int, dataloader_idx: int = 0
	) -> None:
		batch_X, batch_y = batch

		# Forward Pass
		recon_X, mu, logvar = vae(batch_X)
		pred_y = aux_clf(mu)  # 辅助分类器输入为 mu [cite: 245]

		# --- 计算各个 Loss ---

		# 1. VAE Loss (Reconstruction + KL) [cite: 226]
		# MSE sum reduction as per formula typically
		mse_loss = F.mse_loss(recon_X, batch_X, reduction='sum') / batch_X.size(0)
		# KL Divergence
		kl_loss = -0.5 * tc.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_X.size(0)
		vae_loss = mse_loss + Config.LAMBDA_KL * kl_loss

		# 2. Triplet Loss (Metric Learning) [cite: 233, 241]
		# 使用 batch-hard 策略
		triplet_loss = batch_hard_triplet_loss(mu, batch_y, Config.TRIPLET_MARGIN)

		# 3. Auxiliary Classifier Loss (Cross Entropy) [cite: 246]
		cls_loss = F.cross_entropy(pred_y, batch_y)

		# 总损失
		# L_final = L_VAE + lambda * L_triplet + L_classifier
		# 注意：论文算法 1 行 1 和行 2 虽分开写，但通常联合优化或交替优化。这里采用联合优化。
		loss = vae_loss + Config.LAMBDA_TRIPLET * triplet_loss + cls_loss

	# ============
	# 验证阶段
	# ============

	def on_validation_epoch_start(self) -> None:
		self.vflip.startThres()
		for batch in self.trainer.train_dataloader:  # pyright: ignore[reportOptionalIterable]
			[tRawEmbeds] = batch
			self.vflip.updateThres(tRawEmbeds.to(self.device))
		self.vflip.calcThres()

	@override
	def validation_step(self, batch: BatchData, batch_idx: int, dataloader_idx: int = 0) -> None:
		batch_X, batch_y = batch

	# ============
	# 推理阶段
	# ============

	@override
	def predict_step(
		self, batch: BatchData, batch_idx: int, dataloader_idx: int = 0
	) -> tuple[tc.Tensor, tc.Tensor]:
		[tRawEmbeds, labels] = batch
		tIsAnomaly = self.vflip.detect(tRawEmbeds)
		purify = self.vflip.purify(tRawEmbeds, tIsAnomaly)
		return purify, labels


class VFLIPOffCb(VFLCallback):
	def __init__(self, lPartyDims: list[int]) -> None:
		super().__init__()
		self.lPartyDims = lPartyDims

	# ============
	# 训练阶段
	# ============

	@override
	def onFitStart(self, m: BaseVFLArch) -> None:
		self.lTrain: list[tc.Tensor] = []

	@override
	def onTrainTopIns(self, m: BaseVFLModule, v: StepVars) -> None:
		tRawEmbeds = tc.cat(v.lTopIns, 1).detach()
		self.lTrain.append(tRawEmbeds)

	# ============
	# 验证阶段
	# ============

	@override
	def onValEpochStart(self, m: BaseVFLModule) -> None:
		self.lValData: dict[str, list[tc.Tensor]] = {'Origin': [], 'Attack': [], 'Labels': []}

	@override
	def onValTopIns(self, m: BaseVFLModule, d: dict[str, StepVars]) -> None:
		v = d['Origin']
		tRawEmbeds = tc.cat(v.lTopIns, 1).detach()
		self.lValData['Origin'].append(tRawEmbeds)

		v = d['Attack']
		tRawEmbeds = tc.cat(v.lTopIns, 1).detach()
		self.lValData['Attack'].append(tRawEmbeds)
		self.lValData['Labels'].append(v.labels)
		self.lTopInsDims = [t.size(1) for t in v.lTopIns]

	@override
	def onValEpochEnd(self, m: BaseVFLModule) -> None:
		if m.current_epoch % 5 == 4:
			m.print('\nOffline Start\n')

			dpRoot = Path('data/logs/vflip_off')

			tTrain = tc.cat(self.lTrain, 0).cpu()
			lValData = [tc.cat(v, 0).cpu() for v in self.lValData.values()]
			tc.save(tTrain, f'data/logs/vflip_off/embeds/train_{m.current_epoch}.pt')
			tc.save(lValData, f'data/logs/vflip_off/embeds/val_{m.current_epoch}.pt')
			dm = DataModule(tTrain, lValData, 256)

			lm = VFLIPModule(dpRoot, self.lPartyDims)

			trainer = L.Trainer(
				deterministic=True,
				max_epochs=20,
				num_sanity_val_steps=0,
				default_root_dir=dpRoot,
			)
			trainer.fit(lm, datamodule=dm)

			predictions = cast('PRED_TYPE', trainer.predict(datamodule=dm, ckpt_path='best'))

			with tc.no_grad():
				for purify, labels in predictions:
					purify_ = purify.to(m.device)
					labels_ = labels.to(m.device)
					lTopIns = list(tc.split(purify_, self.lTopInsDims, 1))

					zTopOut = m.zTopNet(lTopIns)
					loss = m.criterion(zTopOut, labels_)

					[acc1, acc3] = accuracy(zTopOut, labels_, (1, 3))
					m.logDict({'loss/ValOff': loss, 'acc/ValOff/Top1': acc1, 'acc/ValOff/Top3': acc3})

					tTgtLabels = tc.full_like(labels_, m.ns.iTgtLabel)
					[accT1, accT3] = accuracy(zTopOut, tTgtLabels, (1, 3))
					lossT = m.criterion(zTopOut, tTgtLabels)

					sName = 'ValOff/Tgt'
					m.logDict(
						{f'loss/{sName}': lossT, f'acc/{sName}/Top1': accT1, f'acc/{sName}/Top3': accT3}
					)

			m.print('\nOffline End\n')
