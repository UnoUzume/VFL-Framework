from typing import Any, override

from torch.utils.data import DataLoader, Subset

from utils.config import rng
from utils.misc import selectPerClass
from utils.module import DataConfig, DataHandler, SplitDataModule


class CustomDataModule(SplitDataModule):
	def __init__(self, nParty: int, config: DataConfig, handler: DataHandler | None = None) -> None:
		super().__init__(nParty, config, handler)

	@override
	def setup(self, stage: str) -> None:
		super().setup(stage)

		[idxsE, idxsU] = selectPerClass(self.dsTrain.labels, 10, rng())
		self.dsTrainE = Subset(self.dsTrain, idxsE)
		# self.dsTrainU = Subset(self.dsTrain, idxsU)

	@override
	def getTrainLoader(self) -> list[DataLoader[Any]]:
		return [
			DataLoader(
				self.dsTrain,
				self.cfg.nBatchSize,
				True,
				drop_last=True,
				collate_fn=self.cfg.fnCollate,
				**self.params,
			),
			DataLoader(
				self.dsTrainE,
				min(self.cfg.nBatchSize, len(self.dsTrainE)),
				True,
				drop_last=True,
				collate_fn=self.cfg.fnCollate,
				**self.params,
			),
		]

	@override
	def getValLoader(self) -> list[DataLoader[Any]]:
		return [
			DataLoader(
				self.dsVal, self.cfg.nBatchSize, False, collate_fn=self.cfg.fnCollate, **self.params
			),
			DataLoader(
				self.dsTrain, self.cfg.nBatchSize, False, collate_fn=self.cfg.fnCollate, **self.params
			),
		]
