"""ResNet 模型实现模块

本模块基于 torchvision 的 ResNet 实现，提供了适配特征提取任务的封装。
"""

from typing import override

from torchvision import models

from utils.common import nn, tc


class ResNet18(nn.Module):
	"""ResNet-18 模型，用于特征提取任务

	本类封装了 torchvision 的 ResNet-18 实现，适配于特征提取而非分类任务。
	"""

	def __init__(self, nOutDim: int) -> None:
		"""初始化实例。

		Args:
			nOutDim: 输出维度，即特征向量的长度
		"""
		super().__init__()
		self.net = models.resnet18(num_classes=nOutDim)

	@override
	def forward(self, ins: tc.Tensor) -> tc.Tensor:
		"""执行前向传播计算。

		Args:
			ins: 输入张量，形状为 `(nBatchSize, channels, height, width)`

		Returns:
			特征向量张量，形状为 `(nBatchSize, nOutDim)`
		"""
		return self.net(ins)
