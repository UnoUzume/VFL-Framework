"""垂直联邦学习（VFL）模型定义模块。

本模块定义了垂直联邦学习中使用的底层模型（BottomModel）和顶层模型（TopModel），
底层模型负责从不同参与方提取特征，顶层模型负责融合特征并进行最终分类。
"""

from typing import override

from torchvision import models

from utils.common import nn, tc


class TopModel(nn.Module):
	"""垂直联邦学习的顶层模型。

	负责融合来自多个底层模型的特征，并输出最终的分类结果。
	"""

	def __init__(self) -> None:
		"""初始化顶层模型实例。

		构建顶层模型的神经网络结构，包括全连接层和分类器。
		"""
		super().__init__()
		self.linear1 = nn.Linear(256, 256)
		"""第一层全连接层，输入维度 256，输出维度 256"""
		self.linear2 = nn.Linear(256, 128)
		"""第二层全连接层，输入维度 256，输出维度 128"""
		self.classifier = nn.Linear(128, 10)
		"""分类器层，输入维度 128，输出维度 10（对应 10 个类别）"""

	@override
	def forward(self, input_list: list[tc.Tensor]) -> tc.Tensor:
		"""顶层模型前向传播过程。

		将多个底层模型的输出特征进行融合，并通过全连接层和分类器生成最终分类结果。

		Args:
			input_list: 包含多个底层模型输出特征的张量列表

		Returns:
			分类结果张量，形状为 `(nBatchSize, nClass)`
		"""
		tensor_t = tc.cat(input_list, dim=1)

		x = tensor_t
		x = self.linear1(x)
		x = self.linear2(x)
		x = self.classifier(x)
		return x


class BottomModel(nn.Module):
	"""垂直联邦学习的底层模型。

	负责从本地数据中提取特征，是垂直联邦学习中各参与方的本地模型。
	"""

	def __init__(self, nOutDim: int) -> None:
		"""初始化底层模型实例。

		使用 ResNet-18 作为基础模型，并设置输出维度。

		Args:
			nOutDim: 底层模型输出特征的维度
		"""
		super().__init__()
		self.net = models.resnet18(num_classes=nOutDim)
		"""基于 ResNet-18 的特征提取网络"""

	@override
	def forward(self, ins: tc.Tensor) -> tc.Tensor:
		"""底层模型前向传播过程。

		通过 ResNet-18 网络提取输入图像的特征。

		Args:
			ins: 输入图像张量，形状为 `(nBatchSize, channels, height, width)`

		Returns:
			特征张量，形状为 `(nBatchSize, nOutDim)`
		"""
		return self.net(ins)
