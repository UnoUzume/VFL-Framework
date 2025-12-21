"""全连接神经网络模块，提供可配置的 `FCN` 类

该模块提供了一个可以灵活配置的全连接神经网络类 `FCN`，支持自定义网络结构、
激活函数、Dropout 和 Batch Normalization。
"""

from utils.common import nn, tc


class FCN(nn.Module):
	"""全连接神经网络类

	该类可以根据指定的维度列表动态构建全连接网络，支持配置激活函数、
	Dropout 和 Batch Normalization。
	"""

	def __init__(
		self,
		lDims: list[int],
		hasBN: bool = True,
		sActType: str = 'relu',
		rDropout: float = 0.0,
	) -> None:
		"""初始化实例。

		Args:
			lDims: 网络各层的维度列表，包括输入层、隐藏层和输出层
			hasBN: 是否使用 Batch Normalization，默认为 `True`
			sActType: 激活函数类型，支持 `'relu'`、`'sigmoid'`、`'tanh'`、`'none'`，默认为 `'relu'`
			rDropout: Dropout 比率，默认为 `0.0`（不使用 Dropout）

		Raises:
			ValueError: 当维度列表长度小于 2 时抛出
		"""
		super().__init__()

		if len(lDims) < 2:
			msg = '维度列表至少需要包含输入层和输出层两个维度'
			raise ValueError(msg)

		self.lDims = lDims
		"""网络各层的维度列表"""
		self.hasBN = hasBN
		"""是否使用 Batch Normalization"""
		self.sActType = sActType
		"""激活函数类型"""
		self.rDropout = rDropout
		"""Dropout 比率"""

		self.layers = self._create_layers()
		"""网络层列表"""

	def _create_layers(self) -> nn.ModuleList:
		"""根据维度列表创建全连接网络层。

		该方法根据 `self.lDims` 列表创建全连接网络的所有层，包括隐藏层和输出层。<br>
		隐藏层包含线性变换、可选的 Batch Normalization、激活函数和可选的 Dropout。<br>
		输出层仅包含线性变换。

		Returns:
			包含所有网络层的 `nn.ModuleList` 对象

		Raises:
			ValueError: 当指定了不支持的激活函数类型时抛出
		"""
		layers = nn.ModuleList()

		# 隐藏层处理
		for i in range(len(self.lDims) - 2):
			block = nn.Sequential()

			# 添加线性层
			block.add_module('fc', nn.Linear(self.lDims[i], self.lDims[i + 1]))

			# 添加 Batch Normalization (可选)
			if self.hasBN:
				block.add_module('bn', nn.BatchNorm1d(self.lDims[i + 1]))

			# 添加激活函数
			if self.sActType == 'none':
				pass
			elif self.sActType == 'relu':
				block.add_module('act', nn.ReLU())
			elif self.sActType == 'sigmoid':
				block.add_module('act', nn.Sigmoid())
			elif self.sActType == 'tanh':
				block.add_module('act', nn.Tanh())
			else:
				msg = f'不支持的激活函数：{self.sActType}'
				raise ValueError(msg)

			# 添加 Dropout (可选)
			if self.rDropout > 0:
				block.add_module('do', nn.Dropout(self.rDropout))

			layers.append(block)

		# 输出层仅为线性层
		layers.append(nn.Linear(self.lDims[-2], self.lDims[-1]))

		return layers

	def forward(self, ins: tc.Tensor | list[tc.Tensor]) -> tc.Tensor:
		"""执行网络的前向传播。

		Args:
			ins: 输入数据，可以是单个 `tc.Tensor` 对象或 `tc.Tensor` 对象列表

				如果是列表，则会在第一个维度上拼接。

		Returns:
			网络的输出张量
		"""
		if isinstance(ins, list):
			out = tc.cat(ins, 1)
		else:
			out = ins

		for layer in self.layers:
			out = layer(out)

		return out


if __name__ == '__main__':
	# 创建一个网络：输入层 784 -> 隐藏层 512 -> 隐藏层 256 -> 输出层 10
	model = FCN(lDims=[784, 512, 256, 10], sActType='relu', rDropout=0.3, hasBN=True)

	# 打印网络结构
	print(model)

	# 测试输入
	test_input = tc.randn(1, 784)  # 批量大小为 1
	output = model(test_input)
	print(f'输出形状：{output.shape}')
	print(f'输出值：{output}')
