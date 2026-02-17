"""杂项工具函数模块。

该模块提供了一些通用的工具函数和类型定义，用于数据处理、模型检查和评估等功能。
"""

from collections.abc import Sequence
from typing import TypedDict

from beartype import beartype as typechecker
from jaxtyping import Integer, jaxtyped

from .common import nn, np, tc


def ensureList[T](ins: list[T] | T | None) -> list[T]:
	"""确保输入是列表类型。

	如果输入是 `None`，则返回空列表；如果输入是列表，则直接返回；否则将输入包装为单元素列表。

	Args:
		ins: 输入值，可以是单个元素、列表或 `None`

	Returns:
		转换后的列表
	"""
	if ins is None:
		return []
	if isinstance(ins, list):
		return ins
	return [ins]


def notNone[T](ins: T | None) -> T:
	"""确保输入不为 `None`。

	如果输入为 `None`，则抛出 `AssertionError`；否则返回输入值本身。

	Args:
		ins: 输入值，可以是任意类型或 `None`

	Returns:
		非 `None` 的输入值

	Raises:
		AssertionError: 当输入为 `None` 时抛出
	"""
	assert ins is not None
	return ins


def checkTensorGrad(tensor: tc.Tensor) -> float | None:
	"""检查张量的梯度范数。

	计算并返回张量梯度的 L2 范数，如果梯度为 `None`，则返回 `None`。

	Args:
		tensor: 要检查的 PyTorch 张量

	Returns:
		张量梯度的 L2 范数，如果梯度为 `None` 则返回 `None`
	"""
	if tensor.grad is None:
		return None
	return tc.norm(tensor.grad, 2).item()


def checkTensor(tensor: tc.Tensor) -> float | None:
	"""检查张量的梯度范数。

	计算并返回张量梯度的 L2 范数，如果梯度为 `None`，则返回 `None`。

	Args:
		tensor: 要检查的 PyTorch 张量

	Returns:
		张量梯度的 L2 范数，如果梯度为 `None` 则返回 `None`
	"""
	return tc.norm(tensor, 2).item()


def checkModelGrad(model: nn.Module) -> float | None:
	"""检查模型所有参数的梯度范数总和。

	计算并返回模型所有参数梯度的 L2 范数总和，如果任何参数的梯度为 `None`，则返回 `None`。

	Args:
		model: 要检查的 PyTorch 模型

	Returns:
		所有参数梯度的 L2 范数总和，如果任何参数梯度为 `None` 则返回 `None`
	"""
	total = 0.0

	for param in model.parameters():
		norm = checkTensorGrad(param)
		if norm is None:
			return None
		total += norm

	return total


def checkModel(model: nn.Module) -> float | None:
	total = 0.0

	for param in model.parameters():
		norm = checkTensor(param)
		if norm is None:
			return None
		total += norm

	return total


class LoaderParams(TypedDict):
	"""数据加载器参数类型定义。

	用于指定 PyTorch 数据加载器的工作线程参数。
	"""

	num_workers: int
	"""用于数据加载的子进程数量"""
	persistent_workers: bool
	"""是否在数据集迭代结束后保持工作进程活跃"""


def accuracy(pred: tc.Tensor, target: tc.Tensor, topk: Sequence[int] = (1,)) -> list[float]:
	"""计算分类准确率。

	计算模型预测结果在指定 top-k 值下的准确率。

	Args:
		pred: 模型预测的 logits 或概率值，形状为 `(nBatchSize, nClass)`
		target: 真实标签，形状为 `(nBatchSize,)`
		topk: 要计算的 top-k 值列表，默认为 `(1,)`

	Returns:
		对应于每个 top-k 值的准确率列表
	"""
	maxk = max(topk)
	nBatchSize = target.size(0)

	_, pred = pred.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.reshape(1, -1).expand_as(pred))

	res = []
	for k in topk:
		nCorrect = correct[:k].sum().item()
		res.append(nCorrect / nBatchSize)

	return res


def segment(t: int, ins: tuple[int, int], out: tuple[float, float]) -> float:
	"""在给定的自变量范围内进行线性插值。

	根据输入的自变量 `t` 在指定的自变量区间 `ins` 内进行线性插值计算，
	并返回对应输出区间 `out` 中的相应值。如果自变量超出范围，
	则返回区间的边界值。

	Args:
		t: 当前自变量值
		ins: 自变量区间，格式为 `(开始自变量，结束自变量)`
		out: 输出区间，格式为 `(开始值，结束值)`

	Returns:
		根据线性插值计算得出的浮点数值
	"""
	if t < ins[0]:
		return out[0]
	if t > ins[1]:
		return out[1]
	return out[0] + (out[1] - out[0]) * (t - ins[0]) / (ins[1] - ins[0])


@jaxtyped(typechecker=typechecker)
def selectPerClass(
	lLabels: list[int] | Integer[np.ndarray | tc.Tensor, ' n'],
	nPerClass: int,
	rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
	if isinstance(lLabels, tc.Tensor):
		aLabels = lLabels.numpy()
	else:
		aLabels = np.asarray(lLabels)

	assert aLabels.ndim == 1, '输入标签数组维度必须为 1！'

	lSelectIdxs: list[int] = []
	lOtherIdxs: list[int] = []

	aUniqueClasses = np.unique(aLabels)
	for i in aUniqueClasses:
		idxs = np.flatnonzero(aLabels == i)
		assert idxs.size >= nPerClass, f'类别 {i} 的样本数不足 {nPerClass}！'
		rng.shuffle(idxs)
		lSelectIdxs.extend(idxs[:nPerClass].tolist())
		lOtherIdxs.extend(idxs[nPerClass:].tolist())

	rng.shuffle(lSelectIdxs)
	rng.shuffle(lOtherIdxs)
	return lSelectIdxs, lOtherIdxs
