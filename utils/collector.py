"""数据收集器模块

本模块实现了 `TensorCollector` 类，用于高效收集和处理深度学习训练过程中的批次数据。
"""

from .common import tc


class TensorCollector:
	"""`Tensor` 数据收集器，用于收集、存储和处理多个批次的 `Tensor` 数据

	该类可以将多个批次的数据合并为一个完整的 `Tensor`，适用于深度学习训练和评估过程中的数据收集
	"""

	def __init__(self, device: tc.device | str | None = None) -> None:
		"""初始化实例。

		Args:
			device: 目标设备，可以是 `tc.device` 对象、设备字符串（如 `'cpu'` 或 `'cuda'`）或 `None`

				如果指定了设备，`read()` 方法返回的 `Tensor` 会自动移动到该设备上。
		"""
		self.device = device
		"""目标设备，用于指定返回 `Tensor` 的设备位置"""
		self.buffer: dict[str, list[tc.Tensor]] = {}
		"""用于存储各个数据名称对应的 `Tensor` 列表的缓冲区"""

	def addBatch(self, batch: dict[str, tc.Tensor]) -> None:
		"""添加一个批次的数据到缓冲区。

		Args:
			batch: 包含多个 `Tensor` 的字典，键为数据名称，值为对应的 `Tensor`
		"""
		for k, v in batch.items():
			self.buffer.setdefault(k, []).append(v)

	def read(self) -> dict[str, tc.Tensor]:
		"""读取并合并所有收集的批次数据。

		Returns:
			一个字典，包含所有收集的数据名称及其对应的合并后 `Tensor`

			如果初始化时指定了设备，返回的 `Tensor` 会自动移动到该设备上。
		"""
		ret = {k: tc.cat(v) for k, v in self.buffer.items()}
		if self.device is not None:
			ret = {k: v.to(self.device) for k, v in ret.items()}
		return ret


__all__ = ['TensorCollector']
