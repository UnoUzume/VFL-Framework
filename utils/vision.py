"""视觉处理工具模块。

该模块提供了图像变换相关的工具函数，用于创建和配置图像预处理管道。
"""

from collections.abc import Callable

import torchvision.transforms.v2 as tf

from . import define as de
from .common import tc


def createTrans(
	lTrans: list[tf.Transform] | None = None, needNormalize: bool = True
) -> Callable[[de.TUImage], de.TFImage]:
	"""创建图像变换管道。

	构建一个包含指定变换操作的图像预处理管道，可选择是否包含归一化操作。

	Args:
		lTrans: 自定义变换操作列表，默认值为 `None`
		needNormalize: 是否添加图像归一化操作，默认值为 `True`

	Returns:
		组合后的图像变换函数，接受原始图像并返回变换后的图像
	"""
	if lTrans is None:
		lTrans = []

	if needNormalize:
		lNormalize = [tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
	else:
		lNormalize = []

	return tf.Compose(
		[
			tf.ToImage(),  # 将 Tensor、NDArray 或 PIL 图像转换为 Image，此操作不缩放数值
			*lTrans,
			tf.ToDtype(tc.float32, True),  # Normalize 需要 float 类型
			*lNormalize,
		]
	)


__all__ = ['createTrans', 'tf']
