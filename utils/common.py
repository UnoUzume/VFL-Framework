"""公共工具模块。

该模块从各种常用库中导入常用的类、函数和模块，并将它们重新导出，
以便在项目的其他部分中更方便地使用。这样可以减少重复的导入语句，
提高代码的可读性和一致性。
"""

from copy import copy
from pathlib import Path

import lightning as L
import numpy as np
import numpy.typing as npt
import torch as tc
import torch.nn as nn

__all__ = [
	'L',
	'Path',
	'copy',
	'nn',
	'np',
	'npt',
	'tc',
]
