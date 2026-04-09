"""CIFAR-10 模块的自动化单元测试。

执行命令：pytest tests/modules/test_cifar10.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
import torch as tc

# 假设你的代码存放路径为 modules.cifar10，请根据实际情况调整导入路径
from modules.cifar10 import CIFAR10Dataset, Handler, getAugmentTrans, splitImage
from utils import define as de
from utils.common import np

# ==========================================
# 1. 纯函数测试：图像张量切分 (splitImage)
# ==========================================


@pytest.fixture
def dummy_image_batch() -> tc.Tensor:
	"""Fixture: 生成一个伪造的图像 Batch (BatchSize=4)。

	CIFAR-10 的标准输入为 (N, C, H, W) -> (4, 3, 32, 32)
	且在 Dataset 挂载阶段我们保留了 uint8 类型。
	"""
	return tc.randint(0, 256, (4, 3, 32, 32), dtype=tc.uint8)


def test_split_image_nparty_1(dummy_image_batch: tc.Tensor) -> None:
	"""测试 1 方情况：原样返回"""
	parts = splitImage(dummy_image_batch, nParty=1)
	assert len(parts) == 1
	assert tc.equal(parts[0], dummy_image_batch)


def test_split_image_nparty_2(dummy_image_batch: tc.Tensor) -> None:
	"""测试 2 方情况：宽度中点切分"""
	parts = splitImage(dummy_image_batch, nParty=2)
	assert len(parts) == 2
	assert parts[0].shape == (4, 3, 32, 16)  # 左半边
	assert parts[1].shape == (4, 3, 32, 16)  # 右半边


def test_split_image_nparty_4(dummy_image_batch: tc.Tensor) -> None:
	"""测试 4 方情况：田字格切分"""
	parts = splitImage(dummy_image_batch, nParty=4)
	assert len(parts) == 4
	for i in range(4):
		assert parts[i].shape == (4, 3, 16, 16)


def test_split_image_nparty_8(dummy_image_batch: tc.Tensor) -> None:
	"""测试 8 方情况：插值放大后切分"""
	parts = splitImage(dummy_image_batch, nParty=8)
	assert len(parts) == 8

	# 验证插值后的切片尺寸是否正确
	for i in range(8):
		assert parts[i].shape == (4, 3, 32, 16)

	# 核心验证点：确保 float 插值后，张量被正确还原回了原始的 uint8 类型
	assert parts[0].dtype == tc.uint8


def test_split_image_invalid_party(dummy_image_batch: tc.Tensor) -> None:
	"""测试异常处理：传入不支持的 nParty"""
	with pytest.raises(ValueError, match='不支持的 nParty'):
		splitImage(dummy_image_batch, nParty=5)


# ==========================================
# 2. 纯函数测试：增强管线生成 (getAugmentTrans)
# ==========================================


@pytest.mark.parametrize('nParty', [1, 2, 3, 4, 8])
def test_get_augment_trans_valid(nParty: int) -> None:
	"""测试为合法的 nParty 生成对应尺寸的 Transform 列表"""
	trans_list = getAugmentTrans(nParty)
	# 根据代码逻辑，应该返回 3 个变换步骤：Crop, Flip, ColorJitter
	assert len(trans_list) == 3


def test_get_augment_trans_invalid() -> None:
	"""测试字典键值越界异常"""
	with pytest.raises(KeyError):
		getAugmentTrans(nParty=5)


# ==========================================
# 3. 隔离依赖测试：CIFAR10Dataset 内存挂载
# ==========================================


@patch('modules.cifar10.CIFAR10')
def test_cifar10_dataset_initialization(mock_cifar10_cls: MagicMock) -> None:
	"""测试数据集初始化。

	使用 patch 拦截 torchvision.datasets.CIFAR10，防止真正发起网络下载或磁盘读取。
	而是向 Dataset 注入一个构造好的假 NumPy 数组。
	"""
	# 1. 组装假的 torchvision CIFAR10 实例
	mock_instance = MagicMock()
	# 模拟原生 CIFAR10 的数据格式 (N, H, W, C)
	mock_instance.data = np.random.randint(0, 256, (10, 32, 32, 3), dtype=np.uint8)
	mock_instance.targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	mock_cifar10_cls.return_value = mock_instance

	# 2. 实例化我们自己的 Dataset
	dataset = CIFAR10Dataset(dpRoot='dummy/path', isTrain=True)

	# 3. 验证内存布局转置是否正确执行了 (N, H, W, C) -> (N, C, H, W)
	assert dataset.data.shape == (10, 3, 32, 32)
	assert isinstance(dataset.data, tc.Tensor)
	assert dataset.data.dtype == tc.uint8

	# 4. 验证 __getitem__ 是否能正确打包装箱
	sample = dataset[0]
	assert isinstance(sample, de.TUImageSample)
	assert sample.label == 0
	assert sample.idx == 0


# ==========================================
# 4. 协议契约测试：Handler
# ==========================================


def test_handler_protocol_compliance() -> None:
	"""测试 Handler 是否能正确吐出配置和函数指针"""
	handler = Handler()

	# 验证是否返回了空列表（因为 CIFAR 默认不加 NormalTrans）
	assert handler.getNormalTrans() == []

	# 验证 CollateFn 工厂是否成功生成
	collate_fn = handler.getCollateFn()
	assert callable(collate_fn)

	# 验证 SplitFn 映射
	split_fn = handler.getSplitFn()
	assert split_fn is splitImage
