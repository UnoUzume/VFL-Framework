"""VFL 框架配置模块，定义了模型、运行和应用的配置结构

本模块提供了纵向联邦学习所需的各类配置对象，包括模型结构、训练参数和应用设置。
"""

import shutil
import sys
from collections.abc import Callable
from dataclasses import InitVar, dataclass, field

from torch.optim import Adam, Optimizer, lr_scheduler as lrs

import __main__
from main.arch import BaseVFLArch
from main.callback import OPT_TYPE
from models.fcn import FCN
from models.resnet import ResNet18
from utils.common import Path, nn
from utils.module import DataConfig


def backup_entry_script(target_dir: str | Path) -> Path | None:
	"""将当前运行的入口脚本文件 (.py) 复制到指定目录。

	Args:
		target_dir: 目标目录路径（字符串或 Path 对象）

	Returns:
		Path: 复制后的文件完整路径，如果失败则返回 None
	"""
	# 1. 获取入口文件路径 (逻辑同 _getAppName)
	if not sys.argv or not sys.argv[0]:
		return None

	src_file = Path(sys.argv[0]).resolve()

	# 2. 安全校验：确保是 Python 文件且物理存在
	if src_file.suffix != '.py' or not src_file.is_file():
		return None

	# 3. 准备目标目录
	dst_path = Path(target_dir).resolve()

	try:
		# 确保目标文件夹存在，不存在则递归创建 (mkdir -p)
		dst_path.mkdir(parents=True, exist_ok=True)

		# 构造目标文件的完整路径
		dst_file = dst_path / src_file.name

		# 4. 执行复制
		# 使用 copy2 会尝试保留源文件的元数据（如修改时间、权限等）
		shutil.copy2(src_file, dst_file)

		return dst_file

	except OSError as e:
		# 在实际应用中，建议这里改用 logging 记录异常
		print(f'复制入口文件失败：{e}')
		return None


def _getAppName() -> str | None:
	"""尝试获取当前模块/应用名。

	优先通过 `__package__` 获取（适用于 `python -m projects.xxx.main` 方式运行），
	若不可用则尝试通过物理路径推断。无法获取时返回 `None`。

	Returns:
		当前模块/应用名，如果无法确定则返回 `None`
	"""
	# 1. 优先尝试：通过 __package__ (适用于 python -m projects.xxx.main)
	pkg = getattr(__main__, '__package__', None)
	if pkg and isinstance(pkg, str):
		return pkg.split('.')[-1]

	# 2. 补救方案：如果不是 -m 启动，尝试通过物理路径推断
	if not sys.argv or not sys.argv[0]:
		return None

	fpEntry = Path(sys.argv[0]).resolve()
	if fpEntry.suffix != '.py':
		return None

	return fpEntry.parent.name


def getAppName() -> str:
	"""获取当前运行的应用名称。

	如果无法确定应用名称，将抛出 `RuntimeError` 异常。

	Returns:
		当前运行的应用名称

	Raises:
		RuntimeError: 当无法确定当前运行入口名称时抛出
	"""
	name = _getAppName()
	if not name:
		msg = (
			'无法确定当前的运行入口名称 (App Name)。'
			"请确保你是通过 'python -m projects.xxx.main' 或标准文件路径方式运行。"
		)
		raise RuntimeError(msg)
	return name


def createLRS(
	optimizer: Optimizer, milestones: list[int] | None = None, gamma: float = 0.7
) -> lrs.LRScheduler:
	"""创建学习率调度器链。

	组合使用线性学习率预热和多步学习率衰减策略。

	Args:
		optimizer: 需要应用学习率调度的优化器
		milestones: 学习率衰减的阶段，_可选_，默认值为 `[5, 20, 30]`
		gamma: 学习率衰减因子，_可选_，默认值为 `0.7`

	Returns:
		组合后的学习率调度器
	"""
	if milestones is None:
		milestones = [5, 20, 30]
	scheduler1 = lrs.LinearLR(optimizer, 0.1, total_iters=milestones[0])
	scheduler2 = lrs.MultiStepLR(optimizer, milestones[1:], gamma)
	return lrs.ChainedScheduler([scheduler1, scheduler2], optimizer)


@dataclass
class ModelConfig:
	"""纵向联邦学习模型配置类，定义了底端网络和顶端网络的配置参数

	Args:
		lPartyDims: 各参与方输入数据的维度列表
		lTopDims: 顶端网络各层的维度列表
		_getBtmNets: 获取底端网络的回调函数，_可选_
		_getTopNet: 获取顶端网络的回调函数，_可选_
	"""

	lPartyDims: list[int]
	"""各参与方输入数据的维度列表"""
	lTopDims: list[int]
	"""顶端网络各层的维度列表"""
	_getBtmNets: Callable[[], nn.ModuleList] | None = None
	"""获取底端网络的回调函数，_可选_"""
	_getTopNet: Callable[[], nn.Module] | None = None
	"""获取顶端网络的回调函数，_可选_"""

	def getBtmNets(self) -> nn.ModuleList:
		"""获取各参与方的底端网络列表。

		如果 `_getBtmNets` 回调函数被提供，则调用该函数获取网络；
		否则，使用默认的 `ResNet18` 构建网络列表。

		Returns:
			各参与方的底端网络模块列表
		"""
		if self._getBtmNets:
			return self._getBtmNets()
		return nn.ModuleList([ResNet18(dim) for dim in self.lPartyDims])

	def getTopNet(self) -> nn.Module:
		"""获取顶端网络。

		如果 `_getTopNet` 回调函数被提供，则调用该函数获取网络；
		否则，使用默认的 `FCN` 构建顶端网络。

		Returns:
			顶端网络模块
		"""
		if self._getTopNet:
			return self._getTopNet()
		return FCN(self.lTopDims)


@dataclass
class RunConfig:
	"""训练运行配置类，定义了学习率、训练轮数等超参数

	Args:
		lr: 学习率，默认为 `1e-3`
		epochs: 训练轮数，默认为 `40`
	"""

	lr: float = 1e-3
	"""学习率，默认为 `1e-3`"""
	epochs: int = 40
	"""训练轮数，默认为 `40`"""
	_configOptims: Callable[[BaseVFLArch, float], OPT_TYPE] | None = None
	"""配置优化器和学习率调度器的回调函数，_可选_"""

	def configOptims(self, m: BaseVFLArch) -> OPT_TYPE:
		"""配置优化器和学习率调度器。

		如果 `_configOptims()` 回调函数被提供，则调用该函数获取优化器和学习率调度器；
		否则，使用默认的 Adam 优化器和学习率调度器。

		Args:
			m: 模型架构对象，包含底端网络和顶端网络

		Returns:
			包含优化器和学习率调度器的元组
		"""
		if self._configOptims:
			return self._configOptims(m, self.lr)

		optBtms = [Adam(net.parameters(), self.lr) for net in m.lBtmNets]
		optTop = Adam(m.zTopNet.parameters(), self.lr)

		lrsBtms = [createLRS(opt) for opt in optBtms]
		lrsTop = createLRS(optTop)
		return [*optBtms, optTop], [*lrsBtms, lrsTop]


@dataclass
class AppConfig:
	"""应用主配置类，整合了数据、模型和运行配置

	Args:
		data: 数据配置对象
		model: 模型配置对象
		run: 运行配置对象
		sName: 应用名称，默认通过 `getAppName()` 获取
		dpRoot: 日志根目录，_初始化后只读_
		_dpRoot: 内部初始化用参数，指定日志根目录，_可选_
		fpCkpt: 检查点文件路径，_可选_
	"""

	data: DataConfig
	"""数据配置对象"""
	model: ModelConfig
	"""模型配置对象"""
	run: RunConfig
	"""运行配置对象"""
	sName: str = field(default_factory=getAppName)
	"""应用名称，默认通过 `getAppName()` 获取"""
	dpRoot: Path = field(init=False)
	"""日志根目录，_初始化后只读_"""
	_dpRoot: InitVar[Path | None] = None
	"""内部初始化用参数，指定日志根目录，_可选_"""
	fpCkpt: str | None = None
	"""检查点文件路径，_可选_"""

	def __post_init__(self, _dpRoot: Path | None) -> None:
		"""初始化实例。"""
		self.dpRoot = _dpRoot or Path('data/logs') / f'{self.sName}-{self.data.sName}'

	def getRunName(self) -> str:
		"""生成唯一的实验运行名称。

		Returns:
			基于应用名、数据集名、参与方数量和网络层数构建的运行名称
		"""
		return (
			f'{self.sName}_{self.data.sName}_P{len(self.model.lPartyDims)}L{len(self.model.lTopDims) - 1}'
		)
