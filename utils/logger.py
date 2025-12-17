"""日志工具模块。

该模块提供了创建和配置文件日志记录器的功能，用于统一管理应用程序的日志输出。
"""

import logging as log

from .common import Path


def createFileLogger(fpLog: Path | str = 'app.log', iLevel: int = log.DEBUG) -> log.Logger:
	"""创建文件日志记录器。

	创建或获取一个配置好的日志记录器，用于将日志消息输出到指定文件。
	该函数会避免重复添加文件处理器，确保日志配置的一致性。

	Args:
		fpLog: 日志文件路径，可以是 Path 对象或字符串，默认为 `'app.log'`
		iLevel: 日志记录级别，默认为 `log.DEBUG`

	Returns:
		配置好的日志记录器实例
	"""
	# 创建或获取 logger 实例
	logger = log.getLogger(__name__)
	logger.setLevel(iLevel)  # 设置日志记录级别

	# 避免重复添加处理程序
	if not any(isinstance(handler, log.FileHandler) for handler in logger.handlers):
		# 创建文件处理器（FileHandler）
		hdlr = log.FileHandler(fpLog, encoding='utf-8')

		# 定义日志格式
		fmt = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
		hdlr.setFormatter(fmt)

		# 将处理器添加到 logger
		logger.addHandler(hdlr)

	return logger
