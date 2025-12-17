"""垂直联邦学习（VFL）核心架构包。

本包提供了构建垂直联邦学习系统所需的核心组件，基于 PyTorch Lightning 框架进行封装，
简化了 VFL 模型的开发、训练和部署流程。

主要模块：

- `arch`: 定义 VFL 模型架构基类，包括 `LightningArch`（基础封装）和 `BaseVFLArch`（VFL 核心逻辑）
- `callback`: 提供 VFL 生命周期回调接口，支持通过钩子函数介入训练各阶段
- `module`: 封装数据加载流程，基于 `LightningDataModule` 规范数据集处理接口

本包是整个垂直联邦学习框架的基础，为上层应用提供了统一的模型开发和训练接口。
"""
