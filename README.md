

AscendNPU IR（BiSheng IR）项目
==========================

AscendNPU IR（BiSheng IR）是CANN（Compute Architecture for Neural Networks）的一部分，用于构建和优化神经网络计算。

### AscendNPU IR（BiSheng IR）在CANN中的位置

BiSheng IR在CANN架构中承担中间表示的角色，为上层模型和下层硬件之间提供一个高效的转换桥梁。

### 使用AscendNPU IR（BiSheng IR）

本项目提供了一套工具和转换规则，用于将标准的MLIR操作转换为适用于AscendNPU的特定表示。这包括从多种方言（如Arith, GPU, Linalg, Math, Tensor）到HFusion和HIVM的转换。

### 安装构建BiSheng IR所需的预编译组件

在开始构建BiSheng IR之前，请确保已安装以下组件：

- LLVM和MLIR的开发库
- CMake
- 支持C++17的编译器

### 构建BiShengIR作为外部LLVM项目

1. 下载LLVM和MLIR源代码。
2. 配置CMake以将BiSheng IR作为外部项目添加。
3. 编译并安装BiSheng IR。

### 如何构建端到端用例

端到端示例展示了BiSheng IR如何将高级操作转换为硬件友好的形式。请参阅`examples`目录中的具体示例。

示例
-------

### HIVM Vector Add

这是一个简单的向量加法示例，演示了如何在HIVM中使用BiSheng IR。

#### 要求

- BiSheng IR已正确构建并安装。
- 支持Ascend NPU的运行环境。

#### 如何构建示例

1. 进入`examples/HIVM/VecAdd`目录。
2. 使用CMake配置并构建项目。
3. 运行生成的可执行文件。

有关更多细节，请参阅`README_zh.md`文件。