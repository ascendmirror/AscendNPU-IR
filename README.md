

AscendNPU IR（BiSheng IR）项目
======================

AscendNPU IR（BiSheng IR）是基于MLIR构建的中间表示（IR），主要面向Ascend NPU架构。它是CANN（Compute Architecture for Neural Networks）的一部分，用于支持AI编译器的开发和优化。

## AscendNPU IR（BiSheng IR）在CANN中的位置

AscendNPU IR（BiSheng IR）是CANN的重要组成部分，它位于AI框架和底层硬件之间，为上层框架提供统一的IR接口，并支持对Ascend NPU的高效代码生成和优化。

## 使用AscendNPU IR（BiSheng IR）

AscendNPU IR 提供了多种Dialect，包括但不限于：

- **HACC**: 高性能算子相关Dialect。
- **HFusion**: 支持算子融合的Dialect。
- **HIVM**: 面向NPU虚拟机的IR，支持运行时指令生成。
- **Annotation**: 用于添加注解信息的Dialect。

这些Dialect可以通过MLIR工具链进行解析、转换和优化。

## 安装构建BiSheng IR所需的预编译组件

在构建BiSheng IR之前，请确保安装以下依赖项：

- CMake 3.20 或更高版本
- LLVM 和 MLIR 源码（与BiSheng IR兼容的版本）
- Python 3.x（用于构建脚本）
- GCC 或 Clang 编译器
- 其他构建工具（如ninja、make等）

## 构建BiSheng IR作为外部LLVM项目

BiSheng IR可以作为LLVM的外部项目进行构建。以下是构建步骤的概要：

1. 下载并配置LLVM和MLIR源码。
2. 配置BiSheng IR的CMakeLists.txt，将其作为LLVM的外部项目添加。
3. 使用CMake构建BiSheng IR，并生成相应的库和工具。

### 示例：使用CMake构建

```bash
mkdir build && cd build
cmake -G Ninja ..
cmake --build .
```

## 示例

### minimal-bishengir-opt 示例

`minimal-bishengir-opt` 是一个简单的工具，用于演示如何操作BiSheng IR的Dialect。其主函数如下：

```cpp
int main(int argc, char **argv)
```

该工具可扩展以支持更多BiSheng IR的优化和转换逻辑。

## 贡献

欢迎对AscendNPU IR（BiSheng IR）进行贡献。请确保遵循LLVM和MLIR的开发规范，并提交符合项目风格的PR。

## 许可证

本项目遵循LLVM项目许可证，详细信息请参阅 [LICENSE](LICENSE) 文件。