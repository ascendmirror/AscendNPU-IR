# The AscendNPU IR (BiSheng IR) Project

## Where The AscendNPU IR (BiSheng IR) Is In CANN

![](./doc/pic/ascendnpu-ir-in-cann.png)

## Using AscendNPU IR (BiSheng IR)

### Installing pre-builts that are required to build BiShengIR

1. Extract the package (available in the [release page](https://gitee.com/ascend/ascendnpu-ir/releases)) containing the pre-builts corresponding to your target machine to any location. After install, it should contain the following contents:

   ```bash
   ├── lib
     └── libBiShengIR.so     // used to build bishengir dialects
   └── bin
     └── bishengir-compile   // used to compile `.mlir` to binary
     └── bishengir-yaml-gen  // used to generate files from yaml
   ```

2. Set environment variable to the installed path:

  ```bash
  export BISHENG_IR_INSTALL_PATH= ...
  ```

### Building BiShengIR as an external LLVM Project

1. Find the version of LLVM that BiShengIR builds against. Check `cmake/llvm-release-tag.txt` to see the current version.
  
    For example, if it says: "llvm.19.1.7", it means that the version of BiShengIR you have builds against [LLVM](https://github.com/llvm/llvm-project/tree/llvmorg-19.1.7) release `llvmorg-19.1.7`.

2. `git checkout` LLVM at this revision. Optionally, make additional modifications to LLVM.

3. Add `bishengir` project as a third-party submodule to LLVM.

    ```bash
    git submodule add https://gitee.com/ascend/ascendnpu-ir.git third-party/bishengir
    ```

4. [Build LLVM](https://llvm.org/docs/CMake.html). This is an example cmake config:

    ```bash
    cd ${HOME}/llvm-project  # your clone of LLVM.
    mkdir build
    cd build
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ../llvm \
      -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
      -DLLVM_EXTERNAL_PROJECTS="bishengir" \
      -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR=${HOME}/llvm-project/third-party/bishengir \
      -DBISHENG_IR_INSTALL_PATH=${BISHENG_IR_INSTALL_PATH}
    ```

5. You can build the "check-bishengir" target to build and run unit testcases:

   ```bash
   cmake --build . --target "check-bishengir"
   ```

### Building an end-to-end example

Please refer to the `examples` directory.