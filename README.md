# MLIR EmitC

![Build and test](https://github.com/iml130/mlir-emitc/workflows/Build%20and%20test/badge.svg)

EmitC is a MLIR dialect to emit C++ code. The initial checked in code is forked from https://reviews.llvm.org/D76571.

## Getting Started
### Clone

```shell
git clone https://github.com/iml130/mlir-emitc.git
cd mlir-emitc
git submodule update --init
```

### Build and Run

The setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. You can use the `build_tools/build.sh` shell script to configure, build and install LLVM and MLIR.

To build and launch the tests, run
```shell
mkdir build && cd build
cmake -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-emitc
```

**Note**: If don't use `build_tools/build.sh`, make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

To additionally build and execute the unittests, run
```shell
cmake --build . --target MLIREmitCAllTests
./unittests/MLIREmitCAllTests
```


## Supported Conversions and Translations

Conversions are supported for [MLIR-HLO](https://github.com/tensorflow/mlir-hlo) ops and some ops of the standard and SCF dialect.
The `emitc-opt` tool enables conversions via the following options:

| option                        |                                                                 |
| :---------------------------- |:--------------------------------------------------------------- |
| `--convert-mhlo-to-emitc `    | Convert from MHLO dialect to EmitC dialect                      |
| `--convert-scf-to-emitc`      | Convert SCF dialect to EmitC dialect, replacing IfOp and ForOp. |
| `--convert-std-to-emitc `     | Convert std dialect to EmitC dialect, replacing IndexCastOp.    |
| `--preprocess-mhlo-for-emitc` | Apply MHLO to MHLO transformations for some ops.                |

The currently supported MHLO ops are listed in the [docs/mhlo-op-coverage.md](docs/mhlo-op-coverage.md) document.

Furthermore, the `emitc-opt` tools supports the conversion from `mhlo.constant` to `constant` via `--convert-mhlo-const-to-std `. This conversion is necessary because EmitC does not implement a conversion from `mhlo.constant`, but does support `constant` in the translation to C++.

After converting to EmitC, C++ code can be emited using `emitc-translate --mlir-to-cpp`.
