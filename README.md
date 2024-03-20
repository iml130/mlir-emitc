<!--
SPDX-FileCopyrightText: Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->
# MLIR-EmitC

![Build and test](https://github.com/iml130/mlir-emitc/workflows/Build%20and%20test/badge.svg)

MLIR-EmitC provides a way to translate ML models into C++ code. The repository
contains scripts and tools to translate Keras and TensorFlow models into the
[TOSA](https://mlir.llvm.org/docs/Dialects/TOSA/) and
[StableHLO](https://github.com/openxla/stablehlo/) dialect and to convert those to
[EmitC](https://mlir.llvm.org/docs/Dialects/EmitC/).
The latter is used to generate calls to a reference implementation.

**The [EmitC](https://mlir.llvm.org/docs/Dialects/EmitC/) dialect itself, as well as the C++ emitter, are part of MLIR core and are no longer provided via this repository.**

The initial EmitC dialect and C++ emitter checked into this repository were forked from https://reviews.llvm.org/D76571.

**DISCLAIMER:** This is a research project and not intended for everyday use. The code is made available without any support. However, we welcome any kind of feedback via the issue tracker.


## Getting Started
### Clone

```shell
git clone https://github.com/iml130/mlir-emitc.git
cd mlir-emitc
git submodule update --init
```

### Build and Run

There are two variants to build EmitC: As part of an LLVM/MLIR build (via the LLVM external projects mechanism) and against a pre-built LLVM/MLIR.

#### Building with pre-built LLVM/MLIR

The setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. You can use the [`build_tools/build_mlir.sh`](https://github.com/iml130/mlir-emitc/blob/main/build_tools/build_mlir.sh) shell script to configure, build and install LLVM and MLIR.

**Note**: The hash of the latest tested LLVM version is given in [`build_tools/llvm_version.txt`](https://github.com/iml130/mlir-emitc/blob/main/build_tools/llvm_version.txt). Since MLIR evolves fast, it is possible that EmitC fails to build with a newer LLVM.

To build and launch the tests, run
```shell
mkdir build && cd build
cmake -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-emitc
```

**Note**: If you don't use `build_tools/build_mlir.sh`, make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

To additionally build and execute the unittests, run
```shell
cmake --build . --target MLIREmitCTests
./reference-implementation/unittests/MLIREmitCTests
```

#### Bulding as part of an LLVM/MLIR build

MLIR-EmitC can also be built as part of an LLVM/MLIR build, using the `LLVM_EXTERNAL_PROJECTS` mechanism (see https://llvm.org/docs/CMake.html).

To build and launch the tests, run
```shell
mkdir build && cd build
cmake -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DEMITC_ENABLE_HLO=OFF -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_EXTERNAL_PROJECTS="mlir-emitc" -DLLVM_EXTERNAL_MLIR_EMITC_SOURCE_DIR=`realpath ../` -DLLVM_TARGETS_TO_BUILD=host ${ROOT_PATH_TO_llvm-project}/llvm
cmake --build . --target check-emitc
```

## Supported Conversions and Translations

Conversions are supported for [StableHLO](https://github.com/openxla/stablehlo/) ops and some ops of the arith and Tensor dialect.
In addition, support for converting Tensor Operator Set Architecture [(TOSA)](https://mlir.llvm.org/docs/Dialects/TOSA/) dialect to EmitC is emerging.
The `emitc-opt` tool supports the following options:

| option                                     |                                                                          |
| :----------------------------------------- |:------------------------------------------------------------------------ |
| `--convert-func-to-emitc`                  | Convert Func dialect to EmitC dialect                                    |
| `--convert-scf-to-emitc`                   | Convert SCF dialect to EmitC dialect, maintaining structured control flow|
| `--convert-stablehlo-region-ops-to-emitc ` | Convert StableHLO operations containing regions to EmitC dialect.        |
| `--convert-stablehlo-to-emitc `            | Convert from StableHLO dialect to EmitC dialect.                         |
| `--convert-arith-to-emitc-ext `            | Convert arith dialect to EmitC dialect (extended).                       |
| `--convert-tensor-to-emitc `               | Convert tensor dialect to EmitC dialect.                                 |
| `--convert-tosa-to-emitc `                 | Convert TOSA dialect to EmitC dialect.                                   |
| `--insert-emitc-stablehlo-include`         | Insert an EmitC include for the StableHLO dialect.                       |
| `--insert-emitc-arith-include`             | Insert an EmitC include for the arith dialect.                           |
| `--insert-emitc-tensor-include`            | Insert an EmitC include for the tensor dialect.                          |
| `--insert-emitc-tosa-include`              | Insert an EmitC include for the TOSA dialect.                            |
| `--stablehlo-to-emitc-pipeline`            | Run the StableHLO to EmitC pipeline.                                     |
| `--arith-to-emitc-pipeline`                | Run the Arithmetic to EmitC pipeline.                                    |
| `--tensor-to-emitc-pipeline`               | Run the Tensor to EmitC pipeline.                                        |
| `--tosa-to-emitc-pipeline`                 | Run the TOSA to EmitC pipeline.                                          |

The currently supported StableHLO ops are listed in the [docs/stablehlo-op-coverage.md](docs/stablehlo-op-coverage.md) document.
Supported TOSA ops are listed in the [docs/tosa-op-coverage.md](docs/tosa-op-coverage.md) document.

After converting to EmitC dialect, C++ code can be emitted using `emitc-translate --mlir-to-cpp`.
Furthermore, `emitc-translate` has specific support to emit code with variables declared at top using `--mlir-to-cpp --declare-variables-at-top`.
