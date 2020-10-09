#!/bin/bash
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Drived from/inspired by MLIR NPComp and TensorFlow's MLIR-HLO.

if [[ $# -ne 3 ]] ; then
  echo "Usage: $0 <path/to/llvm/src> <build_dir> <install_dir>"
  exit 1
fi

# LLVM source
LLVM_SRC_DIR="$1"
build_dir="$2"
install_dir="$3"

if ! [ -f "$LLVM_SRC_DIR/llvm/CMakeLists.txt" ]; then
  echo "Expected the path to LLVM to be set correctly (got '$LLVM_SRC_DIR'): can't find CMakeLists.txt"
  exit 1
fi
echo "Using LLVM source dir: $LLVM_SRC_DIR"

# Setup directories.
echo "Building MLIR in $build_dir"
echo "Install MLIR to $install_dir"
mkdir -p "$build_dir"
mkdir -p "$install_dir"

echo "Beginning build (commands will echo)"
set -x

cmake -GNinja \
  "-H$LLVM_SRC_DIR/llvm" \
  "-B$build_dir" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_INCLUDE_TOOLS=ON \
  -DLLVM_BUILD_TOOLS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  "-DCMAKE_INSTALL_PREFIX=$install_dir" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=On

cmake --build "$build_dir" --target install
