# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Forked from https://github.com/google/iree

import argparse

from tensorflow.python import pywrap_mlir  # pylint: disable=no-name-in-module


def optimize(model_path: str, output_path: str):
    pass_pipeline = ",".join([
        "symbol-dce", "tf-standard-pipeline",
        "func(tf-device-index-selector)", "inline", "canonicalize",
        "func(tf-device-decompose-resource-ops)",
        "func(tf-functional-control-flow-to-cfg)", "inline", "symbol-dce",
        "canonicalize", "tf-saved-model-optimize-global-tensors",
        "tf-saved-model-freeze-global-tensors"
    ])
    with open(model_path) as file:
        mlir = file.read()

    with open(output_path, "w") as file:
        file.write(
            pywrap_mlir.experimental_run_pass_pipeline(mlir, pass_pipeline,
                                                       True))


def main():
    parser = argparse.ArgumentParser(
        description="Optimize model in tf dialect")
    parser.add_argument("model_path",
                        metavar="model-path",
                        help="Path to tf mlir model")
    parser.add_argument("output_path",
                        metavar="output-path",
                        help="Output path")
    args = parser.parse_args()

    optimize(args.model_path, args.output_path)


if __name__ == "__main__":
    main()
