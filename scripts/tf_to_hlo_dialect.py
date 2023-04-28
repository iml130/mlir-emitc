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
#
# SPDX-License-Identifier: Apache-2.0

# Forked from https://github.com/google/iree

import argparse

from tensorflow.python import pywrap_mlir  # pylint: disable=no-name-in-module


def convert(model_path: str, output_path: str, hlo_dialect: str):
    pass_pipeline = ["tf-lower-to-mlprogram-and-hlo"]
    if hlo_dialect == "mhlo":
        pass_pipeline.append("stablehlo-legalize-to-hlo")
    pass_pipeline = ",".join(pass_pipeline)
    
    with open(model_path) as file:
        mlir = file.read()

    with open(output_path, "w") as file:
        file.write(
            pywrap_mlir.experimental_run_pass_pipeline(mlir, pass_pipeline,
                                                       True))


def main():
    parser = argparse.ArgumentParser(
        description="Convert model in tf dialect to mhlo dialect")
    parser.add_argument(
        "--hlo-dialect",
        type=str,
        choices=["mhlo", "stablehlo"],
        default="mhlo",
        help="Which flavor of HLO dialect to export",
    )
    parser.add_argument("model_path",
                        metavar="model-path",
                        help="Path to tf mlir model")
    parser.add_argument("output_path",
                        metavar="output-path",
                        help="Output path")
    args = parser.parse_args()

    convert(args.model_path, args.output_path, args.hlo_dialect)


if __name__ == "__main__":
    main()
