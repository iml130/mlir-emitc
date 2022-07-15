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

import argparse

from tensorflow.python import pywrap_mlir  # pylint: disable=no-name-in-module


def translate(model_path: str, output_path: str, exported_names: str):
    with open(output_path, "w") as file:
        file.write(
            pywrap_mlir.experimental_convert_saved_model_to_mlir(
                model_path, exported_names, True))


def main():
    parser = argparse.ArgumentParser(
        description="Translate saved model to tf mlir dialect")
    parser.add_argument(
        "--exported-names",
        help=
        "Names to export from SavedModel, separated by ','. Empty (the default) means export all." \
        " (see tf-savedmodel-exported-names option of tf-mlir-translate)",
    )
    parser.add_argument("model_path",
                        metavar="model-path",
                        help="Path to SavedModel")
    parser.add_argument("output_path",
                        metavar="output-path",
                        help="Output path")
    args = parser.parse_args()

    translate(args.model_path, args.output_path, args.exported_names)


if __name__ == "__main__":
    main()
