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

# forked from https://github.com/google/iree

import argparse
from pathlib import Path
from shutil import which
import subprocess
import sys
import tempfile
from typing import Optional

import tensorflow as tf


class Module(tf.Module):

    def __init__(self, model):
        self._model = model
    
    def predict(self, x):
        return self._model.call(x, training=False)


def extract_tensor_specs(model, batch_size: Optional[int]):
    def extract_tensor_spec(input):
        shape = list(input.shape)

        if batch_size is not None:
            shape[0] = batch_size

        return tf.TensorSpec(shape, input.dtype)
  
    return [extract_tensor_spec(input) for input in model.inputs]


def convert_to_mhlo(model_dir):
    tf_translate = "tf-mlir-translate"
    tf_opt = "tf-opt"

    assert which(tf_translate) is not None, f"Make sure '{tf_translate}' is in your PATH"
    assert which(tf_opt) is not None, f"Make sure '{tf_opt}' is in your PATH"

    p_translate = subprocess.Popen([
        tf_translate,
        "--savedmodel-objectgraph-to-mlir",
        "--tf-savedmodel-exported-names=predict",
        model_dir
    ], stdout=subprocess.PIPE, universal_newlines=True)

    p_opt = subprocess.Popen([
        tf_opt,
        "--symbol-dce",
        "--tf-executor-graph-pruning",
        "--tf-guarantee-all-funcs-one-use",
        "--tf-standard-pipeline",
        "--tf-device-index-selector",
        "--inline",
        "--canonicalize",
        "--tf-device-decompose-resource-ops",
        "--tf-shape-inference",
        "--tf-functional-control-flow-to-cfg",
        "--inline",
        "--symbol-dce",
        "--canonicalize",
        "--tf-saved-model-optimize-global-tensors",
        "--tf-saved-model-freeze-global-tensors",
        "--xla-legalize-tf",
        "--canonicalize",
        "--tf-saved-model-optimize-global-tensors"
    ], stdin=p_translate.stdout, stdout=subprocess.PIPE, universal_newlines=True)
    
    p_translate.stdout.close()
    
    output = p_opt.communicate()[0]

    return output


def translate(file_path: str, batch_size: int):
    model = tf.keras.models.load_model(file_path)
    
    # Produce a concrete function to compile.
    module = Module(model)
    module.predict = tf.function(input_signature=extract_tensor_specs(model, batch_size=batch_size))(module.predict)

    with tempfile.TemporaryDirectory(prefix="tf-mlir-") as tmpdir:
        tf.saved_model.save(module, tmpdir)

        output = convert_to_mhlo(tmpdir)
        print(output)


def main():
    parser = argparse.ArgumentParser(description="Translate keras model to mhlo dialect")
    parser.add_argument("--batch-size", type=int, default=1, help="Set the batch size for inference")
    parser.add_argument("input", help="Keras model")
    args = parser.parse_args()

    translate(args.input, args.batch_size)


if __name__ == "__main__":
    main()
