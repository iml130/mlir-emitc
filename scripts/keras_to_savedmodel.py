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

import argparse

import tensorflow as tf


class Module(tf.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def predict(self, *args):
        return self._model.call(list(args), training=False)


def extract_tensor_specs(model, batch_size: int):
    def extract_tensor_spec(tensor):
        shape = list(tensor.shape)

        if shape[0] is None:
            shape[0] = batch_size

        return tf.TensorSpec(shape, tensor.dtype)

    return [extract_tensor_spec(tensor) for tensor in model.inputs]


def translate(model_path: str, output_path: str, batch_size: int):
    model = tf.keras.models.load_model(model_path)

    # Produce a concrete function to compile.
    module = Module(model)
    module.predict = tf.function(func=module.predict,
                                 input_signature=extract_tensor_specs(
                                     model, batch_size=batch_size))

    tf.saved_model.save(module, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Translate keras model to saved model format")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Set the batch size for inference",
    )
    parser.add_argument("model_path", metavar="model-path", help="Path to keras model")
    parser.add_argument("output_path", metavar="output-path", help="Output directory")
    args = parser.parse_args()

    translate(args.model_path, args.output_path, args.batch_size)


if __name__ == "__main__":
    main()
