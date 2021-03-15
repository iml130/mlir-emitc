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

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import argparse
import numpy as np
import random


def set_fake_weights(model):
    for layer in model.layers:
        if layer.get_weights():
            new_weights = []
            for weight in layer.get_weights():
                const_weight = np.full(weight.shape, 0.5)
                new_weights.append(const_weight)
            layer.set_weights(new_weights)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Downloads MobileNetV2 keras model")
    parser.add_argument(
        "--output-file",
        default="mobilenet_v2.h5",
        help="Output file",
    )
    parser.add_argument(
        "--fake-weights",
        action='store_true',
        default=False,
        help="Sets all weights to 0.5"
    )
    args = parser.parse_args()

    model = MobileNetV2(weights='imagenet')
    if args.fake_weights:
        set_fake_weights(model)
    model.save(args.output_file)


if __name__ == "__main__":
    main()
