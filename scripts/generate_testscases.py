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
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


def generate_examples(model, count: int, seed: int, batch_size: int):
    def generate_example(model_input):
        shape = list(model_input.shape)

        if shape[0] is None:
            shape[0] = batch_size

        dtype = model_input.dtype

        if dtype == np.float32:
            return np.random.uniform(low=0.0, high=1.0,
                                     size=shape).astype(np.float32)
        else:
            raise ValueError(
                f"Random number generation for dtype '{dtype}' not implemented.'"
            )

    np.random.seed(seed=seed)
    result = []
    for _ in range(count):
        model_inputs = [
            generate_example(model_input) for model_input in model.inputs
        ]

        start = time.time()
        outputs = model.predict(model_inputs)
        end = time.time()
        print(f"Inference with tensorflow in Python took {end-start} seconds")

        if not isinstance(outputs, list):
            outputs = [outputs]

        result.append((model_inputs, outputs))
    return result


def save_examples(path: str, examples, file_format: str):
    def c_type_specifier(array: np.ndarray) -> str:
        dtype = array.dtype
        if dtype == np.float32:
            return "float"
        else:
            raise ValueError(
                f"C type conversion for dtype '{dtype}' not implemented.'")

    def c_value(value):
        if isinstance(value, np.float32):
            return f"{value}f"
        else:
            return f"{value}"

    def function_name():
        return "predict"

    if file_format == "header":
        with open(Path(path) / "testcases.h", mode="w") as output_file:
            for i, (inputs, outputs) in enumerate(examples):
                for j, curr_input in enumerate(inputs):
                    output_file.write(
                        f"{c_type_specifier(curr_input)} "
                        f"example_{i}_input_{j}[{curr_input.size}] = "
                        f"{{{', '.join(map(c_value, curr_input.flat))}}};\n")
                for j, curr_output in enumerate(outputs):
                    output_file.write(
                        f"{c_type_specifier(curr_output)} "
                        f"example_{i}_output_{j}[{curr_output.size}] = "
                        f"{{{', '.join(map(c_value, curr_output.flat))}}};\n")
    elif file_format == "cpp":
        with open(Path(path) / "test.cpp", mode="w") as output_file:
            output_file.write("#include <iostream>\n")
            output_file.write('#include "model_generated.h"\n')
            output_file.write("\n")
            output_file.write("""template <typename T, typename U>
bool check_tensor(T result, U expected, float eps, bool print_error) {
    bool error = false;
    for (size_t i = 0; i < result.size(); i++) {
        auto err = std::abs(result[i] - expected[i]);
        error |= (err > eps);
        if (print_error) {
            std::cout << "index " << i << " -> " << err << std::endl;
        }
    }
    return error;
}
  """)
            output_file.write("\n")
            output_file.write("int main(){\n")

            assert len(examples) == 1
            inputs, outputs = examples[0]

            for i, curr_input in enumerate(inputs):
                output_file.write(
                    f"Tensor<{c_type_specifier(curr_input)}, "
                    f"{','.join(map(str,curr_input.shape))}> "
                    f"input{i}{{{', '.join(map(c_value, curr_input.flat))}}};\n"
                )

            for i, curr_output in enumerate(outputs):
                shape = curr_output.shape
                c_type = c_type_specifier(curr_output)
                shape_str = ','.join(map(str, shape))

                output_file.write(
                    f"static const {c_type} output{i}[] = {{{', '.join(map(c_value, curr_output.flat))}}};\n"
                )
                output_file.write(
                    f"Tensor<{c_type}, {shape_str}> result{i};\n")

            output_file.write("bool error = false;\n")
            output_file.write("float EPS = 1e-4;\n")

            if len(outputs) > 1:
                results_str = ",".join(f"result{i}"
                                       for i in range(len(outputs)))
                output_file.write(f"std::tie({results_str})")
            else:
                output_file.write("result0")

            output_file.write(" = ")

            inputs_str = ",".join(f"input{i}" for i in range(len(inputs)))
            output_file.write(f"{function_name()}({inputs_str});\n\n")

            for i in range(len(outputs)):
                output_file.write(
                    f"error |= check_tensor(result{i}, output{i}, EPS, false);\n"
                )
            output_file.write("\n")

            output_file.write(
                'std::cout << (error ? "Error" : "Correct") << std::endl;\n')
            output_file.write("return error;}\n")
    else:
        raise ValueError(
            f"Output file format '{file_format}' not implemented.'")


def generate(
    model_path: str,
    output_path: str,
    file_format: str,
    count: int,
    seed: int,
    batch_size: int,
):
    model = tf.keras.models.load_model(model_path)
    examples = generate_examples(model, count, seed, batch_size)
    save_examples(output_path, examples, file_format)


def main():
    parser = argparse.ArgumentParser(
        description="Generate input/output examples for a keras model")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of testcases to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed for the RNG",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Set the batch size for the testcases",
    )
    parser.add_argument("--file-format",
                        choices=["header", "cpp"],
                        required=True)
    parser.add_argument("model_path",
                        metavar="model-path",
                        help="Path to keras model")
    parser.add_argument("output", help="Path to output directory")
    args = parser.parse_args()

    generate(args.model_path, args.output, args.file_format, args.count,
             args.seed, args.batch_size)


if __name__ == "__main__":
    main()
