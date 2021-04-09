#!/bin/bash
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

set -e

if [[ $# -ne 8 ]] ; then
  echo "Usage: $0 <path/to/model> <path/to/emitc/include/files> <path/to/emitc-opt> <path/to/tf-opt> <compiler> <batch-size> <seed> <output_dir>"
  echo
  echo "Both a keras and a tensorflow saved model is supported."
  echo
  echo "This script expects a python version in the PATH with a recent version of tensorflow installed."
  echo "You need to compile tensorflow from source to build the tf-opt program."
  echo "Tested with python 3.6.9 and tensorflow 2.4.0"

  exit 1
fi

MODEL=$1
EMITC_INCLUDE_DIR=$2
TF_OPT=$3
EMITC_OPT=$4
EMITC_TRANSLATE=$(dirname $EMITC_OPT)/emitc-translate
CPP_COMPILER=$5
BATCH_SIZE=$6
SEED=$7
OUTPUT_DIR=$8

echo "MODEL=$MODEL"
echo "EMITC_INCLUDE_DIR=$EMITC_INCLUDE_DIR"
echo "EMITC_OPT=$EMITC_OPT"
echo "EMITC_TRANSLATE=$EMITC_TRANSLATE"
echo "CPP_COMPILER=$CPP_COMPILER"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "SEED=$SEED"
echo "OUTPUT_DIR=$OUTPUT_DIR"

echo "Setting up output directory"
mkdir -p "$OUTPUT_DIR"

echo "Converting model to saved model format"
python model_to_savedmodel_with_predict_function.py --batch-size "$BATCH_SIZE" "$MODEL" "$OUTPUT_DIR"/model

echo "Translating saved model to tf dialect"
EXPORTED_NAME=predict
python savedmodel_to_tf_dialect.py --exported-names "$EXPORTED_NAME" "$OUTPUT_DIR"/model "$OUTPUT_DIR"/model_tf.mlir

echo "Optimizing tf dialect"
python optimize_tf_dialect.py "$OUTPUT_DIR"/model_tf.mlir "$OUTPUT_DIR"/model_tf_opt.mlir

echo "Converting tf dialect to tosa dialect"
"$TF_OPT" --tf-to-tosa-pipeline "$OUTPUT_DIR"/model_tf_opt.mlir > "$OUTPUT_DIR"/model_tosa.mlir

echo "Removing tf._input_shapes attribute"
sed "s/tf._input_shapes =.*]//" "$OUTPUT_DIR"/model_tosa.mlir > "$OUTPUT_DIR"/model_tosa_noattr.mlir
sed -i "s/, }/}/" "$OUTPUT_DIR"/model_mhlo_noattr.mlir

echo "Fixing function name"
FUNCTION_NAME=$(grep -oe "@[^(]*" "$OUTPUT_DIR"/model_tosa_noattr.mlir)
sed "s/$FUNCTION_NAME/@predict/g" "$OUTPUT_DIR"/model_tosa_noattr.mlir > "$OUTPUT_DIR"/model_fix_name.mlir

echo "Converting tosa dialect to emitc dialect"
"$EMITC_OPT" --convert-tosa-to-emitc "$OUTPUT_DIR"/model_fix_name.mlir > "$OUTPUT_DIR"/model_emitc.mlir

echo "Translating emitc dialect to cpp header"
"$EMITC_TRANSLATE" --mlir-to-cpp "$OUTPUT_DIR"/model_emitc.mlir > "$OUTPUT_DIR"/model_generated.h

echo "Generating test case"
python generate_testscases.py --file-format cpp --count 1 --batch-size "$BATCH_SIZE" --seed "$SEED" "$MODEL" "$OUTPUT_DIR"

echo "Compiling test case"
"$CPP_COMPILER" "$OUTPUT_DIR"/test.cpp -O3 -I "$EMITC_INCLUDE_DIR" -I "$OUTPUT_DIR" -o "$OUTPUT_DIR"/test

echo "Running test case"
time "$OUTPUT_DIR"/test
