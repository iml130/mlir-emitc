// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gmock/gmock.h"

#include "emitc/emitc_tosa.h"
#include "emitc/emitc_types.h"

namespace {

using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pointwise;

const float EPSILON = 5e-4;

// Unary elementwise ops
TEST(tosa, reciprocal) {
  Tensor<float, 4> t0{1.0f, 2.0f, 3.0f, 4.0f};
  Tensor<float, 4> excpected{1.0f, 0.5f, 0.3333f, 0.25f};

  Tensor<float, 4> result = tosa::reciprocal(t0);

  EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), excpected));
}

// Binary elementwise ops
TEST(tosa, mul) {
  // no shift
  Tensor2D<long, 2, 2> s2{3, 1, 4, 9};
  Tensor2D<long, 2, 2> t2{-2, 8, 6, -10};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    int32_t shift{0};
    return tosa::mul<Tensor2D<long, 2, 2>>(s2, t2, shift);
  };
  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {-6, 8, 24, -90}));

  // shift
  Tensor0D<int32_t> s3{3};
  Tensor0D<int32_t> t3{2};

  auto lambda_2d_int = [&s3, &t3]() -> Tensor0D<int32_t> {
    int32_t shift{2};
    return tosa::mul<Tensor0D<int32_t>>(s3, t3, shift);
  };

  EXPECT_THAT(lambda_2d_int(), Pointwise(Eq(), {2}));
}

// Other ops
TEST(tosa, fully_connected) {
  using InputType = Tensor2D<float, 2, 5>;  // N CIN
  using WeightType = Tensor2D<float, 2, 5>; // COUT CIN
  using BiasType = Tensor1D<float, 2>;      // COUT
  using ResultType = Tensor2D<float, 2, 2>; // N COUT
  InputType input{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  WeightType weights{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  BiasType bias{100, 200};
  ResultType expected_result{155, 330, 230, 530};
  ResultType result = tosa::fully_connected<ResultType>(input, weights, bias);

  EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
}

} // namespace
