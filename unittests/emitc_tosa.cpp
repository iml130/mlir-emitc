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
  Tensor2D<long, 2, 2> s0{3, 1, 4, 9};
  Tensor2D<long, 2, 2> t0{-2, 8, 6, -10};

  auto lambda_2d = [&s0, &t0]() -> Tensor2D<long, 2, 2> {
    return tosa::mul(s0, t0);
  };
  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {-6, 8, 24, -90}));

  Tensor1D<int32_t, 1> s1{3};
  Tensor1D<int32_t, 1> t1{2};

  auto lambda_1d_int = [&s1, &t1]() -> Tensor1D<int32_t, 1> {
    return tosa::mul(s1, t1);
  };

  EXPECT_THAT(lambda_1d_int(), Pointwise(Eq(), {6}));

  // shift
  Tensor2D<int32_t, 2, 2> s2{1, 2, 3, 4};
  Tensor2D<int32_t, 2, 2> t2{1, 2, 3, 4};

  auto lambda_1d_int_shift = [&s2, &t2]() -> Tensor2D<int32_t, 2, 2> {
    int32_t shift{2};
    return tosa::mul(s2, t2, shift);
  };

  EXPECT_THAT(lambda_1d_int_shift(), Pointwise(Eq(), {0, 1, 2, 4}));
}

// Other ops
TEST(tosa, conv2d) {
  using InputType = Tensor4D<float, 1, 4, 5, 2>;  // N H W C
  using WeightType = Tensor4D<float, 1, 3, 2, 2>; // COUT KH KW CIN
  using ResultType = Tensor4D<float, 1, 4, 5, 1>; // N H W C
  InputType input{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                  15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                  29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
  WeightType weights{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  ResultType expected_result{600,  736,  872,  1008, 476,  1310, 1466,
                             1622, 1778, 805,  2090, 2246, 2402, 2558,
                             1135, 1080, 1152, 1224, 1296, 524};

  Tensor1D<int64_t, 4> padding{1, 1, 0, 1}; // {pt, pb, pl, pr}
  Tensor1D<int64_t, 2> dilation{1, 1};
  Tensor1D<int64_t, 2> stride{1, 1};

  ResultType result =
      tosa::conv2d<ResultType>(input, weights, padding, stride, dilation);
  EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
}

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
