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

TEST(tosa, matmul) {
  {
    using AType = Tensor2D<float, 3, 1>; // M K
    using BType = Tensor2D<float, 1, 2>; // K N
    using CType = Tensor2D<float, 3, 2>; // M N
    AType a{1, 2, 3};
    BType b{1, 2};
    CType c = tosa::matmul(a, b);

    CType expected_result{1, 2, 2, 4, 3, 6};
    EXPECT_THAT(c, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    using AType = Tensor2D<float, 3, 2>; // M K
    using BType = Tensor2D<float, 2, 2>; // K N
    using CType = Tensor2D<float, 3, 2>; // M N
    AType a{1, 2, 3, 4, 5, 6};
    BType b{7, 8, 9, 10};
    CType c = tosa::matmul(a, b);

    CType expected_result{25, 28, 57, 64, 89, 100};
    EXPECT_THAT(c, Pointwise(FloatNear(EPSILON), expected_result));
  }
}

} // namespace
