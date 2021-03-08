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
  Tensor<float, 4> expected{1.0f, 0.5f, 0.3333f, 0.25f};

  Tensor<float, 4> result = tosa::reciprocal(t0);

  EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected));
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
  {
    // strides = 1
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
  {
    // Strided convolution
    using InputType = Tensor4D<float, 1, 4, 4, 1>;  // N H W C
    using WeightType = Tensor4D<float, 1, 2, 2, 1>; // COUT KH KW CIN
    using ResultType = Tensor4D<float, 1, 2, 2, 1>; // N H W C
    // clang-format off
    InputType input{1,  2,  3,  4,
                    5,  6,  7,  8,
                    9,  10, 11, 12,
                    13, 14, 15, 16};
    WeightType weights{1, 2,
                       3, 4};
    ResultType expected_result{44,  64,
                              124, 144};
    // clang-format on
    Tensor1D<int64_t, 4> padding{0, 0, 0, 1}; // {pt, pb, pl, pr}
    Tensor1D<int64_t, 2> dilation{1, 1};
    Tensor1D<int64_t, 2> stride{2, 2};

    ResultType result =
        tosa::conv2d<ResultType>(input, weights, padding, stride, dilation);
    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
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

TEST(tosa, reduce_all) {
  Tensor<bool, 2, 3> t0{true, true, true, false, true, false};

  Tensor<bool, 3> expected_result0_0{false, true, false};
  Tensor<bool, 2> expected_result0_1{true, false};

  Tensor<bool, 3> result0_0 = tosa::reduce_all<Tensor<bool, 3>>(t0, 0);
  Tensor<bool, 2> result0_1 = tosa::reduce_all<Tensor<bool, 2>>(t0, 1);

  EXPECT_THAT(result0_0, Pointwise(Eq(), expected_result0_0));
  EXPECT_THAT(result0_1, Pointwise(Eq(), expected_result0_1));
}

TEST(tosa, reduce_any) {
  Tensor<bool, 2, 3> t0{true, true, false, true, false, false};
  Tensor<bool, 3> t1{false, false, false};

  Tensor<bool, 3> expected_result0_0{true, true, false};
  Tensor<bool, 2> expected_result0_1{true, true};
  Tensor<bool> expected_result1{false};

  Tensor<bool, 3> result0_0 = tosa::reduce_any<Tensor<bool, 3>>(t0, 0);
  Tensor<bool, 2> result0_1 = tosa::reduce_any<Tensor<bool, 2>>(t0, 1);
  Tensor<bool> result1 = tosa::reduce_any<Tensor<bool>>(t1, 0);

  EXPECT_THAT(result0_0, Pointwise(Eq(), expected_result0_0));
  EXPECT_THAT(result0_1, Pointwise(Eq(), expected_result0_1));
  EXPECT_THAT(result1, Pointwise(Eq(), expected_result1));
}

TEST(tosa, reduce_max) {
  Tensor<int32_t, 2, 3> t0{1, 2, 3, 4, 5, 6};

  Tensor<int32_t, 3> expected_result0_0{4, 5, 6};
  Tensor<int32_t, 2> expected_result0_1{3, 6};

  Tensor<int32_t, 3> result0_0 = tosa::reduce_max<Tensor<int32_t, 3>>(t0, 0);
  Tensor<int32_t, 2> result0_1 = tosa::reduce_max<Tensor<int32_t, 2>>(t0, 1);

  EXPECT_THAT(result0_0, Pointwise(Eq(), expected_result0_0));
  EXPECT_THAT(result0_1, Pointwise(Eq(), expected_result0_1));
}

TEST(tosa, reduce_min) {
  Tensor<int32_t, 2, 3> t0{1, 2, 3, 4, 5, 6};

  Tensor<int32_t, 3> expected_result0_0{1, 2, 3};
  Tensor<int32_t, 2> expected_result0_1{1, 4};

  Tensor<int32_t, 3> result0_0 = tosa::reduce_min<Tensor<int32_t, 3>>(t0, 0);
  Tensor<int32_t, 2> result0_1 = tosa::reduce_min<Tensor<int32_t, 2>>(t0, 1);

  EXPECT_THAT(result0_0, Pointwise(Eq(), expected_result0_0));
  EXPECT_THAT(result0_1, Pointwise(Eq(), expected_result0_1));
}

TEST(tosa, reduce_prod) {
  Tensor<int32_t, 2, 3> t0{1, 2, 3, 4, 5, 6};

  Tensor<int32_t, 3> expected_result0_0{4, 10, 18};
  Tensor<int32_t, 2> expected_result0_1{6, 120};

  Tensor<int32_t, 3> result0_0 = tosa::reduce_prod<Tensor<int32_t, 3>>(t0, 0);
  Tensor<int32_t, 2> result0_1 = tosa::reduce_prod<Tensor<int32_t, 2>>(t0, 1);

  EXPECT_THAT(result0_0, Pointwise(Eq(), expected_result0_0));
  EXPECT_THAT(result0_1, Pointwise(Eq(), expected_result0_1));
}

TEST(tosa, reduce_sum) {
  Tensor<int32_t, 2, 3> t0{1, 2, 3, 4, 5, 6};
  Tensor<int32_t, 4, 2, 3> t1{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
                              1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
  Tensor<int32_t, 3> expected_result0_0{5, 7, 9};
  Tensor<int32_t, 2> expected_result0_1{6, 15};
  Tensor<int32_t, 2, 3> expected_result1_0{4, 8, 12, 16, 20, 24};
  Tensor<int32_t, 4, 3> expected_result1_1{5, 7, 9, 5, 7, 9, 5, 7, 9, 5, 7, 9};
  Tensor<int32_t, 4, 2> expected_result1_2{6, 15, 6, 15, 6, 15, 6, 15};

  Tensor<int32_t, 3> result0_0 = tosa::reduce_sum<Tensor<int32_t, 3>>(t0, 0);
  Tensor<int32_t, 2> result0_1 = tosa::reduce_sum<Tensor<int32_t, 2>>(t0, 1);
  Tensor<int32_t, 2, 3> result1_0 =
      tosa::reduce_sum<Tensor<int32_t, 2, 3>>(t1, 0);
  Tensor<int32_t, 4, 3> result1_1 =
      tosa::reduce_sum<Tensor<int32_t, 4, 3>>(t1, 1);
  Tensor<int32_t, 4, 2> result1_2 =
      tosa::reduce_sum<Tensor<int32_t, 4, 2>>(t1, 2);

  EXPECT_THAT(result0_0, Pointwise(Eq(), expected_result0_0));
  EXPECT_THAT(result0_1, Pointwise(Eq(), expected_result0_1));
  EXPECT_THAT(result1_0, Pointwise(Eq(), expected_result1_0));
  EXPECT_THAT(result1_1, Pointwise(Eq(), expected_result1_1));
  EXPECT_THAT(result1_2, Pointwise(Eq(), expected_result1_2));
}

TEST(tosa, reshape) {
  Tensor2D<int, 1, 2> t0 = {1, 2};
  Tensor3D<int, 1, 1, 2> s0 = tosa::reshape<Tensor3D<int, 1, 1, 2>>(t0);
  EXPECT_THAT(s0, Pointwise(Eq(), t0));

  Tensor3D<int, 2, 1, 2> t1 = {1, 2, 3, 4};
  Tensor3D<int, 1, 2, 2> s1 = tosa::reshape<Tensor3D<int, 1, 2, 2>>(t1);
  EXPECT_THAT(s1, Pointwise(Eq(), t1));

  Tensor1D<int, 10> t2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  Tensor2D<int, 2, 5> s2 = tosa::reshape<Tensor2D<int, 2, 5>>(t2);
  EXPECT_THAT(s2, Pointwise(Eq(), t2));

  Tensor3D<int, 2, 2, 2> t3 = {1, 2, 3, 4, 5, 6, 7, 8};
  Tensor1D<int, 8> s3 = tosa::reshape<Tensor1D<int, 8>>(t3);
  EXPECT_THAT(s3, Pointwise(Eq(), t3));
}

} // namespace
