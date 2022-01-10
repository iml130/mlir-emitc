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

using namespace emitc;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pointwise;

const float EPSILON = 5e-4;

// Unary elementwise ops
TEST(tosa, clamp) {
  Tensor<float, 2, 1> t0{-1.5f, 5.0f};
  float min0 = -1.0f;
  float max0 = 3.0f;
  Tensor<float, 2, 1> expected_result0{-1.0, 3.0f};
  Tensor<float, 2, 1> s0 = tosa::clamp(t0, min0, max0);

  EXPECT_THAT(s0, Pointwise(FloatEq(), expected_result0));

  Tensor<int64_t, 4, 2, 1> t1{-2, 2, -2, 3, 4, -5, 5, 5};
  int64_t min1 = 1;
  int64_t max1 = 3;
  Tensor<int64_t, 4, 2, 1> expected_result1{1, 2, 1, 3, 3, 1, 3, 3};
  Tensor<int64_t, 4, 2, 1> s1 = tosa::clamp(t1, min1, max1);
  EXPECT_THAT(s1, Pointwise(Eq(), expected_result1));

  int64_t min2 = -1;
  int64_t max2 = 4;
  Tensor<int64_t, 4, 2, 1> expected_result2{-1, 2, -1, 3, 4, -1, 4, 4};
  Tensor<int64_t, 4, 2, 1> s2 = tosa::clamp(t1, min2, max2);
  EXPECT_THAT(s2, Pointwise(Eq(), expected_result2));
}

TEST(tosa, clz) {
  Tensor<int32_t, 5> t0{0, 1, 0xC000, -1, -0x7FFFFFFF};
  Tensor<int32_t, 5> expected{32, 31, 16, 0, 0};
  Tensor<int32_t, 5> result = tosa::clz(t0);
  EXPECT_THAT(result, Pointwise(Eq(), expected));
}

TEST(tosa, reciprocal) {
  Tensor<float, 4> t0{1.0f, 2.0f, 3.0f, 4.0f};
  Tensor<float, 4> expected{1.0f, 0.5f, 0.3333f, 0.25f};

  Tensor<float, 4> result = tosa::reciprocal(t0);

  EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected));
}

TEST(tosa, reluN) {
  Tensor<float, 2, 1> t0{0.0f, 5.0f};
  float max0 = 3.0f;
  Tensor<float, 2, 1> expected_result0{0.0, 3.0f};
  Tensor<float, 2, 1> s0 = tosa::reluN(t0, max0);

  EXPECT_THAT(s0, Pointwise(FloatEq(), expected_result0));

  Tensor<int64_t, 4, 2, 1> t1{-2, 2, -2, 3, 4, -5, 5, 5};
  int64_t max1 = 3;
  Tensor<int64_t, 4, 2, 1> expected_result1{0, 2, 0, 3, 3, 0, 3, 3};
  Tensor<int64_t, 4, 2, 1> s1 = tosa::reluN(t1, max1);
  EXPECT_THAT(s1, Pointwise(Eq(), expected_result1));

  int64_t max2 = 100;
  Tensor<int64_t, 4, 2, 1> expected_result2{0, 2, 0, 3, 4, 0, 5, 5};
  Tensor<int64_t, 4, 2, 1> s2 = tosa::reluN(t1, max2);
  EXPECT_THAT(s2, Pointwise(Eq(), expected_result2));
}

// Binary elementwise ops
TEST(tosa, arithmetic_right_shift) {
  {
    Tensor1D<int16_t, 5> in1{0b10, 0b10, -0b10, 0b1, -0b1};
    Tensor1D<int16_t, 5> in2{0, 1, 1, 1, 1};
    bool round = false;
    Tensor1D<int16_t, 5> expected_result{0b10, 0b1, -0b1, 0b0, -0b1};
    Tensor1D<int16_t, 5> result = tosa::arithmetic_right_shift(in1, in2, round);
    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor1D<int16_t, 4> in1{0b1, 0b1, 0b10, 0b110};
    Tensor1D<int16_t, 4> in2{0, 1, 1, 2};
    bool round = true;
    Tensor1D<int16_t, 4> expected_result{0b1, 0b1, 0b1, 0b10};
    Tensor1D<int16_t, 4> result = tosa::arithmetic_right_shift(in1, in2, round);
    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
}

TEST(tosa, logical_left_shift) {
  Tensor1D<int16_t, 4> s0{0b1, 0b1, -0b1, 0b101};
  Tensor1D<int16_t, 4> t0{0, 1, 1, 2};
  Tensor1D<int16_t, 4> expected_result{0b1, 0b10, -0b10, 0b10100};
  Tensor1D<int16_t, 4> result = tosa::logical_left_shift(s0, t0);
  EXPECT_THAT(result, Pointwise(Eq(), expected_result));
}

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

TEST(tosa, broadcastable_op) {
  // In the CallOpBroadcastableConversion ops where the tensor shape of the
  // operands don't match, a broadcast_in_dim op is inserted. This unittest
  // verifies that procedure.

  // %0 = "tosa.add"(%arg0, %arg1) : (tensor<2x1x3xf32>, tensor<1x1x3xf32>) ->
  // tensor<2x1x3xf32>
  Tensor<float, 2, 1, 3> t0_arg0{3, 3, 3, 3, 3, 3};
  Tensor<float, 1, 1, 3> t0_arg1{1, 2, 3};
  Tensor<float, 2, 1, 3> t0_arg1_broadcasted =
      emitc::broadcast_in_dim<Tensor<float, 2, 1, 3>>(t0_arg1, {0, 1, 2});
  EXPECT_THAT(t0_arg1_broadcasted, Pointwise(Eq(), {1, 2, 3, 1, 2, 3}));
  tosa::add(t0_arg0,
            t0_arg1_broadcasted); // Just make sure it compiles in this test

  // %0 = "tosa.add"(%arg0, %arg1) : (tensor<2x1x3xf32>, tensor<3xf32>) ->
  // tensor<2x1x3xf32>
  Tensor<float, 2, 1, 3> t1_arg0{4, 4, 4, 4, 4, 4};
  Tensor<float, 3> t1_arg1{1, 2, 3};
  Tensor<float, 2, 1, 3> t1_arg1_broadcasted =
      emitc::broadcast_in_dim<Tensor<float, 2, 1, 3>>(t1_arg1, {2});
  EXPECT_THAT(t1_arg1_broadcasted, Pointwise(Eq(), {1, 2, 3, 1, 2, 3}));
  tosa::add(t1_arg0,
            t1_arg1_broadcasted); // Just make sure it compiles in this test
}

// Other ops
TEST(tosa, depthwise_conv2d) {
  {
    // test for channel_multiplier=1
    using InputType = Tensor4D<float, 1, 4, 5, 2>;  // N H W C
    using WeightType = Tensor4D<float, 2, 2, 2, 1>; // KH KW CIN M
    using ResultType = Tensor4D<float, 1, 3, 4, 2>; // N H W CXM
    InputType input{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
    WeightType weights{1, 2, 3, 4, 5, 6, 7, 8};
    ResultType expected_result{156, 204, 188, 244, 220, 284, 252, 324,
                               316, 404, 348, 444, 380, 484, 412, 524,
                               476, 604, 508, 644, 540, 684, 572, 724};

    Tensor1D<int64_t, 4> padding{0, 0, 0, 0}; // {pt, pb, pl, pr}
    Tensor1D<int64_t, 2> dilation{1, 1};
    Tensor1D<int64_t, 2> stride{1, 1};

    ResultType result = tosa::depthwise_conv2d<ResultType>(
        input, weights, padding, stride, dilation);
    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    // test for channel_multiplier=2
    using InputType = Tensor4D<float, 1, 4, 5, 2>;  // N H W C
    using WeightType = Tensor4D<float, 2, 2, 2, 2>; // KH KW CIN M
    using ResultType = Tensor4D<float, 1, 3, 4, 4>; // N H W CXM
    InputType input{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
    WeightType weights{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    ResultType expected_result{
        284, 312,  376,  408,  340, 376,  448,  488,  396,  440,  520,  568,
        452, 504,  592,  648,  564, 632,  736,  808,  620,  696,  808,  888,
        676, 760,  880,  968,  732, 824,  952,  1048, 844,  952,  1096, 1208,
        900, 1016, 1168, 1288, 956, 1080, 1240, 1368, 1012, 1144, 1312, 1448};

    Tensor1D<int64_t, 4> padding{0, 0, 0, 0}; // {pt, pb, pl, pr}
    Tensor1D<int64_t, 2> dilation{1, 1};
    Tensor1D<int64_t, 2> stride{1, 1};

    ResultType result = tosa::depthwise_conv2d<ResultType>(
        input, weights, padding, stride, dilation);
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
    using AType = Tensor3D<float, 1, 3, 1>; // M K
    using BType = Tensor3D<float, 1, 1, 2>; // K N
    using CType = Tensor3D<float, 1, 3, 2>; // M N
    AType a{1, 2, 3};
    BType b{1, 2};
    CType c = tosa::matmul(a, b);

    CType expected_result{1, 2, 2, 4, 3, 6};
    EXPECT_THAT(c, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    using AType = Tensor3D<float, 1, 3, 2>; // M K
    using BType = Tensor3D<float, 1, 2, 2>; // K N
    using CType = Tensor3D<float, 1, 3, 2>; // M N
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

TEST(tosa, slice) {
  // Slice Tensor1D
  Tensor1D<float, 5> s1{0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  auto t1 = tosa::slice<Tensor1D<float, 2>>(s1, {2}, {2});
  EXPECT_THAT(t1, Pointwise(FloatEq(), {2.0f, 3.0f}));

  // Slice Tensor2D
  Tensor2D<float, 4, 3> s2{0.0f, 1.0f, 2.0f, 3.0f, 4.0f,  5.0f,
                           6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  auto t2 = tosa::slice<Tensor2D<float, 2, 2>>(s2, {2, 1}, {2, 2});

  EXPECT_THAT(t2, Pointwise(FloatEq(), {7.0f, 8.0f, 10.0f, 11.0f}));

  // Slice Tensor3D
  Tensor3D<float, 4, 3, 2> s3{0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
                              6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
                              12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
                              18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f};
  auto t3 = tosa::slice<Tensor3D<float, 2, 2, 2>>(s3, {2, 1, 0}, {2, 2, 2});
  EXPECT_THAT(t3, Pointwise(FloatEq(), {14.0f, 15.0f, 16.0f, 17.0f, 20.0f,
                                        21.0f, 22.0f, 23.0f}));

  // Slice Tensor4D
  Tensor4D<float, 4, 3, 1, 2> s4{0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
                                 6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
                                 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
                                 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f};
  auto t4 =
      tosa::slice<Tensor4D<float, 2, 2, 1, 2>>(s4, {2, 1, 0, 0}, {2, 2, 1, 2});
  EXPECT_THAT(t4, Pointwise(FloatEq(), {14.0f, 15.0f, 16.0f, 17.0f, 20.0f,
                                        21.0f, 22.0f, 23.0f}));

  auto t4_2 =
      tosa::slice<Tensor4D<float, 4, 3, 1, 2>>(s4, {0, 0, 0, 0}, {4, 3, 1, 2});
  EXPECT_THAT(t4_2, Pointwise(FloatEq(), s4));
}

TEST(tosa, pad) {
  // clang-format off
  Tensor<int32_t, 2, 3> operand0{1, 2, 3,
                                 4, 5, 6};
  Tensor<int32_t, 2, 2, 3> operand1{1, 2, 3,  4,  5,  6,
                                    7, 8, 9, 10, 11, 12};

  Tensor<int32_t, 2, 2> padding0{0, 1,
                                 1, 2};
  Tensor<int32_t, 3, 2> padding1_0{0, 0,
                                   0, 0,
                                   0, 0};
  Tensor<int32_t, 3, 2> padding1_1{1, 1,
                                   1, 1,
                                   1, 1};
  Tensor<int32_t, 3, 2> padding1_2{1, 0,
                                   0, 1,
                                   1, 0};

  Tensor<int32_t, 3, 6> expected_result0{0, 1, 2, 3, 0, 0,
                                         0, 4, 5, 6, 0, 0,
                                         0, 0, 0, 0, 0, 0};
  Tensor<int32_t, 2, 2, 3> expected_result1_0{1, 2, 3,  4,  5,  6,
                                              7, 8, 9, 10, 11, 12};
  Tensor<int32_t, 4, 4, 5> expected_result1_1{
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0,  4,  5,  6,  0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 7, 8, 9, 0, 0, 10, 11, 12,  0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0, 0, 0, 0, 0, 0};
  Tensor<int32_t, 3, 3, 4> expected_result1_2{
      0, 0, 0, 0, 0,  0,  0,  0, 0, 0, 0, 0,
      0, 1, 2, 3, 0,  4,  5,  6, 0, 0, 0, 0,
      0, 7, 8, 9, 0, 10, 11, 12, 0, 0, 0, 0};

  // clang-format on
  Tensor<int32_t, 3, 6> result0 =
      tosa::pad<Tensor<int32_t, 3, 6>>(operand0, padding0);
  Tensor<int32_t, 2, 2, 3> result1_0 =
      tosa::pad<Tensor<int32_t, 2, 2, 3>>(operand1, padding1_0);
  Tensor<int32_t, 4, 4, 5> result1_1 =
      tosa::pad<Tensor<int32_t, 4, 4, 5>>(operand1, padding1_1);
  Tensor<int32_t, 3, 3, 4> result1_2 =
      tosa::pad<Tensor<int32_t, 3, 3, 4>>(operand1, padding1_2);

  EXPECT_THAT(result0, Pointwise(Eq(), expected_result0));
  EXPECT_THAT(result1_0, Pointwise(Eq(), expected_result1_0));
  EXPECT_THAT(result1_1, Pointwise(Eq(), expected_result1_1));
  EXPECT_THAT(result1_2, Pointwise(Eq(), expected_result1_2));
}

TEST(tosa, transpose) {
  // clang-format off
  Tensor2D<float, 3, 2> t0 = {1, 2,
                              3, 4,
                              5, 6};
  Tensor1D<int32_t, 2> perms_i32 = {1, 0};
  Tensor1D<int64_t, 2> perms_i64 = {1, 0};
  Tensor1D<int64_t, 2> no_perms =  {0, 1};
  Tensor2D<float, 2, 3> expected_result0 = {1, 3, 5,
                                            2, 4, 6};
  // clang-format on
  Tensor2D<float, 2, 3> s0 =
      tosa::transpose<Tensor2D<float, 2, 3>>(t0, perms_i32);
  Tensor2D<float, 2, 3> s0_2 =
      tosa::transpose<Tensor2D<float, 2, 3>>(t0, perms_i64);
  Tensor2D<float, 3, 2> s0_3 =
      tosa::transpose<Tensor2D<float, 3, 2>>(t0, no_perms);
  EXPECT_THAT(s0, Pointwise(Eq(), expected_result0));
  EXPECT_THAT(s0_2, Pointwise(Eq(), expected_result0));
  EXPECT_THAT(s0_3, Pointwise(Eq(), t0));

  // clang-format off
  Tensor3D<float, 1, 3, 2> t1 = {1, 2,
                                 3, 4,
                                 5, 6};
  Tensor1D<int32_t, 3> perms1 = {2, 0, 1};
  Tensor3D<float, 2, 1, 3> expected_result1 = {1, 3, 5,
                                               2, 4, 6};
  // clang-format on
  Tensor3D<float, 2, 1, 3> s1 =
      tosa::transpose<Tensor3D<float, 2, 1, 3>>(t1, perms1);
  EXPECT_THAT(s1, Pointwise(Eq(), expected_result1));

  // clang-format off
  Tensor3D<float, 2, 3, 4> t2 = {1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
                                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  Tensor1D<int32_t, 3> perms2 = {2, 0, 1};
  Tensor3D<float, 4, 2, 3> expected_result2 = {1, 5,  9, 13, 17, 21,
                                               2, 6, 10, 14, 18, 22,
                                               3, 7, 11, 15, 19, 23,
                                               4, 8, 12, 16, 20, 24};
  // clang-format on
  Tensor3D<float, 4, 2, 3> s2 =
      tosa::transpose<Tensor3D<float, 4, 2, 3>>(t2, perms2);
  EXPECT_THAT(s2, Pointwise(Eq(), expected_result2));
}

} // namespace
