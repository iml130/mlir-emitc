// Copyright Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <tuple>

#include "gmock/gmock.h"

#include "emitc/stablehlo.h"
#include "emitc/types.h"

namespace {

using namespace emitc;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pointwise;

const float EPSILON = 5e-4;

// Unary elementwise ops

TEST(stablehlo, abs) {
  EXPECT_EQ(1, stablehlo::abs(-1));

  {
    Tensor0D<int> x{-1};
    Tensor0D<int> expected_result{1};
    Tensor0D<int> result = stablehlo::abs(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor1D<float, 2> x{-1.0f, -2.0f};
    Tensor1D<float, 2> expected_result{1.0f, 2.0f};
    Tensor1D<float, 2> result = stablehlo::abs(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor2D<long, 2, 2> x{-2, -1, 0, 2};
    Tensor2D<long, 2, 2> expected_result{2, 1, 0, 2};
    Tensor2D<long, 2, 2> result = stablehlo::abs(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor3D<int32_t, 2, 1, 2> x{-2, -1, 0, 2};
    Tensor3D<int32_t, 2, 1, 2> expected_result{2, 1, 0, 2};
    Tensor3D<int32_t, 2, 1, 2> result = stablehlo::abs(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor4D<int, 2, 2, 1, 2> x{-2, -1, 0, 0, 3, -3, -2, 1};
    Tensor4D<int, 2, 2, 1, 2> expected_result{2, 1, 0, 0, 3, 3, 2, 1};
    Tensor4D<int, 2, 2, 1, 2> result = stablehlo::abs(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  // TODO:: Test complex to real.
}

TEST(stablehlo, ceil) {
  EXPECT_EQ(1.0, stablehlo::ceil(0.7));

  {
    Tensor0D<float> x{0.7f};
    Tensor0D<float> expected_result{1.0f};
    Tensor0D<float> result = stablehlo::ceil(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor1D<float, 2> x{1.6f, 2.0f};
    Tensor1D<float, 2> expected_result{2.0f, 2.0f};
    Tensor1D<float, 2> result = stablehlo::ceil(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor2D<double, 2, 2> x{2.1, 1.6, 0.0, 2.0};
    Tensor2D<double, 2, 2> expected_result{3.0, 2.0, 0.0, 2.0};
    Tensor2D<double, 2, 2> result = stablehlo::ceil(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor3D<double, 2, 1, 2> x{2.1, 1.6, 0.0, 2.0};
    Tensor3D<double, 2, 1, 2> expected_result{3.0, 2.0, 0.0, 2.0};
    Tensor3D<double, 2, 1, 2> result = stablehlo::ceil(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor4D<double, 1, 2, 1, 2> x{2.1, 1.6, 0.0, 2.0};
    Tensor4D<double, 1, 2, 1, 2> expected_result{3.0, 2.0, 0.0, 2.0};
    Tensor4D<double, 1, 2, 1, 2> result = stablehlo::ceil(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
}

TEST(stablehlo, convert) {
  {
    uint32_t x = 1;
    uint64_t expected_result = 1;
    uint64_t result = stablehlo::convert<uint64_t>(x);

    EXPECT_EQ(result, expected_result);
  }
  {
    Tensor0D<uint32_t> x{1};
    Tensor0D<size_t> expected_result{1};
    Tensor0D<size_t> result = stablehlo::convert<Tensor0D<size_t>>(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor1D<uint16_t, 2> x{1, 3};
    Tensor1D<size_t, 2> expected_result{1, 3};
    Tensor1D<size_t, 2> result = stablehlo::convert<Tensor1D<size_t, 2>>(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor2D<float, 2, 2> x{1.0f, 2.0f, 4.0f, 8.0f};
    Tensor2D<double, 2, 2> expected_result{1.0, 2.0, 4.0, 8.0};
    Tensor2D<double, 2, 2> result =
        stablehlo::convert<Tensor2D<double, 2, 2>>(x);

    EXPECT_THAT(result, Pointwise(DoubleEq(), expected_result));
  }
  {
    Tensor3D<float, 2, 1, 2> x{1.0f, 2.0f, 4.0f, 8.0f};
    Tensor3D<double, 2, 1, 2> expected_result{1.0, 2.0, 4.0, 8.0};
    Tensor3D<double, 2, 1, 2> result =
        stablehlo::convert<Tensor3D<double, 2, 1, 2>>(x);

    EXPECT_THAT(result, Pointwise(DoubleEq(), expected_result));
  }
  {
    Tensor4D<double, 2, 1, 2, 2> x{1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0};
    Tensor4D<float, 2, 1, 2, 2> expected_result{1.0f,  2.0f,  4.0f,  8.0f,
                                                16.0f, 32.0f, 64.0f, 128.0f};
    Tensor4D<float, 2, 1, 2, 2> result =
        stablehlo::convert<Tensor4D<float, 2, 1, 2, 2>>(x);

    EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));
  }
}

TEST(stablehlo, cos) {
  EXPECT_NEAR(1.0f, stablehlo::cos(0.0f), EPSILON);

  {
    Tensor0D<float> x{M_PIf32};
    Tensor0D<float> expected_result{-1.0f};
    Tensor0D<float> result = stablehlo::cos(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor1D<float, 2> x{M_PI_2f32, -M_PI_2f32};
    Tensor1D<float, 2> expected_result{0.0f, 0.0f};
    Tensor1D<float, 2> result = stablehlo::cos(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor2D<double, 2, 2> x{2 * M_PIf32, 0.0f, -0.5f, 0.5f};
    Tensor2D<double, 2, 2> expected_result{1.0f, 1.0f, 0.8775826f, 0.8775826f};
    Tensor2D<double, 2, 2> result = stablehlo::cos(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor3D<double, 2, 2, 1> x{2 * M_PIf32, 0.0f, -0.5f, 0.5f};
    Tensor3D<double, 2, 2, 1> expected_result{1.0f, 1.0f, 0.8775826f,
                                              0.8775826f};
    Tensor3D<double, 2, 2, 1> result = stablehlo::cos(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor4D<double, 1, 2, 1, 2> x{0.5f, 0.0f, -0.5f, 2 * M_PIf32};
    Tensor4D<double, 1, 2, 1, 2> expected_result{0.8775826f, 1.0f, 0.8775826f,
                                                 1.0f};
    Tensor4D<double, 1, 2, 1, 2> result = stablehlo::cos(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
}

TEST(stablehlo, exponential) {
  EXPECT_NEAR(M_Ef32, stablehlo::exponential(1.0f), EPSILON);

  {
    Tensor0D<float> x{0.0f};
    Tensor0D<float> expected_result{1.0f};
    Tensor0D<float> result = stablehlo::exponential(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor1D<float, 2> x{M_LN2f32, M_LN10f32};
    Tensor1D<float, 2> expected_result{2.0f, 10.0f};
    Tensor1D<float, 2> result = stablehlo::exponential(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor2D<double, 2, 2> x{1.0f, 2.0f, 3.0f, -1.0f};
    Tensor2D<double, 2, 2> expected_result{2.718281f, 7.389056f, 20.085536f,
                                           0.367879f};
    Tensor2D<double, 2, 2> result = stablehlo::exponential(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor3D<double, 2, 2, 2> x{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    Tensor3D<double, 2, 2, 2> expected_result{
        1.0f,       2.718281f,   7.389056f,   20.085536f,
        54.598150f, 148.413159f, 403.428793f, 1096.633158f};
    Tensor3D<double, 2, 2, 2> result = stablehlo::exponential(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor4D<double, 1, 2, 1, 2> x{-1.0f, -2.0f, -3.0f, 4.0f};
    Tensor4D<double, 1, 2, 1, 2> expected_result{0.367879f, 0.135335f,
                                                 0.049787f, 54.598150f};
    Tensor4D<double, 1, 2, 1, 2> result = stablehlo::exponential(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
}

TEST(stablehlo, exponential_minus_one) {
  EXPECT_NEAR(M_Ef32 - 1, stablehlo::exponential_minus_one(1.0f), EPSILON);

  {
    Tensor0D<float> x{0.0f};
    Tensor0D<float> expected_result{0.0f};
    Tensor0D<float> result = stablehlo::exponential_minus_one(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor1D<float, 2> x{M_LN2f32, M_LN10f32};
    Tensor1D<float, 2> expected_result{1.0f, 9.0f};
    Tensor1D<float, 2> result = stablehlo::exponential_minus_one(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor2D<double, 2, 2> x{-1.0f, 2.0f, 3.0f, 4.0f};
    Tensor2D<double, 2, 2> expected_result{-0.632120f, 6.389056f, 19.085536f,
                                           53.598150f};
    Tensor2D<double, 2, 2> result = stablehlo::exponential_minus_one(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor3D<double, 2, 2, 2> x{0.0f, 1.0f, 2.0f, 3.0f,
                                4.0f, 5.0f, 6.0f, -7.0f};
    Tensor3D<double, 2, 2, 2> expected_result{
        0.0f,       1.718281f,   6.389056f,   19.085536f,
        53.598150f, 147.413159f, 402.428793f, -0.9990881f};
    Tensor3D<double, 2, 2, 2> result = stablehlo::exponential_minus_one(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor4D<double, 1, 2, 1, 2> x{1.0f, 2.0f, 3.0f, -4.0f};
    Tensor4D<double, 1, 2, 1, 2> expected_result{1.718281f, 6.389056f,
                                                 19.085536f, -0.981684f};
    Tensor4D<double, 1, 2, 1, 2> result = stablehlo::exponential_minus_one(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
}

TEST(stablehlo, floor) {
  EXPECT_EQ(0.0, stablehlo::floor(0.7));

  {
    Tensor0D<float> x{0.7f};
    Tensor0D<float> expected_result{0.0f};
    Tensor0D<float> result = stablehlo::floor(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor1D<float, 2> x{1.6f, 2.0f};
    Tensor1D<float, 2> expected_result{1.0f, 2.0f};
    Tensor1D<float, 2> result = stablehlo::floor(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor2D<double, 2, 2> x{2.1, 1.6, 0.0, 2.0};
    Tensor2D<double, 2, 2> expected_result{2.0, 1.0, 0.0, 2.0};
    Tensor2D<double, 2, 2> result = stablehlo::floor(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor3D<double, 2, 2, 2> x{2.1, 1.6, 0.0, -2.0, 3.2, 5.1, 6.9, 3.14};
    Tensor3D<double, 2, 2, 2> expected_result{2.0, 1.0, 0.0, -2.0,
                                              3.0, 5.0, 6.0, 3.0};
    Tensor3D<double, 2, 2, 2> result = stablehlo::floor(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor4D<double, 1, 2, 2, 2> x{2.1, 1.6, 0.0, 2.0, -3.2, 5.1, -6.9, 3.14};
    Tensor4D<double, 1, 2, 2, 2> expected_result{2.0,  1.0, 0.0,  2.0,
                                                 -4.0, 5.0, -7.0, 3.0};
    Tensor4D<double, 1, 2, 2, 2> result = stablehlo::floor(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
}

TEST(stablehlo, is_finite) {
  EXPECT_EQ(true, stablehlo::is_finite(0.0f));

  {
    Tensor0D<float> x{M_PIf32};
    Tensor0D<bool> expected_result{true};
    Tensor0D<bool> result = stablehlo::is_finite(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor1D<float, 2> x{M_PI_2f32, INFINITY};
    Tensor1D<bool, 2> expected_result{true, false};
    Tensor1D<bool, 2> result = stablehlo::is_finite(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor2D<float, 2, 2> x{INFINITY, -INFINITY, NAN, -0.0f};
    Tensor2D<bool, 2, 2> expected_result{false, false, false, true};
    Tensor2D<bool, 2, 2> result = stablehlo::is_finite(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor3D<float, 2, 2, 1> x{INFINITY, -INFINITY, NAN, -0.0f};
    Tensor3D<bool, 2, 2, 1> expected_result{false, false, false, true};
    Tensor3D<bool, 2, 2, 1> result = stablehlo::is_finite(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor4D<float, 2, 2, 1, 1> x{INFINITY, -INFINITY, NAN, -0.0f};
    Tensor4D<bool, 2, 2, 1, 1> expected_result{false, false, false, true};
    Tensor4D<bool, 2, 2, 1, 1> result = stablehlo::is_finite(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
}

TEST(stablehlo, log) {
  EXPECT_NEAR(0.0f, stablehlo::log(1.0f), EPSILON);

  {
    Tensor0D<float> x{M_Ef32};
    Tensor0D<float> expected_result{1.0f};
    Tensor0D<float> result = stablehlo::log(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor1D<float, 2> x{M_Ef32 * M_Ef32, M_Ef32 * M_Ef32 * M_Ef32};
    Tensor1D<float, 2> expected_result{2.0f, 3.0f};
    Tensor1D<float, 2> result = stablehlo::log(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor2D<float, 2, 2> x{1.0f, 2.0f, 3.0f, 4.0f};
    Tensor2D<float, 2, 2> expected_result{0.0f, 0.693147f, 1.098612f,
                                          1.386294f};
    Tensor2D<float, 2, 2> result = stablehlo::log(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor3D<float, 2, 2, 2> x{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    Tensor3D<float, 2, 2, 2> expected_result{0.0f,      0.693147f, 1.098612f,
                                             1.386294f, 1.609437f, 1.791759f,
                                             1.945910f, 2.079441f};
    Tensor3D<float, 2, 2, 2> result = stablehlo::log(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor4D<float, 2, 2, 2, 1> x{1.0f, 2.0f, 3.0f, 4.0f,
                                  5.0f, 6.0f, 7.0f, 8.0f};
    Tensor4D<float, 2, 2, 2, 1> expected_result{0.0f,      0.693147f, 1.098612f,
                                                1.386294f, 1.609437f, 1.791759f,
                                                1.945910f, 2.079441f};
    Tensor4D<float, 2, 2, 2, 1> result = stablehlo::log(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
}

TEST(stablehlo, log_plus_one) {
  EXPECT_NEAR(0.693147f, stablehlo::log_plus_one(1.0f), EPSILON);

  {
    Tensor0D<float> x{M_Ef32 - 1};
    Tensor0D<float> expected_result{1.0f};
    Tensor0D<float> result = stablehlo::log_plus_one(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor1D<float, 2> x{M_Ef32 * M_Ef32, M_Ef32 * M_Ef32 * M_Ef32};
    Tensor1D<float, 2> expected_result{2.126928f, 3.048587f};
    Tensor1D<float, 2> result = stablehlo::log_plus_one(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor2D<double, 2, 2> x{0.0, 1.0, 2.0, 3.0};
    Tensor2D<double, 2, 2> expected_result{0.0, 0.693147, 1.098612, 1.386294};
    Tensor2D<double, 2, 2> result = stablehlo::log_plus_one(x);

    EXPECT_THAT(result, Pointwise(DoubleNear(EPSILON), expected_result));
  }
  {
    Tensor3D<double, 2, 2, 1> x{0.0, 1.0, 2.0, 3.0};
    Tensor3D<double, 2, 2, 1> expected_result{0.0, 0.693147, 1.098612,
                                              1.386294};
    Tensor3D<double, 2, 2, 1> result = stablehlo::log_plus_one(x);

    EXPECT_THAT(result, Pointwise(DoubleNear(EPSILON), expected_result));
  }
  {
    Tensor4D<double, 2, 2, 1, 2> x{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    Tensor4D<double, 2, 2, 1, 2> expected_result{0.0,      0.693147, 1.098612,
                                                 1.386294, 1.609437, 1.791759,
                                                 1.945910, 2.079441};
    Tensor4D<double, 2, 2, 1, 2> result = stablehlo::log_plus_one(x);

    EXPECT_THAT(result, Pointwise(DoubleNear(EPSILON), expected_result));
  }
}

TEST(stablehlo, negate) {
  {
    int x = -1;
    int expected_result = 1;
    int result = stablehlo::negate(x);

    EXPECT_EQ(expected_result, result);
  }
  {
    Tensor0D<int> x{3};
    Tensor0D<int> expected_result{-3};
    Tensor0D<int> result = stablehlo::negate(x);

    EXPECT_THAT(expected_result, Pointwise(Eq(), result));
  }
  {
    Tensor1D<float, 2> x{-1.3f, 2.4f};
    Tensor1D<float, 2> expected_result{1.3f, -2.4f};
    Tensor1D<float, 2> result = stablehlo::negate(x);

    EXPECT_THAT(expected_result, Pointwise(FloatNear(EPSILON), result));
  }
  {
    Tensor2D<long, 2, 2> x{3, 1, -4, 0};
    Tensor2D<long, 2, 2> expected_result{-3, -1, 4, 0};
    Tensor2D<long, 2, 2> result = stablehlo::negate(x);

    EXPECT_THAT(expected_result, Pointwise(Eq(), result));
  }
  {
    Tensor3D<double, 2, 1, 1> x{3.1415, -2.7183};
    Tensor3D<double, 2, 1, 1> expected_result{-3.1415, 2.7183};
    Tensor3D<double, 2, 1, 1> result = stablehlo::negate(x);

    EXPECT_THAT(expected_result, Pointwise(FloatNear(EPSILON), result));
  }
  {
    Tensor4D<int64_t, 1, 2, 1, 2> x{9223372036854775807, -4,
                                    -9223372036854775807, 4};
    Tensor4D<int64_t, 1, 2, 1, 2> expected_result{-9223372036854775807, 4,
                                                  9223372036854775807, -4};
    Tensor4D<int64_t, 1, 2, 1, 2> result = stablehlo::negate(x);

    EXPECT_THAT(expected_result, Pointwise(Eq(), result));
  }
}

TEST(stablehlo, round) {
  EXPECT_EQ(1.0, stablehlo::round(0.7));
  EXPECT_EQ(0.0, stablehlo::round(0.4));

  {
    Tensor0D<float> x{0.7f};
    Tensor0D<float> expected_result{1.0f};
    Tensor0D<float> result = stablehlo::round(x);

    EXPECT_THAT(expected_result, Pointwise(Eq(), result));
  }
  {
    Tensor1D<float, 3> x{1.4f, -1.6f, 2.0f};
    Tensor1D<float, 3> expected_result{1.0f, -2.0f, 2.0f};
    Tensor1D<float, 3> result = stablehlo::round(x);

    EXPECT_THAT(expected_result, Pointwise(Eq(), result));
  }
  {
    Tensor2D<double, 2, 2> x{-2.1, 1.6, 0.0, 2.0};
    Tensor2D<double, 2, 2> expected_result{-2.0, 2.0, 0.0, 2.0};
    Tensor2D<double, 2, 2> result = stablehlo::round(x);

    EXPECT_THAT(expected_result, Pointwise(Eq(), result));
  }
  {
    Tensor3D<float, 2, 1, 2> x{2.1f, 1.6f, 0.0f, 2.0f};
    Tensor3D<float, 2, 1, 2> expected_result{2.0f, 2.0f, 0.0f, 2.0f};
    Tensor3D<float, 2, 1, 2> result = stablehlo::round(x);

    EXPECT_THAT(expected_result, Pointwise(Eq(), result));
  }
  {
    Tensor4D<double, 2, 2, 1, 1> x{2.1, -3.2, 0.0, 2.0};
    Tensor4D<double, 2, 2, 1, 1> expected_result{2.0, -3.0, 0.0, 2.0};
    Tensor4D<double, 2, 2, 1, 1> result = stablehlo::round(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
}

TEST(stablehlo, sin) {
  EXPECT_NEAR(0.0f, stablehlo::sin(0.0f), EPSILON);

  {
    Tensor0D<float> x{M_PIf32};
    Tensor0D<float> expected_result{0.0f};
    Tensor0D<float> result = stablehlo::sin(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor1D<float, 2> x{M_PI_2f32, -M_PI_2f32};
    Tensor1D<float, 2> expected_result{1.0f, -1.0f};
    Tensor1D<float, 2> result = stablehlo::sin(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor2D<double, 2, 2> x{2 * M_PIf32, 0.0, -0.5, 0.5};
    Tensor2D<double, 2, 2> expected_result{0.0, 0.0, -0.479426, 0.479426};
    Tensor2D<double, 2, 2> result = stablehlo::sin(x);

    EXPECT_THAT(result, Pointwise(DoubleNear(EPSILON), expected_result));
  }
  {
    Tensor3D<double, 1, 2, 2> x{2 * M_PIf32, 0.0, -0.5, 0.5};
    Tensor3D<double, 1, 2, 2> expected_result{0.0, 0.0, -0.479426, 0.479426};
    Tensor3D<double, 1, 2, 2> result = stablehlo::sin(x);

    EXPECT_THAT(result, Pointwise(DoubleNear(EPSILON), expected_result));
  }
  {
    Tensor4D<double, 2, 1, 2, 1> x{2 * M_PIf32, 0.0, -0.5, 0.5};
    Tensor4D<double, 2, 1, 2, 1> expected_result{0.0, 0.0, -0.479426, 0.479426};
    Tensor4D<double, 2, 1, 2, 1> result = stablehlo::sin(x);

    EXPECT_THAT(result, Pointwise(DoubleNear(EPSILON), expected_result));
  }
}

TEST(stablehlo, sqrt) {
  EXPECT_NEAR(3.0f, stablehlo::sqrt(9.0f), EPSILON);

  {
    Tensor0D<float> x{4.0f};
    Tensor0D<float> expected_result{2.0f};
    Tensor0D<float> result = stablehlo::sqrt(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor1D<float, 2> x{0.0f, 81.0f};
    Tensor1D<float, 2> expected_result{0.0f, 9.0f};
    Tensor1D<float, 2> result = stablehlo::sqrt(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor2D<double, 2, 2> x{2.0, 3.0, 10.0, 1.0};
    Tensor2D<double, 2, 2> expected_result{1.414213, 1.732050, 3.162277, 1.0};
    Tensor2D<double, 2, 2> result = stablehlo::sqrt(x);

    EXPECT_THAT(result, Pointwise(DoubleNear(EPSILON), expected_result));
  }
  {
    Tensor3D<double, 2, 1, 2> x{2.0, 3.0, 10.0, 1.0};
    Tensor3D<double, 2, 1, 2> expected_result{1.414213, 1.732050, 3.162277,
                                              1.0};
    Tensor3D<double, 2, 1, 2> result = stablehlo::sqrt(x);

    EXPECT_THAT(result, Pointwise(DoubleNear(EPSILON), expected_result));
  }
  {
    Tensor4D<double, 2, 2, 1, 2> x{2.0, 3.0, 10.0, 1.0, 18.0, 9.0, 5.0, 25.0};
    Tensor4D<double, 2, 2, 1, 2> expected_result{
        1.414213, 1.732050, 3.162277, 1.0, 4.242640, 3.0, 2.236067, 5.0};
    Tensor4D<double, 2, 2, 1, 2> result = stablehlo::sqrt(x);

    EXPECT_THAT(result, Pointwise(DoubleNear(EPSILON), expected_result));
  }
}

TEST(stablehlo, tanh) {
  EXPECT_NEAR(0.0f, stablehlo::tanh(0.0f), EPSILON);

  {
    Tensor0D<float> x{0.0f};
    Tensor0D<float> expected_result{0.0f};
    Tensor0D<float> result = stablehlo::tanh(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor1D<float, 2> x{0.0f, 1.0f};
    Tensor1D<float, 2> expected_result{0.0f, 0.761594f};
    Tensor1D<float, 2> result = stablehlo::tanh(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor2D<double, 2, 2> x{0.0, 1.0, -1.0, 0.0};
    Tensor2D<double, 2, 2> expected_result{0.0, 0.761594, -0.761594, 0.0};
    Tensor2D<double, 2, 2> result = stablehlo::tanh(x);

    EXPECT_THAT(result, Pointwise(DoubleNear(EPSILON), expected_result));
  }
  {
    Tensor3D<float, 1, 2, 2> x{0.0f, 1.0f, -1.0f, 0.0f};
    Tensor3D<float, 1, 2, 2> expected_result{0.0f, 0.761594f, -0.761594f, 0.0f};
    Tensor3D<float, 1, 2, 2> result = stablehlo::tanh(x);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    Tensor4D<double, 3, 1, 1, 2> x{0.0, 1.0, -1.0, 0.0, M_PIf64, -M_2_PIf64};
    Tensor4D<double, 3, 1, 1, 2> expected_result{0.0, 0.761594, -0.761594,
                                                 0.0, 0.996272, -0.562593};
    Tensor4D<double, 3, 1, 1, 2> result = stablehlo::tanh(x);

    EXPECT_THAT(result, Pointwise(DoubleNear(EPSILON), expected_result));
  }
}

// Binary elementwise ops

TEST(stablehlo, add) {
  EXPECT_EQ(2, stablehlo::add(-1, 3));

  Tensor0D<int> s0{-3};
  Tensor0D<int> t0{8};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return stablehlo::add<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {5}));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};
  Tensor1D<float, 2> t1{0.2f, -3.7f};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<float, 2> {
    return stablehlo::add<Tensor1D<float, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(FloatEq(), {-1.1f, -1.3f}));

  Tensor2D<long, 2, 2> s2{3, 1, 4, 9};
  Tensor2D<long, 2, 2> t2{-2, 8, 6, -10};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return stablehlo::add<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {1, 9, 10, -1}));

  Tensor3D<int16_t, 2, 1, 2> s3{3, 1, 4, 9};
  Tensor3D<int16_t, 2, 1, 2> t3{-2, 8, 6, -10};

  auto lambda_3d = [&s3, &t3]() -> Tensor3D<int16_t, 2, 1, 2> {
    return stablehlo::add<Tensor3D<int16_t, 2, 1, 2>>(s3, t3);
  };

  EXPECT_THAT(lambda_3d(), Pointwise(Eq(), {1, 9, 10, -1}));

  Tensor4D<int8_t, 2, 1, 2, 2> s4{3, 1, 4, 9, 10, 11, 0, 1};
  Tensor4D<int8_t, 2, 1, 2, 2> t4{-2, 8, 6, -10, -10, 11, 1, 1};

  auto lambda_4d = [&s4, &t4]() -> Tensor4D<int8_t, 2, 1, 2, 2> {
    return stablehlo::add<Tensor4D<int8_t, 2, 1, 2, 2>>(s4, t4);
  };

  EXPECT_THAT(lambda_4d(), Pointwise(Eq(), {1, 9, 10, -1, 0, 22, 1, 2}));
}

TEST(stablehlo, atan2) {
  EXPECT_NEAR(0.321751f, stablehlo::atan2(1.0f, 3.0f), EPSILON);

  Tensor0D<float> s0{1.0f};
  Tensor0D<float> t0{3.0f};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<float> {
    return stablehlo::atan2<Tensor0D<float>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(FloatNear(EPSILON), {0.321751f}));

  Tensor1D<float, 2> s1{1.0f, 0.5f};
  Tensor1D<float, 2> t1{3.0f, -0.5f};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<float, 2> {
    return stablehlo::atan2<Tensor1D<float, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(),
              Pointwise(FloatNear(EPSILON), {0.321751f, 2.35619f}));

  Tensor2D<double, 2, 2> s2{1.0, 0.5, -0.5, 0.5};
  Tensor2D<double, 2, 2> t2{3.0, -0.5, 0.5, 0.5};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<double, 2, 2> {
    return stablehlo::atan2<Tensor2D<double, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(FloatNear(EPSILON),
                                     {0.321751, 2.35619, -0.785398, 0.785398}));

  Tensor3D<double, 1, 2, 2> s3{1.0, 0.5, -0.5, 0.5};
  Tensor3D<double, 1, 2, 2> t3{3.0, -0.5, 0.5, 0.5};

  auto lambda_3d = [&s3, &t3]() -> Tensor3D<double, 1, 2, 2> {
    return stablehlo::atan2<Tensor3D<double, 1, 2, 2>>(s3, t3);
  };

  EXPECT_THAT(lambda_3d(), Pointwise(FloatNear(EPSILON),
                                     {0.321751, 2.35619, -0.785398, 0.785398}));

  Tensor4D<float, 1, 3, 1, 2> s4{1.0f, 0.5f, -0.5f, 0.5f, M_PIf32, 0.0f};
  Tensor4D<float, 1, 3, 1, 2> t4{3.0f, -0.5f, 0.5f, 0.5f, 0.0f, -M_PIf32};

  auto lambda_4d = [&s4, &t4]() -> Tensor4D<float, 1, 3, 1, 2> {
    return stablehlo::atan2<Tensor4D<float, 1, 3, 1, 2>>(s4, t4);
  };

  EXPECT_THAT(lambda_4d(),
              Pointwise(FloatNear(EPSILON), {0.321751f, 2.35619f, -0.785398f,
                                             0.785398f, 1.570796f, M_PIf32}));
}

TEST(stablehlo, div) {
  EXPECT_EQ(-3, stablehlo::div(-3, 1));
  EXPECT_EQ(-6.75, stablehlo::div(27.0, -4.0));
  EXPECT_EQ(-6, stablehlo::div<int>(27.0, -4.0));

  Tensor0D<int> s0{27};
  Tensor0D<int> t0{-4};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return stablehlo::div<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {-6}));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};
  Tensor1D<float, 2> t1{0.2f, -3.7f};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<float, 2> {
    return stablehlo::div<Tensor1D<float, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(FloatNear(EPSILON), {-6.5f, -0.6486f}));

  Tensor2D<long, 2, 2> s2{3, 14, -31, -51};
  Tensor2D<long, 2, 2> t2{-2, 2, 6, 7};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return stablehlo::div<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {-1, 7, -5, -7}));

  Tensor3D<int8_t, 2, 2, 1> s3{3, 14, -31, -51};
  Tensor3D<int8_t, 2, 2, 1> t3{-2, 2, 6, 7};

  auto lambda_3d = [&s3, &t3]() -> Tensor3D<int8_t, 2, 2, 1> {
    return stablehlo::div<Tensor3D<int8_t, 2, 2, 1>>(s3, t3);
  };

  EXPECT_THAT(lambda_3d(), Pointwise(Eq(), {-1, 7, -5, -7}));

  Tensor4D<int16_t, 2, 1, 3, 1> s4{3, 14, -31, -51, 16, -2};
  Tensor4D<int16_t, 2, 1, 3, 1> t4{-2, 2, 6, 7, 8, -2};

  auto lambda_4d = [&s4, &t4]() -> Tensor4D<int16_t, 2, 1, 3, 1> {
    return stablehlo::div<Tensor4D<int16_t, 2, 1, 3, 1>>(s4, t4);
  };

  EXPECT_THAT(lambda_4d(), Pointwise(Eq(), {-1, 7, -5, -7, 2, 1}));
}

TEST(stablehlo, max) {
  EXPECT_EQ(3, stablehlo::max(-1, 3));

  Tensor0D<int> s0{-3};
  Tensor0D<int> t0{8};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return stablehlo::max<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {8}));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};
  Tensor1D<float, 2> t1{0.2f, -3.7f};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<float, 2> {
    return stablehlo::max<Tensor1D<float, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(FloatEq(), {0.2f, 2.4f}));

  Tensor2D<long, 2, 2> s2{3, 1, 4, 9};
  Tensor2D<long, 2, 2> t2{-2, 8, 6, -10};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return stablehlo::max<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {3, 8, 6, 9}));
}

TEST(stablehlo, min) {
  EXPECT_EQ(-1, stablehlo::min(-1, 3));

  Tensor0D<int> s0{-3};
  Tensor0D<int> t0{8};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return stablehlo::min<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {-3}));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};
  Tensor1D<float, 2> t1{0.2f, -3.7f};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<float, 2> {
    return stablehlo::min<Tensor1D<float, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(FloatEq(), {-1.3f, -3.7f}));

  Tensor2D<long, 2, 2> s2{3, 1, 4, 9};
  Tensor2D<long, 2, 2> t2{-2, 8, 6, -10};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return stablehlo::min<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {-2, 1, 4, -10}));
}

TEST(stablehlo, mul) {
  EXPECT_EQ(-3, stablehlo::mul(-1, 3));

  Tensor0D<int> s0{-3};
  Tensor0D<int> t0{8};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return stablehlo::mul<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {-24}));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};
  Tensor1D<float, 2> t1{0.2f, -3.7f};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<float, 2> {
    return stablehlo::mul<Tensor1D<float, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(FloatEq(), {-0.26f, -8.88f}));

  Tensor2D<long, 2, 2> s2{3, 1, 4, 9};
  Tensor2D<long, 2, 2> t2{-2, 8, 6, -10};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return stablehlo::mul<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {-6, 8, 24, -90}));
}

TEST(stablehlo, pow) {
  EXPECT_EQ(9, stablehlo::pow(3, 2));

  Tensor0D<int> s0{2};
  Tensor0D<int> t0{4};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return stablehlo::pow<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {16}));

  Tensor1D<float, 2> s1{4.0f, 2.0f};
  Tensor1D<float, 2> t1{0.5f, -2.0f};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<float, 2> {
    return stablehlo::pow<Tensor1D<float, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(FloatNear(EPSILON), {2.0f, 0.25f}));

  Tensor2D<long, 2, 2> s2{3, 1, 4, 2};
  Tensor2D<long, 2, 2> t2{0, -1, 3, -2};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return stablehlo::pow<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {1, 1, 64, 0}));
}

TEST(stablehlo, shift_left) {
  EXPECT_EQ(16u, stablehlo::shift_left(2u, 3u));

  Tensor0D<uint> s0{2};
  Tensor0D<uint> t0{8};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<uint> {
    return stablehlo::shift_left<Tensor0D<uint>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {512}));

  Tensor1D<uint8_t, 2> s1{3, 0};
  Tensor1D<uint8_t, 2> t1{2, 3};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<uint8_t, 2> {
    return stablehlo::shift_left<Tensor1D<uint8_t, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(Eq(), {12, 0}));

  Tensor2D<uint64_t, 2, 2> s2{0, 2, 5, 10};
  Tensor2D<uint64_t, 2, 2> t2{0, 1, 3, 4};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<uint64_t, 2, 2> {
    return stablehlo::shift_left<Tensor2D<uint64_t, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {0, 4, 40, 160}));
}

TEST(stablehlo, shift_right_logical) {
  EXPECT_EQ(2u, stablehlo::shift_right_logical(4u, 1u));

  Tensor0D<uint> s0{6};
  Tensor0D<uint> t0{2};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<uint> {
    return stablehlo::shift_right_logical<Tensor0D<uint>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {1}));

  Tensor1D<uint8_t, 2> s1{17, 32};
  Tensor1D<uint8_t, 2> t1{1, 3};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<uint8_t, 2> {
    return stablehlo::shift_right_logical<Tensor1D<uint8_t, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(Eq(), {8, 4}));

  Tensor2D<uint64_t, 2, 2> s2{0, 2, 25, 10};
  Tensor2D<uint64_t, 2, 2> t2{0, 1, 3, 2};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<uint64_t, 2, 2> {
    return stablehlo::shift_right_logical<Tensor2D<uint64_t, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {0, 1, 3, 2}));
}

TEST(stablehlo, sub) {
  EXPECT_EQ(-4, stablehlo::sub(-1, 3));

  Tensor0D<int> s0{-3};
  Tensor0D<int> t0{8};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return stablehlo::sub<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {-11}));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};
  Tensor1D<float, 2> t1{0.2f, -3.7f};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<float, 2> {
    return stablehlo::sub<Tensor1D<float, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(FloatEq(), {-1.5f, 6.1f}));

  Tensor2D<long, 2, 2> s2{3, 1, 4, 9};
  Tensor2D<long, 2, 2> t2{-2, 8, 6, -10};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return stablehlo::sub<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {5, -7, -2, 19}));
}

// Binary logical elementwise ops

TEST(stablehlo, or) {
  EXPECT_EQ(1, stablehlo::logical_or(2, 3));

  Tensor0D<int> s0{2};
  Tensor0D<int> t0{8};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return stablehlo::logical_or<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {1}));

  Tensor1D<int8_t, 2> s1{-1, 0};
  Tensor1D<int8_t, 2> t1{0, 0};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<int8_t, 2> {
    return stablehlo::logical_or<Tensor1D<int8_t, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(Eq(), {1, 0}));

  Tensor2D<long, 2, 2> s2{0, 2, 0, -1};
  Tensor2D<long, 2, 2> t2{0, 0, -2, -2};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return stablehlo::logical_or<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {0, 1, 1, 1}));
}

TEST(stablehlo, xor) {
  EXPECT_EQ(1, stablehlo::logical_xor(2, 3));

  Tensor0D<int> s0{2};
  Tensor0D<int> t0{8};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return stablehlo::logical_xor<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {1}));

  Tensor1D<int8_t, 2> s1{-1, 0};
  Tensor1D<int8_t, 2> t1{0, 0};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<int8_t, 2> {
    return stablehlo::logical_xor<Tensor1D<int8_t, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(Eq(), {1, 0}));

  Tensor2D<long, 2, 2> s2{0, 2, 0, -1};
  Tensor2D<long, 2, 2> t2{0, 0, -2, -2};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return stablehlo::logical_xor<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {0, 1, 1, 1}));
}

// Tuple ops

TEST(stablehlo, compare) {
  auto lambda = []() { return stablehlo::compare<int, std::less>(-1, 3); };
  EXPECT_EQ(true, lambda());

  Tensor0D<int> s0{-3};
  Tensor0D<int> t0{-8};

  auto lambda_0d = [&s0, &t0]() {
    return stablehlo::compare<Tensor0D<int>, std::less_equal>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {false}));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};
  Tensor1D<float, 2> t1{0.2f, 2.4f};

  auto lambda_1d = [&s1, &t1]() {
    return stablehlo::compare<Tensor1D<float, 2>, std::equal_to>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(Eq(), {false, true}));

  Tensor2D<long, 2, 2> s2{3, 1, 4, 9};
  Tensor2D<long, 2, 2> t2{-2, 1, 6, -10};

  auto lambda_2d = [&s2, &t2]() {
    return stablehlo::compare<Tensor2D<long, 2, 2>, std::greater_equal>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {true, true, false, true}));

  Tensor3D<long, 2, 1, 2> s3{3, 1, 4, 9};
  Tensor3D<long, 2, 1, 2> t3{-2, 1, 6, -10};

  auto lambda_3d = [&s3, &t3]() {
    return stablehlo::compare<Tensor3D<long, 2, 1, 2>, std::greater_equal>(s3,
                                                                           t3);
  };

  EXPECT_THAT(lambda_3d(), Pointwise(Eq(), {true, true, false, true}));

  Tensor4D<int, 2, 1, 2, 2> s4{3, 1, 4, 9, 10, 12, -4, 8};
  Tensor4D<int, 2, 1, 2, 2> t4{-2, 1, 6, -10, 9, 13, 4, 10};

  auto lambda_4d = [&s4, &t4]() {
    return stablehlo::compare<Tensor4D<int, 2, 1, 2, 2>, std::greater_equal>(
        s4, t4);
  };

  EXPECT_THAT(lambda_4d(), Pointwise(Eq(), {true, true, false, true, true,
                                            false, false, false}));
}

// Slice ops

TEST(stablehlo, slice) {
  // Slice Tensor1D
  Tensor1D<float, 5> s1{0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  auto t1 = stablehlo::slice<Tensor1D<float, 2>, Tensor1D<float, 5>>(s1, {2},
                                                                     {4}, {1});
  EXPECT_THAT(t1, Pointwise(FloatEq(), {2.0f, 3.0f}));

  auto t1_strided = stablehlo::slice<Tensor1D<float, 2>, Tensor1D<float, 5>>(
      s1, {1}, {4}, {2});
  EXPECT_THAT(t1_strided, Pointwise(FloatEq(), {1.0f, 3.0f}));

  // Slice Tensor2D
  Tensor2D<float, 4, 3> s2{0.0f, 1.0f, 2.0f, 3.0f, 4.0f,  5.0f,
                           6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  auto t2 = stablehlo::slice<Tensor2D<float, 2, 2>, Tensor2D<float, 4, 3>>(
      s2, {2, 1}, {4, 3}, {1, 1});

  EXPECT_THAT(t2, Pointwise(FloatEq(), {7.0f, 8.0f, 10.0f, 11.0f}));

  auto t2_strided =
      stablehlo::slice<Tensor2D<float, 2, 2>, Tensor2D<float, 4, 3>>(
          s2, {1, 0}, {4, 3}, {2, 2});

  EXPECT_THAT(t2_strided, Pointwise(FloatEq(), {3.0f, 5.0f, 9.0f, 11.0f}));

  // Slice Tensor3D
  Tensor3D<float, 4, 3, 2> s3{0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
                              6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
                              12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
                              18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f};
  auto t3 =
      stablehlo::slice<Tensor3D<float, 2, 2, 2>, Tensor3D<float, 4, 3, 2>>(
          s3, {2, 1, 0}, {4, 3, 2}, {1, 1, 1});
  EXPECT_THAT(t3, Pointwise(FloatEq(), {14.0f, 15.0f, 16.0f, 17.0f, 20.0f,
                                        21.0f, 22.0f, 23.0f}));

  auto t3_strided =
      stablehlo::slice<Tensor3D<float, 2, 2, 1>, Tensor3D<float, 4, 3, 2>>(
          s3, {0, 1, 0}, {4, 3, 2}, {2, 1, 2});
  EXPECT_THAT(t3_strided, Pointwise(FloatEq(), {2.0f, 4.0f, 14.0f, 16.0f}));

  auto t3_strided2 =
      stablehlo::slice<Tensor3D<float, 1, 2, 1>, Tensor3D<float, 4, 3, 2>>(
          s3, {0, 1, 0}, {2, 3, 2}, {2, 1, 2});
  EXPECT_THAT(t3_strided2, Pointwise(FloatEq(), {2.0f, 4.0f}));

  // Slice Tensor4D
  Tensor4D<float, 4, 3, 1, 2> s4{0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
                                 6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
                                 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
                                 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f};
  auto t4 = stablehlo::slice<Tensor4D<float, 2, 2, 1, 2>,
                             Tensor4D<float, 4, 3, 1, 2>>(
      s4, {2, 1, 0, 0}, {4, 3, 1, 2}, {1, 1, 1, 1});
  EXPECT_THAT(t4, Pointwise(FloatEq(), {14.0f, 15.0f, 16.0f, 17.0f, 20.0f,
                                        21.0f, 22.0f, 23.0f}));

  auto t4_2 = stablehlo::slice<Tensor4D<float, 4, 3, 1, 2>,
                               Tensor4D<float, 4, 3, 1, 2>>(
      s4, {0, 0, 0, 0}, {4, 3, 1, 2}, {1, 1, 1, 1});
  EXPECT_THAT(t4_2, Pointwise(FloatEq(), s4));

  auto t4_strided = stablehlo::slice<Tensor4D<float, 3, 2, 1, 1>,
                                     Tensor4D<float, 4, 3, 1, 2>>(
      s4, {1, 0, 0, 0}, {4, 3, 1, 2}, {1, 2, 1, 2});
  EXPECT_THAT(t4_strided,
              Pointwise(FloatEq(), {6.0f, 10.0f, 12.0f, 16.0f, 18.0f, 22.0f}));

  auto t4_strided_2 = stablehlo::slice<Tensor4D<float, 2, 1, 1, 1>,
                                       Tensor4D<float, 4, 3, 1, 2>>(
      s4, {0, 2, 0, 0}, {4, 3, 1, 1}, {2, 1, 1, 1});
  EXPECT_THAT(t4_strided_2, Pointwise(FloatEq(), {4.0f, 16.0f}));
}

TEST(stablehlo, dynamic_slice) {
  Tensor1D<float, 5> s1{0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  auto t1 = stablehlo::dynamic_slice<Tensor1D<float, 2>, Tensor1D<float, 5>>(
      s1, {2}, {2});
  EXPECT_THAT(t1, Pointwise(FloatEq(), {2.0f, 3.0f}));

  Tensor2D<float, 4, 3> s2{0.0f, 1.0f, 2.0f, 3.0f, 4.0f,  5.0f,
                           6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  auto t2 =
      stablehlo::dynamic_slice<Tensor2D<float, 2, 2>, Tensor2D<float, 4, 3>>(
          s2, {2}, {1}, {2, 2});

  EXPECT_THAT(t2, Pointwise(FloatEq(), {7.0f, 8.0f, 10.0f, 11.0f}));
}

TEST(stablehlo, dynamic_update_slice) {
  Tensor1D<float, 5> s1{0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  Tensor1D<float, 2> u1{5.0f, 6.0f};
  auto t1 =
      stablehlo::dynamic_update_slice<Tensor1D<float, 2>, Tensor1D<float, 5>>(
          s1, u1, {2});
  EXPECT_THAT(t1, Pointwise(FloatEq(), {0.0f, 1.0f, 5.0f, 6.0f, 4.0f}));

  Tensor2D<float, 4, 3> s2{0.0f, 1.0f, 2.0f, 3.0f, 4.0f,  5.0f,
                           6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  Tensor2D<float, 3, 2> u2{12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f};
  auto t2 =
      stablehlo::dynamic_update_slice<Tensor2D<float, 3, 2>,
                                      Tensor2D<float, 4, 3>>(s2, u2, {1}, {1});

  EXPECT_THAT(t2,
              Pointwise(FloatEq(), {0.0f, 1.0f, 2.0f, 3.0f, 12.0f, 13.0f, 6.0f,
                                    14.0f, 15.0f, 9.0f, 16.0f, 17.0f}));
}

// Other ops

TEST(stablehlo, batch_norm_inference) {
  Tensor<float, 4, 2> input{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  Tensor<float, 4, 2> expected_result{-3.0f, 2.0f,  1.0f, 6.0f,
                                      5.0f,  10.0f, 9.0f, 14.0f};
  Tensor<float, 4, 2> result =
      stablehlo::batch_norm_inference<Tensor<float, 4, 2>, Tensor<float, 2>>(
          input, {1.0f, 2.0f}, {1.0f, 2.0f}, {2.0f, 1.0f}, {0.249f, 0.999f},
          0.001f, 1);

  EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
}

TEST(stablehlo, bitcast_convert) {
  uint8_t a = 128;
  int8_t b = -128;
  EXPECT_EQ(b, stablehlo::bitcast_convert<int8_t>(a));

  Tensor0D<int16_t> t0{-1};
  auto lambda_0d = [&t0]() {
    return stablehlo::bitcast_convert<Tensor0D<uint16_t>>(t0);
  };
  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {65535}));

  Tensor1D<uint16_t, 2> t1{1, 2};
  auto lambda_1d = [&t1]() {
    return stablehlo::bitcast_convert<Tensor1D<int16_t, 2>>(t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(Eq(), {1, 2}));

  Tensor2D<int8_t, 2, 2> t2{0, -4, 3, -12};
  auto lambda_2d = [&t2]() {
    return stablehlo::bitcast_convert<Tensor2D<uint8_t, 2, 2>>(t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(DoubleEq(), {0, 252, 3, 244}));

  Tensor3D<int8_t, 2, 1, 2> t3{0, -4, 3, -12};
  auto lambda_3d = [&t3]() {
    return stablehlo::bitcast_convert<Tensor3D<uint8_t, 2, 1, 2>>(t3);
  };

  EXPECT_THAT(lambda_3d(), Pointwise(DoubleEq(), {0, 252, 3, 244}));

  Tensor4D<int8_t, 2, 1, 2, 2> t4{0, -4, 3, -12, -11, 0, 2, -4};
  auto lambda_4d = [&t4]() {
    return stablehlo::bitcast_convert<Tensor4D<uint8_t, 2, 1, 2, 2>>(t4);
  };

  EXPECT_THAT(lambda_4d(),
              Pointwise(DoubleEq(), {0, 252, 3, 244, 245, 0, 2, 252}));
}

TEST(stablehlo, broadcast_in_dim) {
  Tensor0D<int> t0{1};
  Tensor1D<int64_t, 0> b0;

  { // 0D -> 1D
    using Dest = Tensor1D<int, 4>;
    Dest expected_result{1, 1, 1, 1};
    Dest result = stablehlo::broadcast_in_dim<Dest>(t0, b0);
    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  { // 0D -> 2D
    using Dest = Tensor2D<int, 2, 3>;
    Dest expected_result{1, 1, 1, 1, 1, 1};
    Dest result = stablehlo::broadcast_in_dim<Dest>(t0, b0);
    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }

  Tensor<int, 2> t1{1, 2};
  { // 1D -> 2D
    using Dest = Tensor<int, 3, 2>;
    Dest expected_result{1, 2, 1, 2, 1, 2};
    Dest result = stablehlo::broadcast_in_dim<Dest>(t1, {1});
    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  { // 1D -> 2D
    using Dest = Tensor<int, 2, 3>;
    Dest expected_result{1, 1, 1, 2, 2, 2};
    Dest result = stablehlo::broadcast_in_dim<Dest>(t1, {0});
    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }

  Tensor<int, 2, 3> t2{1, 2, 3, 4, 5, 6};
  { // 2D transpose
    using Dest = Tensor<int, 3, 2>;
    Dest expected_result{1, 4, 2, 5, 3, 6};
    Dest result = stablehlo::broadcast_in_dim<Dest>(t2, {1, 0});
    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }

  Tensor2D<float, 1, 3> t3{1.1, 1.2, 1.3};
  { // 2D -> 3D
    using Dest = Tensor3D<float, 1, 2, 3>;
    Tensor1D<int64_t, 2> broadcast_dim{1, 2};
    Dest result = stablehlo::broadcast_in_dim<Dest>(t3, broadcast_dim);
    Dest expected_result{1.1, 1.2, 1.3, 1.1, 1.2, 1.3};
    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  { // 2D -> 3D + transpose
    using Dest = Tensor3D<float, 1, 3, 2>;
    Tensor1D<int64_t, 2> broadcast_dim{2, 1};
    Dest result = stablehlo::broadcast_in_dim<Dest>(t3, broadcast_dim);
    Dest expected_result{1.1, 1.1, 1.2, 1.2, 1.3, 1.3};
    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }

  { // 2D -> 4D
    using Dest = Tensor4D<float, 1, 2, 2, 3>;
    Tensor1D<int64_t, 2> broadcast_dim{2, 3};
    Dest result = stablehlo::broadcast_in_dim<Dest>(t3, broadcast_dim);
    Dest expected_result{1.1, 1.2, 1.3, 1.1, 1.2, 1.3,
                         1.1, 1.2, 1.3, 1.1, 1.2, 1.3};
    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }

  Tensor3D<float, 2, 2, 3> t4{1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                              2.1, 2.2, 2.3, 2.4, 2.5, 2.6};
  { // 3D -> 4D
    using Dest = Tensor4D<float, 2, 2, 2, 3>;

    Tensor1D<int64_t, 3> broadcast_dim{1, 2, 3};

    Dest result = stablehlo::broadcast_in_dim<Dest>(t4, broadcast_dim);
    Dest expected_result{1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.2,
                         2.3, 2.4, 2.5, 2.6, 1.1, 1.2, 1.3, 1.4,
                         1.5, 1.6, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6};
    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
}

TEST(stablehlo, clamp) {
  Tensor<float, 2, 1> operand{-1.0f, 1.0f};
  Tensor<float, 2, 1> min{0.0f, 0.0f};
  Tensor<float, 2, 1> max{3.0f, 0.0f};
  Tensor<float, 2, 1> expected_result{0.0, 0.0f};
  Tensor<float, 2, 1> result = stablehlo::clamp(min, operand, max);

  EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));

  // broadcasting
  Tensor<int32_t, 4, 2, 1> operand_b{0, 1, 2, 3, 4, 5, 6, 7};
  Tensor<int32_t> min_b{2};
  Tensor<int32_t> max_b{5};
  Tensor<int32_t, 4, 2, 1> expected_result_b{2, 2, 2, 3, 4, 5, 5, 5};
  Tensor<int32_t, 4, 2, 1> result_b = stablehlo::clamp(min_b, operand_b, max_b);

  EXPECT_THAT(result_b, Pointwise(Eq(), expected_result_b));
}

TEST(stablehlo, concatenate) {
  Tensor1D<int, 1> t1{1};
  Tensor1D<int, 2> t2{2, 3};
  Tensor1D<int, 3> t3{4, 5, 6};

  auto lambda_1d_1 = [&t1]() -> Tensor1D<int, 1> {
    return stablehlo::concatenate<0, Tensor1D<int, 1>, Tensor1D<int, 1>>(t1);
  };

  EXPECT_THAT(lambda_1d_1(), Pointwise(Eq(), {1}));

  auto lambda_1d_2 = [&t1, &t2]() -> Tensor1D<int, 3> {
    return stablehlo::concatenate<0, Tensor1D<int, 3>, Tensor1D<int, 1>,
                                  Tensor1D<int, 2>>(t1, t2);
  };

  EXPECT_THAT(lambda_1d_2(), Pointwise(Eq(), {1, 2, 3}));

  auto lambda_1d_3 = [&t1, &t2, &t3]() -> Tensor1D<int, 6> {
    return stablehlo::concatenate<0, Tensor1D<int, 6>, Tensor1D<int, 1>,
                                  Tensor1D<int, 2>, Tensor1D<int, 3>>(t1, t2,
                                                                      t3);
  };

  EXPECT_THAT(lambda_1d_3(), Pointwise(Eq(), {1, 2, 3, 4, 5, 6}));

  Tensor2D<float, 1, 2> t4{1.0f, 2.0f};
  Tensor2D<float, 2, 2> t5{3.0f, 4.0f, 5.0f, 6.0f};
  Tensor2D<float, 3, 2> t6{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  auto lambda_2d_2_row = [&t4, &t5]() -> Tensor2D<float, 3, 2> {
    return stablehlo::concatenate<0, Tensor2D<float, 3, 2>,
                                  Tensor2D<float, 1, 2>, Tensor2D<float, 2, 2>>(
        t4, t5);
  };

  EXPECT_THAT(lambda_2d_2_row(),
              Pointwise(FloatEq(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));

  auto lambda_2d_2_col = [&t4, &t5]() -> Tensor2D<float, 2, 3> {
    Tensor2D<float, 2, 1> t4_reshape =
        stablehlo::reshape<Tensor2D<float, 2, 1>>(t4);
    return stablehlo::concatenate<1, Tensor2D<float, 2, 3>,
                                  Tensor2D<float, 2, 1>, Tensor2D<float, 2, 2>>(
        t4_reshape, t5);
  };

  EXPECT_THAT(lambda_2d_2_col(),
              Pointwise(FloatEq(), {1.0f, 3.0f, 4.0f, 2.0f, 5.0f, 6.0f}));

  auto lambda_2d_3_row = [&t4, &t5, &t6]() -> Tensor2D<float, 6, 2> {
    return stablehlo::concatenate<0, Tensor2D<float, 6, 2>,
                                  Tensor2D<float, 1, 2>, Tensor2D<float, 2, 2>,
                                  Tensor2D<float, 3, 2>>(t4, t5, t6);
  };

  EXPECT_THAT(lambda_2d_3_row(),
              Pointwise(FloatEq(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                                    8.0f, 9.0f, 10.0f, 11.0f, 12.0f}));

  auto lambda_2d_3_col = [&t4, &t5, &t6]() -> Tensor2D<float, 2, 6> {
    Tensor2D<float, 2, 1> t4_reshape =
        stablehlo::reshape<Tensor2D<float, 2, 1>>(t4);
    Tensor2D<float, 2, 3> t6_reshape =
        stablehlo::reshape<Tensor2D<float, 2, 3>>(t6);
    return stablehlo::concatenate<1, Tensor2D<float, 2, 6>,
                                  Tensor2D<float, 2, 1>, Tensor2D<float, 2, 2>,
                                  Tensor2D<float, 2, 3>>(t4_reshape, t5,
                                                         t6_reshape);
  };

  EXPECT_THAT(lambda_2d_3_col(),
              Pointwise(FloatEq(), {1.0f, 3.0f, 4.0f, 7.0f, 8.0f, 9.0f, 2.0f,
                                    5.0f, 6.0f, 10.0f, 11.0f, 12.0f}));

  Tensor3D<float, 2, 2, 2> t7{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  Tensor3D<float, 2, 2, 2> t8{9.0f,  10.0f, 11.0f, 12.0f,
                              13.0f, 14.0f, 15.0f, 16.0f};
  Tensor3D<float, 2, 2, 1> t9{9.0f, 10.0f, 11.0f, 12.0f};

  auto lambda_3d_422 = [&t7, &t8]() -> Tensor3D<float, 4, 2, 2> {
    return stablehlo::concatenate<0, Tensor3D<float, 4, 2, 2>,
                                  Tensor3D<float, 2, 2, 2>,
                                  Tensor3D<float, 2, 2, 2>>(t7, t8);
  };

  EXPECT_THAT(lambda_3d_422(),
              Pointwise(FloatEq(),
                        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                         10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}));

  auto lambda_3d_242 = [&t7, &t8]() -> Tensor3D<float, 2, 4, 2> {
    return stablehlo::concatenate<1, Tensor3D<float, 2, 4, 2>,
                                  Tensor3D<float, 2, 2, 2>,
                                  Tensor3D<float, 2, 2, 2>>(t7, t8);
  };

  EXPECT_THAT(lambda_3d_242(),
              Pointwise(FloatEq(),
                        {1.0f, 2.0f, 3.0f, 4.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                         5.0f, 6.0f, 7.0f, 8.0f, 13.0f, 14.0f, 15.0f, 16.0f}));

  auto lambda_3d_223 = [&t7, &t9]() -> Tensor3D<float, 2, 2, 3> {
    return stablehlo::concatenate<2, Tensor3D<float, 2, 2, 3>,
                                  Tensor3D<float, 2, 2, 2>,
                                  Tensor3D<float, 2, 2, 1>>(t7, t9);
  };

  EXPECT_THAT(lambda_3d_223(),
              Pointwise(FloatEq(), {1.0f, 2.0f, 9.0f, 3.0f, 4.0f, 10.0f, 5.0f,
                                    6.0f, 11.0f, 7.0f, 8.0f, 12.0f}));

  Tensor4D<float, 2, 2, 2, 2> t10{1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                                  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                                  13.0f, 14.0f, 15.0f, 16.0f};
  Tensor4D<float, 2, 2, 2, 2> t11{17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f,
                                  23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f,
                                  29.0f, 30.0f, 31.0f, 32.0f};
  Tensor4D<float, 2, 2, 1, 2> t12{33.0f, 34.0f, 35.0f, 36.0f,
                                  37.0f, 38.0f, 39.0f, 40.0f};

  auto lambda_4d_4222 = [&t10, &t11]() -> Tensor4D<float, 4, 2, 2, 2> {
    return stablehlo::concatenate<0, Tensor4D<float, 4, 2, 2, 2>,
                                  Tensor4D<float, 2, 2, 2, 2>,
                                  Tensor4D<float, 2, 2, 2, 2>>(t10, t11);
  };

  EXPECT_THAT(
      lambda_4d_4222(),
      Pointwise(FloatEq(),
                {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                 9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f}));

  auto lambda_4d_2422 = [&t10, &t11]() -> Tensor4D<float, 2, 4, 2, 2> {
    return stablehlo::concatenate<1, Tensor4D<float, 2, 4, 2, 2>,
                                  Tensor4D<float, 2, 2, 2, 2>,
                                  Tensor4D<float, 2, 2, 2, 2>>(t10, t11);
  };

  EXPECT_THAT(
      lambda_4d_2422(),
      Pointwise(FloatEq(),
                {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
                 9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f}));

  auto lambda_4d_2242 = [&t10, &t11]() -> Tensor4D<float, 2, 2, 4, 2> {
    return stablehlo::concatenate<2, Tensor4D<float, 2, 2, 4, 2>,
                                  Tensor4D<float, 2, 2, 2, 2>,
                                  Tensor4D<float, 2, 2, 2, 2>>(t10, t11);
  };

  EXPECT_THAT(
      lambda_4d_2242(),
      Pointwise(FloatEq(),
                {1.0f,  2.0f,  3.0f,  4.0f,  17.0f, 18.0f, 19.0f, 20.0f,
                 5.0f,  6.0f,  7.0f,  8.0f,  21.0f, 22.0f, 23.0f, 24.0f,
                 9.0f,  10.0f, 11.0f, 12.0f, 25.0f, 26.0f, 27.0f, 28.0f,
                 13.0f, 14.0f, 15.0f, 16.0f, 29.0f, 30.0f, 31.0f, 32.0f}));

  auto lambda_4d_2224 = [&t10, &t11]() -> Tensor4D<float, 2, 2, 2, 4> {
    return stablehlo::concatenate<3, Tensor4D<float, 2, 2, 2, 4>,
                                  Tensor4D<float, 2, 2, 2, 2>,
                                  Tensor4D<float, 2, 2, 2, 2>>(t10, t11);
  };

  EXPECT_THAT(
      lambda_4d_2224(),
      Pointwise(FloatEq(),
                {1.0f,  2.0f,  17.0f, 18.0f, 3.0f,  4.0f,  19.0f, 20.0f,
                 5.0f,  6.0f,  21.0f, 22.0f, 7.0f,  8.0f,  23.0f, 24.0f,
                 9.0f,  10.0f, 25.0f, 26.0f, 11.0f, 12.0f, 27.0f, 28.0f,
                 13.0f, 14.0f, 29.0f, 30.0f, 15.0f, 16.0f, 31.0f, 32.0f}));

  auto lambda_4d_2232 = [&t10, &t12]() -> Tensor4D<float, 2, 2, 3, 2> {
    return stablehlo::concatenate<2, Tensor4D<float, 2, 2, 3, 2>,
                                  Tensor4D<float, 2, 2, 2, 2>,
                                  Tensor4D<float, 2, 2, 1, 2>>(t10, t12);
  };

  EXPECT_THAT(lambda_4d_2232(),
              Pointwise(FloatEq(), {1.0f,  2.0f,  3.0f,  4.0f,  33.0f, 34.0f,
                                    5.0f,  6.0f,  7.0f,  8.0f,  35.0f, 36.0f,
                                    9.0f,  10.0f, 11.0f, 12.0f, 37.0f, 38.0f,
                                    13.0f, 14.0f, 15.0f, 16.0f, 39.0f, 40.0f}));
}

TEST(stablehlo, convolution) {
  int64_t batch_group_count = 1;
  int64_t input_batch_dimension = 0;
  int64_t input_feature_dimension = 3;
  Tensor1D<int64_t, 2> input_spatial_dimensions{1, 2};
  int64_t kernel_input_feature_dimension = 2;
  int64_t kernel_output_feature_dimension = 3;
  Tensor1D<int64_t, 2> kernel_spatial_dimensions{0, 1};
  int64_t output_batch_dimension = 0;
  int64_t output_feature_dimension = 3;
  Tensor1D<int64_t, 2> output_spatial_dimensions{1, 2};
  int64_t feature_group_count = 1;
  Tensor1D<int64_t, 2> rhs_dilation{1, 1};
  Tensor1D<int64_t, 2> lhs_dilation{1, 1};
  {
    /// Adapted from
    /// https://github.com/google/iree/blob/efd78a0b47a46457a644f43d98617d3e279b2a79/iree/test/e2e/xla_ops/convolution.mlir#L33
    using InputType = Tensor4D<float, 1, 4, 5, 2>;  // N H W C
    using WeightType = Tensor4D<float, 3, 2, 2, 1>; // KH KW CIN COUT
    using ResultType = Tensor4D<float, 1, 4, 5, 1>; // N H W C
    InputType input{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
    WeightType weights{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    ResultType expected_result{600,  736,  872,  1008, 476,  1310, 1466,
                               1622, 1778, 805,  2090, 2246, 2402, 2558,
                               1135, 1080, 1152, 1224, 1296, 524};

    Tensor2D<int64_t, 2, 2> padding{1, 1, 0, 1}; // {pt, pb, pl, pr}
    Tensor1D<int64_t, 2> window_strides{1, 1};
    ResultType result =
        stablehlo::convolution<ResultType, InputType, WeightType>(
            input, weights, batch_group_count, input_batch_dimension,
            input_feature_dimension, input_spatial_dimensions,
            kernel_input_feature_dimension, kernel_output_feature_dimension,
            kernel_spatial_dimensions, output_batch_dimension,
            output_feature_dimension, output_spatial_dimensions,
            feature_group_count, padding, lhs_dilation, rhs_dilation,
            window_strides);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    // Strided convolution
    using InputType = Tensor4D<float, 1, 4, 4, 1>;  // N H W C
    using WeightType = Tensor4D<float, 2, 2, 1, 1>; // KH KW CIN COUT
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

    Tensor2D<int64_t, 2, 2> padding{0, 0, 0, 0}; // {pt, pb, pl, pr}
    Tensor1D<int64_t, 2> window_strides{2, 2};
    ResultType result =
        stablehlo::convolution<ResultType, InputType, WeightType>(
            input, weights, batch_group_count, input_batch_dimension,
            input_feature_dimension, input_spatial_dimensions,
            kernel_input_feature_dimension, kernel_output_feature_dimension,
            kernel_spatial_dimensions, output_batch_dimension,
            output_feature_dimension, output_spatial_dimensions,
            feature_group_count, padding, lhs_dilation, rhs_dilation,
            window_strides);

    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
}

TEST(stablehlo, convolution_depthwise) {
  using InputType = Tensor4D<float, 1, 4, 5, 2>;
  using WeightType = Tensor4D<float, 2, 2, 1, 2>;
  using ResultType = Tensor4D<float, 1, 3, 4, 2>;
  InputType input{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                  15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                  29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
  WeightType weights{1, 2, 3, 4, 5, 6, 7, 8};
  ResultType expected_result{156, 204, 188, 244, 220, 284, 252, 324,
                             316, 404, 348, 444, 380, 484, 412, 524,
                             476, 604, 508, 644, 540, 684, 572, 724};

  int64_t batch_group_count = 1;
  int64_t input_batch_dimension = 0;
  int64_t input_feature_dimension = 3;
  Tensor1D<int64_t, 2> input_spatial_dimensions{1, 2};
  int64_t kernel_input_feature_dimension = 2;
  int64_t kernel_output_feature_dimension = 3;
  Tensor1D<int64_t, 2> kernel_spatial_dimensions{0, 1};
  int64_t output_batch_dimension = 0;
  int64_t output_feature_dimension = 3;
  Tensor1D<int64_t, 2> output_spatial_dimensions{1, 2};
  int64_t feature_group_count = 2;
  Tensor2D<int64_t, 2, 2> padding{0, 0, 0, 0};
  Tensor1D<int64_t, 2> rhs_dilation{1, 1};
  Tensor1D<int64_t, 2> lhs_dilation{1, 1};
  Tensor1D<int64_t, 2> window_strides{1, 1};

  ResultType result = stablehlo::convolution<ResultType, InputType, WeightType>(
      input, weights, batch_group_count, input_batch_dimension,
      input_feature_dimension, input_spatial_dimensions,
      kernel_input_feature_dimension, kernel_output_feature_dimension,
      kernel_spatial_dimensions, output_batch_dimension,
      output_feature_dimension, output_spatial_dimensions, feature_group_count,
      padding, lhs_dilation, rhs_dilation, window_strides);

  EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
}

TEST(stablehlo, DISABLED_convolution_grouped) {
  // TODO implement test
}

TEST(stablehlo, DISABLED_convolution_dilated) {
  // TODO implement test
}

TEST(stablehlo, dot) {
  Tensor2D<int, 2, 2> a2{1, 0, 0, 1};
  Tensor2D<int, 2, 2> b2{4, 1, 2, 2};

  auto lambda_2d = [&a2, &b2]() -> Tensor2D<int, 2, 2> {
    return stablehlo::dot<Tensor2D<int, 2, 2>>(a2, b2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {4, 1, 2, 2}));
}

TEST(stablehlo, reshape) {
  Tensor0D<int> s0{-3};
  auto t0 = stablehlo::reshape<Tensor1D<int, 1>>(s0);
  auto t0_1 = stablehlo::reshape<Tensor2D<int, 1, 1>>(s0);
  EXPECT_THAT(t0, Pointwise(Eq(), {-3}));
  EXPECT_THAT(t0_1, Pointwise(Eq(), {-3}));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};
  auto t1 = stablehlo::reshape<Tensor2D<float, 1, 2>>(s1);

  EXPECT_THAT(t1, Pointwise(FloatEq(), {-1.3f, 2.4f}));

  Tensor2D<long, 2, 2> s2{3, 1, 4, 9};
  auto t2 = stablehlo::reshape<Tensor1D<long, 4>>(s2);

  EXPECT_THAT(t2, Pointwise(Eq(), {3, 1, 4, 9}));
}

TEST(stablehlo, pad) {
  Tensor<int32_t, 2, 3> operand{1, 2, 3, 4, 5, 6};
  Tensor<int32_t> value{0};

  Tensor<int32_t, 3, 6> expected_result0{0, 1, 2, 3, 0, 0, 0, 4, 5,
                                         6, 0, 0, 0, 0, 0, 0, 0, 0};
  Tensor<int32_t, 3, 6> result0 = stablehlo::pad<Tensor<int32_t, 3, 6>>(
      operand, value, {0, 1}, {1, 2}, {0, 0});

  EXPECT_THAT(result0, Pointwise(Eq(), expected_result0));

  Tensor<int32_t, 3, 5> expected_result1{1, 0, 2, 0, 3, 0, 0, 0,
                                         0, 0, 4, 0, 5, 0, 6};
  Tensor<int32_t, 3, 5> result1 = stablehlo::pad<Tensor<int32_t, 3, 5>>(
      operand, value, {0, 0}, {0, 0}, {1, 1});

  EXPECT_THAT(result1, Pointwise(Eq(), expected_result1));

  Tensor<int32_t, 4, 8> expected_result2{0, 1, 0, 2, 0, 3, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 4, 0, 5, 0, 6,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Tensor<int32_t, 4, 8> result2 = stablehlo::pad<Tensor<int32_t, 4, 8>>(
      operand, value, {0, 1}, {1, 2}, {1, 1});

  EXPECT_THAT(result2, Pointwise(Eq(), expected_result2));
}

Tensor<int32_t> reduce_computation(Tensor<int32_t> a, Tensor<int32_t> b) {
  Tensor<int32_t> v0 = stablehlo::add(a, b);
  return v0;
}

std::tuple<Tensor<int32_t>, Tensor<int32_t>>
reduce_computation_tuple(Tensor<int32_t> reduction_a, Tensor<int32_t> next_a,
                         Tensor<int32_t> reduction_b, Tensor<int32_t> next_b) {
  Tensor<int32_t> v0 = stablehlo::add(reduction_a, next_a);
  Tensor<int32_t> v1 = stablehlo::min(reduction_b, next_b);
  return std::make_tuple(v0, v1);
}

TEST(stablehlo, reduce) {
  {
    Tensor<int32_t, 2, 3> x{1, 2, 3, 4, 5, 6};
    Tensor<int32_t> initValue{0};

    Tensor<int32_t, 3> expected_result{5, 7, 9};
    Tensor<int32_t, 3> result = stablehlo::reduce<Tensor<int32_t, 3>, 1>(
        x, initValue, {0}, reduce_computation);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }

  {
    Tensor<int32_t, 2, 3> x{1, 2, 3, 4, 5, 6};
    Tensor<int32_t> initValue{0};

    Tensor<int32_t, 2> expected_result{6, 15};
    Tensor<int32_t, 2> result = stablehlo::reduce<Tensor<int32_t, 2>, 1>(
        x, initValue, {1}, reduce_computation);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }

  {
    Tensor<int32_t, 2, 3> x{1, 2, 3, 4, 5, 6};
    Tensor<int32_t> initValue{0};

    Tensor<int32_t> expected_result{21};
    Tensor<int32_t> result = stablehlo::reduce<Tensor<int32_t>, 2>(
        x, initValue, {0, 1}, reduce_computation);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }

  {
    Tensor<int32_t, 4, 2, 3> x{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
                               1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
    Tensor<int32_t> initValue{0};

    Tensor<int32_t, 2, 3> expected_result{4, 8, 12, 16, 20, 24};
    Tensor<int32_t, 2, 3> result = stablehlo::reduce<Tensor<int32_t, 2, 3>, 1>(
        x, initValue, {0}, reduce_computation);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }

  {
    Tensor<int32_t, 4, 2, 3> x{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
                               1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
    Tensor<int32_t> initValue{0};

    Tensor<int32_t, 4, 3> expected_result{5, 7, 9, 5, 7, 9, 5, 7, 9, 5, 7, 9};
    Tensor<int32_t, 4, 3> result = stablehlo::reduce<Tensor<int32_t, 4, 3>, 1>(
        x, initValue, {1}, reduce_computation);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }

  {
    Tensor<int32_t, 4, 2, 3> x{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
                               1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
    Tensor<int32_t> initValue{0};

    Tensor<int32_t, 4, 2> expected_result{6, 15, 6, 15, 6, 15, 6, 15};
    Tensor<int32_t, 4, 2> result = stablehlo::reduce<Tensor<int32_t, 4, 2>, 1>(
        x, initValue, {2}, reduce_computation);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }

  {
    Tensor<int32_t, 4, 2, 3> x{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
                               1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
    Tensor<int32_t> initValue{0};

    Tensor<int32_t, 3> expected_result{20, 28, 36};
    Tensor<int32_t, 3> result = stablehlo::reduce<Tensor<int32_t, 3>, 2>(
        x, initValue, {0, 1}, reduce_computation);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }

  {
    Tensor<int32_t, 4, 2, 3> x{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
                               1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
    Tensor<int32_t> initValue{0};

    Tensor<int32_t, 2> expected_result{24, 60};
    Tensor<int32_t, 2> result = stablehlo::reduce<Tensor<int32_t, 2>, 2>(
        x, initValue, {0, 2}, reduce_computation);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }

  {
    Tensor<int32_t, 4, 2, 3> x{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
                               1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
    Tensor<int32_t> initValue{0};

    Tensor<int32_t, 4> expected_result{21, 21, 21, 21};
    Tensor<int32_t, 4> result = stablehlo::reduce<Tensor<int32_t, 4>, 2>(
        x, initValue, {1, 2}, reduce_computation);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }

  {
    Tensor<int32_t, 4, 2, 3> x{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
                               1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
    Tensor<int32_t> initValue{0};

    Tensor<int32_t> expected_result{84};
    Tensor<int32_t> result = stablehlo::reduce<Tensor<int32_t>, 3>(
        x, initValue, {0, 1, 2}, reduce_computation);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }

  // 2 return values
  {
    Tensor<int32_t, 2, 3> x1{1, 2, 3, 4, 5, 6};
    Tensor<int32_t, 2, 3> x2{1, 2, 3, 4, 5, 6};
    Tensor<int32_t> initValue1{0};
    Tensor<int32_t> initValue2{std::numeric_limits<int32_t>::max()};

    std::tuple<Tensor<int32_t, 3>, Tensor<int32_t, 3>> expected_result{
        {5, 7, 9}, {1, 2, 3}};

    std::tuple<Tensor<int32_t, 3>, Tensor<int32_t, 3>> result =
        stablehlo::reduce<Tensor<int32_t, 3>, Tensor<int32_t, 3>, 1>(
            x1, x2, initValue1, initValue2, {0}, reduce_computation_tuple);

    EXPECT_THAT(std::get<0>(result),
                Pointwise(Eq(), std::get<0>(expected_result)));
    EXPECT_THAT(std::get<1>(result),
                Pointwise(Eq(), std::get<1>(expected_result)));
  }

  {
    Tensor<int32_t, 4, 2, 3> x1{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
                                1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
    Tensor<int32_t, 4, 2, 3> x2{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
                                1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
    Tensor<int32_t> initValue1{0};
    Tensor<int32_t> initValue2{std::numeric_limits<int32_t>::max()};

    std::tuple<Tensor<int32_t>, Tensor<int32_t>> expected_result{{84}, {1}};
    std::tuple<Tensor<int32_t>, Tensor<int32_t>> result =
        stablehlo::reduce<Tensor<int32_t>, Tensor<int32_t>, 3>(
            x1, x2, initValue1, initValue2, {0, 1, 2},
            reduce_computation_tuple);

    EXPECT_THAT(std::get<0>(result),
                Pointwise(Eq(), std::get<0>(expected_result)));
    EXPECT_THAT(std::get<1>(result),
                Pointwise(Eq(), std::get<1>(expected_result)));
  }
}

TEST(stablehlo, reduce_window) {
  Tensor<int32_t> c0{std::numeric_limits<int32_t>::min()};
  Tensor<int32_t, 4, 8> t0{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                           23, 24, 25, 26, 27, 28, 29, 30, 31, 32};

  auto max = [](Tensor<int32_t> a, Tensor<int32_t> b) {
    return stablehlo::max(a, b);
  };

  Tensor<int32_t, 2, 4> expected_result0{10, 12, 14, 16, 26, 28, 30, 32};
  Tensor<int32_t, 2, 4> result0 =
      stablehlo::reduce_window<Tensor<int32_t, 2, 4>>(
          t0, c0, {2, 2}, {2, 2}, {1, 1}, {1, 1}, {0, 0, 0, 0}, max);

  EXPECT_THAT(result0, Pointwise(Eq(), expected_result0));

  Tensor<int32_t, 3, 7> expected_result1{10, 11, 12, 13, 14, 15, 16,
                                         18, 19, 20, 21, 22, 23, 24,
                                         26, 27, 28, 29, 30, 31, 32};
  Tensor<int32_t, 3, 7> result1 =
      stablehlo::reduce_window<Tensor<int32_t, 3, 7>>(
          t0, c0, {2, 2}, {1, 1}, {1, 1}, {1, 1}, {0, 0, 0, 0}, max);

  EXPECT_THAT(result1, Pointwise(Eq(), expected_result1));

  auto min = [](Tensor<float> a, Tensor<float> b) {
    return stablehlo::min(a, b);
  };

  Tensor<float> c1{std::numeric_limits<float>::max()};
  Tensor<float, 5> t1{10000.0f, 1000.0f, 100.0f, 10.0f, 1.0f};

  Tensor<float, 2> expected_result2{100.0f, 1.0f};
  Tensor<float, 2> result2 = stablehlo::reduce_window<Tensor<float, 2>>(
      t1, c1, {3}, {2}, {1}, {1}, {0, 0}, min);

  EXPECT_THAT(result2, Pointwise(Eq(), expected_result2));

  Tensor<float, 3> expected_result3{1000.0f, 10.0f, 1.0f};
  Tensor<float, 3> result3 = stablehlo::reduce_window<Tensor<float, 3>>(
      t1, c1, {3}, {2}, {1}, {1}, {1, 1}, min);

  EXPECT_THAT(result3, Pointwise(Eq(), expected_result3));
}

TEST(stablehlo, select) {
  EXPECT_EQ(-1, stablehlo::select(true, -1, 3));
  EXPECT_EQ(3, stablehlo::select(false, -1, 3));

  {
    Tensor0D<int> s{-3};
    Tensor0D<int> t{8};
    Tensor0D<bool> p{true};

    Tensor0D<int> expected_result{-3};
    Tensor0D<int> result = stablehlo::select<Tensor0D<int>>(p, s, t);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor1D<float, 2> s{-1.3f, 2.4f};
    Tensor1D<float, 2> t{0.2f, -3.7f};
    Tensor1D<bool, 2> p{true, false};

    Tensor1D<float, 2> expected_result{-1.3f, -3.7f};
    Tensor1D<float, 2> result = stablehlo::select<Tensor1D<float, 2>>(p, s, t);

    EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));
  }
  {
    Tensor1D<float, 2> s{-1.3f, 2.4f};
    Tensor1D<float, 2> t{0.2f, -3.7f};
    Tensor0D<bool> p{true};

    Tensor1D<float, 2> expected_result = s;
    Tensor1D<float, 2> result = stablehlo::select<Tensor1D<float, 2>>(p, s, t);

    EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));
  }
  {
    Tensor2D<long, 2, 2> s{3, 1, 4, 9};
    Tensor2D<long, 2, 2> t{-2, 8, 6, -10};
    Tensor2D<bool, 2, 2> p{false, true, true, false};

    Tensor2D<long, 2, 2> expected_result{-2, 1, 4, -10};
    Tensor2D<long, 2, 2> result =
        stablehlo::select<Tensor2D<long, 2, 2>>(p, s, t);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor2D<long, 2, 2> s{3, 1, 4, 9};
    Tensor2D<long, 2, 2> t{-2, 8, 6, -10};
    Tensor0D<bool> p{false};

    Tensor2D<long, 2, 2> expected_result = t;
    Tensor2D<long, 2, 2> result =
        stablehlo::select<Tensor2D<long, 2, 2>>(p, s, t);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
}

TEST(stablehlo, transpose) {
  {
    Tensor0D<int32_t> operand{1};
    Tensor1D<int64_t, 0> perms;

    Tensor0D<int32_t> expected_result{1};
    Tensor0D<int32_t> result =
        stablehlo::transpose<Tensor0D<int32_t>>(operand, perms);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor1D<double, 2> operand{1., 2.};
    Tensor1D<int64_t, 1> perms{0};

    Tensor1D<double, 2> expected_result{1., 2.};
    Tensor1D<double, 2> result =
        stablehlo::transpose<Tensor1D<double, 2>>(operand, perms);

    EXPECT_THAT(result, Pointwise(DoubleEq(), expected_result));
  }
  {
    Tensor2D<int64_t, 2, 3> operand{1, 2, 3, 4, 5, 6};
    Tensor1D<int64_t, 2> perms{1, 0};

    Tensor2D<int64_t, 3, 2> expected_result{1, 4, 2, 5, 3, 6};
    Tensor2D<int64_t, 3, 2> result =
        stablehlo::transpose<Tensor2D<int64_t, 3, 2>>(operand, perms);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor3D<float, 2, 1, 2> operand{1.f, 2.f, 3.f, 4.f};
    Tensor1D<int64_t, 3> perms{0, 1, 2};

    Tensor3D<float, 2, 1, 2> expected_result{1.f, 2.f, 3.f, 4.f};
    Tensor3D<float, 2, 1, 2> result =
        stablehlo::transpose<Tensor3D<float, 2, 1, 2>>(operand, perms);

    EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));
  }
  {
    Tensor4D<int32_t, 1, 2, 4, 3> operand{1,  2,  3,  4,  5,  6,  7,  8,
                                          9,  10, 11, 12, 13, 14, 15, 16,
                                          17, 18, 19, 20, 21, 22, 23, 24};
    Tensor1D<int64_t, 4> perms{3, 1, 2, 0};

    Tensor4D<int32_t, 3, 2, 4, 1> expected_result{1, 4, 7, 10, 13, 16, 19, 22,
                                                  2, 5, 8, 11, 14, 17, 20, 23,
                                                  3, 6, 9, 12, 15, 18, 21, 24};
    Tensor4D<int32_t, 3, 2, 4, 1> result =
        stablehlo::transpose<Tensor4D<int32_t, 3, 2, 4, 1>>(operand, perms);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
}

} // namespace
