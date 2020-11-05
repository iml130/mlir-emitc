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

#include "emitc/emitc_mhlo.h"
#include "emitc/emitc_tensor.h"

namespace {

using ::testing::DoubleEq;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pointwise;

const float EPSILON = 5e-4;

// Unary elementwise ops

TEST(mhlo, abs) {
  EXPECT_EQ(1, mhlo::abs(-1));

  Tensor0D<int> t0{-1};
  Tensor1D<float, 2> t1{-1.0f, -2.0f};
  Tensor2D<long, 2, 2> t2{-2, -1, 0, 2};
  Tensor3D<int32_t, 2, 1, 2> t3{-2, -1, 0, 2};
  Tensor4D<int, 2, 2, 1, 2> t4{-2, -1, 0, 0, 3, -3, -2, 1};

  EXPECT_THAT(mhlo::abs(t0), Pointwise(Eq(), {1}));
  EXPECT_THAT(mhlo::abs(t1), Pointwise(Eq(), {1.0f, 2.0f}));
  EXPECT_THAT(mhlo::abs(t2), Pointwise(Eq(), {2, 1, 0, 2}));
  EXPECT_THAT(mhlo::abs(t3), Pointwise(Eq(), {2, 1, 0, 2}));
  EXPECT_THAT(mhlo::abs(t4), Pointwise(Eq(), {2, 1, 0, 0, 3, 3, 2, 1}));

  // TODO:: Test complex to real.
}

TEST(mhlo, ceil) {
  EXPECT_EQ(1.0, mhlo::ceil(0.7));

  Tensor0D<float> t0{0.7f};
  Tensor1D<float, 2> t1{1.6f, 2.0f};
  Tensor2D<double, 2, 2> t2{2.1, 1.6, 0.0, 2.0};
  Tensor3D<double, 2, 1, 2> t3{2.1, 1.6, 0.0, 2.0};
  Tensor4D<double, 1, 2, 1, 2> t4{2.1, 1.6, 0.0, 2.0};

  EXPECT_THAT(mhlo::ceil(t0), Pointwise(Eq(), {1.0f}));
  EXPECT_THAT(mhlo::ceil(t1), Pointwise(Eq(), {2.0f, 2.0f}));
  EXPECT_THAT(mhlo::ceil(t2), Pointwise(Eq(), {3.0, 2.0, 0.0, 2.0}));
  EXPECT_THAT(mhlo::ceil(t3), Pointwise(Eq(), {3.0, 2.0, 0.0, 2.0}));
  EXPECT_THAT(mhlo::ceil(t4), Pointwise(Eq(), {3.0, 2.0, 0.0, 2.0}));
}

TEST(mhlo, convert) {
  uint32_t a = 1;
  uint64_t b = 1;
  EXPECT_EQ(b, mhlo::convert<uint64_t>(a));

  Tensor0D<uint32_t> t0{1};
  auto lambda_0d = [&t0]() -> Tensor0D<size_t> {
    return mhlo::convert<Tensor0D<size_t>>(t0);
  };
  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {1}));

  Tensor1D<uint16_t, 2> t1{1, 2};
  auto lambda_1d = [&t1]() -> Tensor1D<size_t, 2> {
    return mhlo::convert<Tensor1D<size_t, 2>>(t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(Eq(), {1, 2}));

  Tensor2D<float, 2, 2> t2{1.0f, 2.0f, 4.0f, 8.0f};
  auto lambda_2d = [&t2]() -> Tensor2D<double, 2, 2> {
    return mhlo::convert<Tensor2D<double, 2, 2>>(t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(DoubleEq(), {1.0, 2.0, 4.0, 8.0}));

  Tensor3D<float, 2, 1, 2> t3{1.0f, 2.0f, 4.0f, 8.0f};
  auto lambda_3d = [&t3]() -> Tensor3D<double, 2, 1, 2> {
    return mhlo::convert<Tensor3D<double, 2, 1, 2>>(t3);
  };

  EXPECT_THAT(lambda_3d(), Pointwise(DoubleEq(), {1.0, 2.0, 4.0, 8.0}));

  Tensor4D<double, 2, 1, 2, 2> t4{1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0};
  auto lambda_4d = [&t4]() -> Tensor4D<float, 2, 1, 2, 2> {
    return mhlo::convert<Tensor4D<float, 2, 1, 2, 2>>(t4);
  };

  EXPECT_THAT(lambda_4d(), Pointwise(FloatEq(), {1.0f, 2.0f, 4.0f, 8.0f, 16.0f,
                                                 32.0f, 64.0f, 128.0f}));
}

TEST(mhlo, cos) {
  EXPECT_NEAR(1.0f, mhlo::cos(0.0f), EPSILON);

  Tensor0D<float> t0{M_PIf32};
  Tensor1D<float, 2> t1{M_PI_2f32, -M_PI_2f32};
  Tensor2D<double, 2, 2> t2{2 * M_PIf32, 0.0f, -0.5f, 0.5f};

  EXPECT_THAT(mhlo::cos(t0), Pointwise(FloatNear(EPSILON), {-1.0f}));
  EXPECT_THAT(mhlo::cos(t1), Pointwise(FloatNear(EPSILON), {0.0f, 0.0f}));
  EXPECT_THAT(mhlo::cos(t2), Pointwise(FloatNear(EPSILON),
                                       {1.0f, 1.0f, 0.8775826f, 0.8775826f}));
}

TEST(mhlo, exponential) {
  EXPECT_NEAR(M_Ef32, mhlo::exponential(1.0f), EPSILON);

  Tensor0D<float> t0{0.0f};
  Tensor1D<float, 2> t1{M_LN2f32, M_LN10f32};
  Tensor2D<double, 2, 2> t2{1.0f, 2.0f, 3.0f, 4.0f};

  EXPECT_THAT(mhlo::exponential(t0), Pointwise(FloatNear(EPSILON), {1.0f}));
  EXPECT_THAT(mhlo::exponential(t1),
              Pointwise(FloatNear(EPSILON), {2.0f, 10.0f}));
  EXPECT_THAT(mhlo::exponential(t2),
              Pointwise(FloatNear(EPSILON),
                        {2.718281f, 7.389056f, 20.085536f, 54.598150f}));
}

TEST(mhlo, floor) {
  EXPECT_EQ(0.0, mhlo::floor(0.7));

  Tensor0D<float> t0{0.7f};
  Tensor1D<float, 2> t1{1.6f, 2.0f};
  Tensor2D<double, 2, 2> t2{2.1, 1.6, 0.0, 2.0};

  EXPECT_THAT(mhlo::floor(t0), Pointwise(Eq(), {0.0f}));
  EXPECT_THAT(mhlo::floor(t1), Pointwise(Eq(), {1.0f, 2.0f}));
  EXPECT_THAT(mhlo::floor(t2), Pointwise(Eq(), {2.0, 1.0, 0.0, 2.0}));
}

TEST(mhlo, is_finite) {
  EXPECT_EQ(true, mhlo::is_finite(0.0f));

  Tensor0D<float> t0{M_PIf32};
  Tensor1D<float, 2> t1{M_PI_2f32, INFINITY};
  Tensor2D<double, 2, 2> t2{INFINITY, -INFINITY, NAN, -0.0f};

  EXPECT_THAT(mhlo::is_finite(t0), Pointwise(Eq(), {true}));
  EXPECT_THAT(mhlo::is_finite(t1), Pointwise(Eq(), {true, false}));
  EXPECT_THAT(mhlo::is_finite(t2),
              Pointwise(Eq(), {false, false, false, true}));
}

TEST(mhlo, log) {
  EXPECT_NEAR(0.0f, mhlo::log(1.0f), EPSILON);

  Tensor0D<float> t0{M_Ef32};
  Tensor1D<float, 2> t1{M_Ef32 * M_Ef32, M_Ef32 * M_Ef32 * M_Ef32};
  Tensor2D<double, 2, 2> t2{1.0f, 2.0f, 3.0f, 4.0f};

  EXPECT_THAT(mhlo::log(t0), Pointwise(FloatNear(EPSILON), {1.0f}));
  EXPECT_THAT(mhlo::log(t1), Pointwise(FloatNear(EPSILON), {2.0f, 3.0f}));
  // clang-format off
  EXPECT_THAT(mhlo::log(t2), Pointwise(FloatNear(EPSILON), {0.0f,
  0.693147f, 1.098612f, 1.386294f}));
  // clang-format on
}

TEST(mhlo, negate) {
  EXPECT_EQ(1, mhlo::negate(-1));

  Tensor0D<int> s0{-3};

  auto lambda_0d = [&s0]() -> Tensor0D<int> {
    return mhlo::negate<Tensor0D<int>>(s0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {3}));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};

  auto lambda_1d = [&s1]() -> Tensor1D<float, 2> {
    return mhlo::negate<Tensor1D<float, 2>>(s1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(FloatEq(), {1.3f, -2.4f}));

  Tensor2D<long, 2, 2> s2{3, 1, -4, 0};

  auto lambda_2d = [&s2]() -> Tensor2D<long, 2, 2> {
    return mhlo::negate<Tensor2D<long, 2, 2>>(s2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {-3, -1, 4, 0}));
}

TEST(mhlo, round) {
  EXPECT_EQ(1.0, mhlo::round(0.7));
  EXPECT_EQ(0.0, mhlo::round(0.4));

  Tensor0D<float> t0{0.7f};
  Tensor1D<float, 3> t1{1.4f, 1.6f, 2.0f};
  Tensor2D<double, 2, 2> t2{2.1, 1.6, 0.0, 2.0};
  Tensor3D<float, 2, 1, 2> t3{2.1, 1.6, 0.0, 2.0};
  Tensor4D<double, 2, 2, 1, 1> t4{2.1, 1.6, 0.0, 2.0};

  EXPECT_THAT(mhlo::round(t0), Pointwise(Eq(), {1.0f}));
  EXPECT_THAT(mhlo::round(t1), Pointwise(Eq(), {1.0f, 2.0f, 2.0f}));
  EXPECT_THAT(mhlo::round(t2), Pointwise(Eq(), {2.0, 2.0, 0.0, 2.0}));
  EXPECT_THAT(mhlo::round(t3), Pointwise(Eq(), {2.0f, 2.0f, 0.0f, 2.0f}));
  EXPECT_THAT(mhlo::round(t4), Pointwise(Eq(), {2.0, 2.0, 0.0, 2.0}));
}

TEST(mhlo, sin) {
  EXPECT_NEAR(0.0f, mhlo::sin(0.0f), EPSILON);

  Tensor0D<float> t0{M_PIf32};
  Tensor1D<float, 2> t1{M_PI_2f32, -M_PI_2f32};
  Tensor2D<double, 2, 2> t2{2 * M_PIf32, 0.0f, -0.5f, 0.5f};
  Tensor3D<double, 1, 2, 2> t3{2 * M_PIf32, 0.0f, -0.5f, 0.5f};
  Tensor4D<double, 2, 1, 2, 1> t4{2 * M_PIf32, 0.0f, -0.5f, 0.5f};

  EXPECT_THAT(mhlo::sin(t0), Pointwise(FloatNear(EPSILON), {0.0f}));
  EXPECT_THAT(mhlo::sin(t1), Pointwise(FloatNear(EPSILON), {1.0f, -1.0f}));
  EXPECT_THAT(mhlo::sin(t2), Pointwise(FloatNear(EPSILON),
                                       {0.0f, 0.0f, -0.479426f, 0.479426f}));
  EXPECT_THAT(mhlo::sin(t3), Pointwise(FloatNear(EPSILON),
                                       {0.0f, 0.0f, -0.479426f, 0.479426f}));
  EXPECT_THAT(mhlo::sin(t4), Pointwise(FloatNear(EPSILON),
                                       {0.0f, 0.0f, -0.479426f, 0.479426f}));
}

TEST(mhlo, sqrt) {
  EXPECT_NEAR(3.0f, mhlo::sqrt(9.0f), EPSILON);

  Tensor0D<float> t0{4.0f};
  Tensor1D<float, 2> t1{0.0f, 81.0f};
  Tensor2D<double, 2, 2> t2{2.0, 3.0, 10.0, 1.0};
  Tensor3D<double, 2, 1, 2> t3{2.0, 3.0, 10.0, 1.0};
  Tensor4D<double, 2, 2, 1, 2> t4{2.0, 3.0, 10.0, 1.0, 18.0, 9.0, 5.0, 25.0};

  EXPECT_THAT(mhlo::sqrt(t0), Pointwise(FloatNear(EPSILON), {2.0f}));
  EXPECT_THAT(mhlo::sqrt(t1), Pointwise(FloatNear(EPSILON), {0.0f, 9.0f}));
  EXPECT_THAT(
      mhlo::sqrt(t2),
      Pointwise(FloatNear(EPSILON), {1.414213f, 1.732050f, 3.162277f, 1.0f}));
  EXPECT_THAT(
      mhlo::sqrt(t3),
      Pointwise(FloatNear(EPSILON), {1.414213f, 1.732050f, 3.162277f, 1.0f}));
  EXPECT_THAT(mhlo::sqrt(t4), Pointwise(FloatNear(EPSILON),
                                        {1.414213f, 1.732050f, 3.162277f, 1.0f,
                                         4.242640f, 3.0f, 2.236067f, 5.0f}));
}

TEST(mhlo, tanh) {
  EXPECT_NEAR(0.0f, mhlo::tanh(0.0f), EPSILON);

  Tensor0D<float> t0{0.0f};
  Tensor1D<float, 2> t1{0.0f, 1.0f};
  Tensor2D<double, 2, 2> t2{0.0, 1.0, -1.0, 0.0};
  Tensor3D<float, 1, 2, 2> t3{0.0f, 1.0f, -1.0f, 0.0f};
  Tensor4D<double, 3, 1, 1, 2> t4{0.0, 1.0, -1.0, 0.0f, M_PIf64, -M_2_PIf64};

  EXPECT_THAT(mhlo::tanh(t0), Pointwise(FloatNear(EPSILON), {0.0f}));
  EXPECT_THAT(mhlo::tanh(t1), Pointwise(FloatNear(EPSILON), {0.0f, 0.761594f}));
  EXPECT_THAT(mhlo::tanh(t2), Pointwise(FloatNear(EPSILON),
                                        {0.0f, 0.761594f, -0.761594f, 0.0f}));
  EXPECT_THAT(mhlo::tanh(t3), Pointwise(FloatNear(EPSILON),
                                        {0.0f, 0.761594f, -0.761594f, 0.0f}));
  EXPECT_THAT(mhlo::tanh(t4),
              Pointwise(FloatNear(EPSILON), {0.0f, 0.761594f, -0.761594f, 0.0f,
                                             0.996272f, -0.562593f}));
}

// Binary elementwise ops

TEST(mhlo, add) {
  EXPECT_EQ(2, mhlo::add(-1, 3));

  Tensor0D<int> s0{-3};
  Tensor0D<int> t0{8};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return mhlo::add<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {5}));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};
  Tensor1D<float, 2> t1{0.2f, -3.7f};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<float, 2> {
    return mhlo::add<Tensor1D<float, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(FloatEq(), {-1.1f, -1.3f}));

  Tensor2D<long, 2, 2> s2{3, 1, 4, 9};
  Tensor2D<long, 2, 2> t2{-2, 8, 6, -10};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return mhlo::add<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {1, 9, 10, -1}));

  Tensor3D<int16_t, 2, 1, 2> s3{3, 1, 4, 9};
  Tensor3D<int16_t, 2, 1, 2> t3{-2, 8, 6, -10};

  auto lambda_3d = [&s3, &t3]() -> Tensor3D<int16_t, 2, 1, 2> {
    return mhlo::add<Tensor3D<int16_t, 2, 1, 2>>(s3, t3);
  };

  EXPECT_THAT(lambda_3d(), Pointwise(Eq(), {1, 9, 10, -1}));

  Tensor4D<int8_t, 2, 1, 2, 2> s4{3, 1, 4, 9, 10, 11, 0, 1};
  Tensor4D<int8_t, 2, 1, 2, 2> t4{-2, 8, 6, -10, -10, 11, 1, 1};

  auto lambda_4d = [&s4, &t4]() -> Tensor4D<int8_t, 2, 1, 2, 2> {
    return mhlo::add<Tensor4D<int8_t, 2, 1, 2, 2>>(s4, t4);
  };

  EXPECT_THAT(lambda_4d(), Pointwise(Eq(), {1, 9, 10, -1, 0, 22, 1, 2}));
}

TEST(mhlo, atan2) {
  EXPECT_NEAR(0.321751f, mhlo::atan2(1.0f, 3.0f), EPSILON);

  Tensor0D<float> s0{1.0f};
  Tensor0D<float> t0{3.0f};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<float> {
    return mhlo::atan2<Tensor0D<float>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(FloatNear(EPSILON), {0.321751f}));

  Tensor1D<float, 2> s1{1.0f, 0.5f};
  Tensor1D<float, 2> t1{3.0f, -0.5f};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<float, 2> {
    return mhlo::atan2<Tensor1D<float, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(),
              Pointwise(FloatNear(EPSILON), {0.321751f, 2.35619f}));

  Tensor2D<double, 2, 2> s2{1.0, 0.5, -0.5, 0.5};
  Tensor2D<double, 2, 2> t2{3.0, -0.5, 0.5, 0.5};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<double, 2, 2> {
    return mhlo::atan2<Tensor2D<double, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(FloatNear(EPSILON),
                                     {0.321751, 2.35619, -0.785398, 0.785398}));

  Tensor3D<double, 1, 2, 2> s3{1.0, 0.5, -0.5, 0.5};
  Tensor3D<double, 1, 2, 2> t3{3.0, -0.5, 0.5, 0.5};

  auto lambda_3d = [&s3, &t3]() -> Tensor3D<double, 1, 2, 2> {
    return mhlo::atan2<Tensor3D<double, 1, 2, 2>>(s3, t3);
  };

  EXPECT_THAT(lambda_3d(), Pointwise(FloatNear(EPSILON),
                                     {0.321751, 2.35619, -0.785398, 0.785398}));

  Tensor4D<float, 1, 3, 1, 2> s4{1.0f, 0.5f, -0.5f, 0.5f, M_PIf32, 0.0f};
  Tensor4D<float, 1, 3, 1, 2> t4{3.0f, -0.5f, 0.5f, 0.5f, 0.0f, -M_PIf32};

  auto lambda_4d = [&s4, &t4]() -> Tensor4D<float, 1, 3, 1, 2> {
    return mhlo::atan2<Tensor4D<float, 1, 3, 1, 2>>(s4, t4);
  };

  EXPECT_THAT(lambda_4d(),
              Pointwise(FloatNear(EPSILON), {0.321751f, 2.35619f, -0.785398f,
                                             0.785398f, 1.570796f, M_PIf32}));
}

TEST(mhlo, div) {
  EXPECT_EQ(-3, mhlo::div(-3, 1));
  EXPECT_EQ(-6.75, mhlo::div(27.0, -4.0));
  EXPECT_EQ(-6, mhlo::div<int>(27.0, -4.0));

  Tensor0D<int> s0{27};
  Tensor0D<int> t0{-4};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return mhlo::div<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {-6}));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};
  Tensor1D<float, 2> t1{0.2f, -3.7f};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<float, 2> {
    return mhlo::div<Tensor1D<float, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(FloatNear(EPSILON), {-6.5f, -0.6486f}));

  Tensor2D<long, 2, 2> s2{3, 14, -31, -51};
  Tensor2D<long, 2, 2> t2{-2, 2, 6, 7};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return mhlo::div<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {-1, 7, -5, -7}));

  Tensor3D<int8_t, 2, 2, 1> s3{3, 14, -31, -51};
  Tensor3D<int8_t, 2, 2, 1> t3{-2, 2, 6, 7};

  auto lambda_3d = [&s3, &t3]() -> Tensor3D<int8_t, 2, 2, 1> {
    return mhlo::div<Tensor3D<int8_t, 2, 2, 1>>(s3, t3);
  };

  EXPECT_THAT(lambda_3d(), Pointwise(Eq(), {-1, 7, -5, -7}));

  Tensor4D<int16_t, 2, 1, 3, 1> s4{3, 14, -31, -51, 16, -2};
  Tensor4D<int16_t, 2, 1, 3, 1> t4{-2, 2, 6, 7, 8, -2};

  auto lambda_4d = [&s4, &t4]() -> Tensor4D<int16_t, 2, 1, 3, 1> {
    return mhlo::div<Tensor4D<int16_t, 2, 1, 3, 1>>(s4, t4);
  };

  EXPECT_THAT(lambda_4d(), Pointwise(Eq(), {-1, 7, -5, -7, 2, 1}));
}

TEST(mhlo, max) {
  EXPECT_EQ(3, mhlo::max(-1, 3));

  Tensor0D<int> s0{-3};
  Tensor0D<int> t0{8};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return mhlo::max<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {8}));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};
  Tensor1D<float, 2> t1{0.2f, -3.7f};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<float, 2> {
    return mhlo::max<Tensor1D<float, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(FloatEq(), {0.2f, 2.4f}));

  Tensor2D<long, 2, 2> s2{3, 1, 4, 9};
  Tensor2D<long, 2, 2> t2{-2, 8, 6, -10};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return mhlo::max<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {3, 8, 6, 9}));
}

TEST(mhlo, min) {
  EXPECT_EQ(-1, mhlo::min(-1, 3));

  Tensor0D<int> s0{-3};
  Tensor0D<int> t0{8};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return mhlo::min<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {-3}));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};
  Tensor1D<float, 2> t1{0.2f, -3.7f};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<float, 2> {
    return mhlo::min<Tensor1D<float, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(FloatEq(), {-1.3f, -3.7f}));

  Tensor2D<long, 2, 2> s2{3, 1, 4, 9};
  Tensor2D<long, 2, 2> t2{-2, 8, 6, -10};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return mhlo::min<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {-2, 1, 4, -10}));
}

TEST(mhlo, mul) {
  EXPECT_EQ(-3, mhlo::mul(-1, 3));

  Tensor0D<int> s0{-3};
  Tensor0D<int> t0{8};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return mhlo::mul<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {-24}));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};
  Tensor1D<float, 2> t1{0.2f, -3.7f};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<float, 2> {
    return mhlo::mul<Tensor1D<float, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(FloatEq(), {-0.26f, -8.88f}));

  Tensor2D<long, 2, 2> s2{3, 1, 4, 9};
  Tensor2D<long, 2, 2> t2{-2, 8, 6, -10};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return mhlo::mul<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {-6, 8, 24, -90}));
}

TEST(mhlo, pow) {
  EXPECT_EQ(9, mhlo::pow(3, 2));

  Tensor0D<int> s0{2};
  Tensor0D<int> t0{4};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return mhlo::pow<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {16}));

  Tensor1D<float, 2> s1{4.0f, 2.0f};
  Tensor1D<float, 2> t1{0.5f, -2.0f};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<float, 2> {
    return mhlo::pow<Tensor1D<float, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(FloatNear(EPSILON), {2.0f, 0.25f}));

  Tensor2D<long, 2, 2> s2{3, 1, 4, 2};
  Tensor2D<long, 2, 2> t2{0, -1, 3, -2};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return mhlo::pow<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {1, 1, 64, 0}));
}

TEST(mhlo, shift_left) {
  EXPECT_EQ(16u, mhlo::shift_left(2u, 3u));

  Tensor0D<uint> s0{2};
  Tensor0D<uint> t0{8};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<uint> {
    return mhlo::shift_left<Tensor0D<uint>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {512}));

  Tensor1D<uint8_t, 2> s1{3, 0};
  Tensor1D<uint8_t, 2> t1{2, 3};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<uint8_t, 2> {
    return mhlo::shift_left<Tensor1D<uint8_t, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(Eq(), {12, 0}));

  Tensor2D<uint64_t, 2, 2> s2{0, 2, 5, 10};
  Tensor2D<uint64_t, 2, 2> t2{0, 1, 3, 4};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<uint64_t, 2, 2> {
    return mhlo::shift_left<Tensor2D<uint64_t, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {0, 4, 40, 160}));
}

TEST(mhlo, shift_right_logical) {
  EXPECT_EQ(2u, mhlo::shift_right_logical(4u, 1u));

  Tensor0D<uint> s0{6};
  Tensor0D<uint> t0{2};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<uint> {
    return mhlo::shift_right_logical<Tensor0D<uint>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {1}));

  Tensor1D<uint8_t, 2> s1{17, 32};
  Tensor1D<uint8_t, 2> t1{1, 3};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<uint8_t, 2> {
    return mhlo::shift_right_logical<Tensor1D<uint8_t, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(Eq(), {8, 4}));

  Tensor2D<uint64_t, 2, 2> s2{0, 2, 25, 10};
  Tensor2D<uint64_t, 2, 2> t2{0, 1, 3, 2};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<uint64_t, 2, 2> {
    return mhlo::shift_right_logical<Tensor2D<uint64_t, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {0, 1, 3, 2}));
}

TEST(mhlo, sub) {
  EXPECT_EQ(-4, mhlo::sub(-1, 3));

  Tensor0D<int> s0{-3};
  Tensor0D<int> t0{8};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return mhlo::sub<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {-11}));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};
  Tensor1D<float, 2> t1{0.2f, -3.7f};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<float, 2> {
    return mhlo::sub<Tensor1D<float, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(FloatEq(), {-1.5f, 6.1f}));

  Tensor2D<long, 2, 2> s2{3, 1, 4, 9};
  Tensor2D<long, 2, 2> t2{-2, 8, 6, -10};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return mhlo::sub<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {5, -7, -2, 19}));
}

// Binary logical elementwise ops

TEST(mhlo, or) {
  EXPECT_EQ(1, mhlo::logical_or(2, 3));

  Tensor0D<int> s0{2};
  Tensor0D<int> t0{8};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return mhlo::logical_or<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {1}));

  Tensor1D<int8_t, 2> s1{-1, 0};
  Tensor1D<int8_t, 2> t1{0, 0};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<int8_t, 2> {
    return mhlo::logical_or<Tensor1D<int8_t, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(Eq(), {1, 0}));

  Tensor2D<long, 2, 2> s2{0, 2, 0, -1};
  Tensor2D<long, 2, 2> t2{0, 0, -2, -2};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return mhlo::logical_or<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {0, 1, 1, 1}));
}

TEST(mhlo, xor) {
  EXPECT_EQ(1, mhlo::logical_xor(2, 3));

  Tensor0D<int> s0{2};
  Tensor0D<int> t0{8};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return mhlo::logical_xor<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {1}));

  Tensor1D<int8_t, 2> s1{-1, 0};
  Tensor1D<int8_t, 2> t1{0, 0};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<int8_t, 2> {
    return mhlo::logical_xor<Tensor1D<int8_t, 2>>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(Eq(), {1, 0}));

  Tensor2D<long, 2, 2> s2{0, 2, 0, -1};
  Tensor2D<long, 2, 2> t2{0, 0, -2, -2};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return mhlo::logical_xor<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {0, 1, 1, 1}));
}

// Tuple ops

TEST(mhlo, compare) {
  auto lambda = []() { return mhlo::compare<int, std::less>(-1, 3); };
  EXPECT_EQ(true, lambda());

  Tensor0D<int> s0{-3};
  Tensor0D<int> t0{-8};

  auto lambda_0d = [&s0, &t0]() {
    return mhlo::compare<Tensor0D<int>, std::less_equal>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {false}));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};
  Tensor1D<float, 2> t1{0.2f, 2.4f};

  auto lambda_1d = [&s1, &t1]() {
    return mhlo::compare<Tensor1D<float, 2>, std::equal_to>(s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(Eq(), {false, true}));

  Tensor2D<long, 2, 2> s2{3, 1, 4, 9};
  Tensor2D<long, 2, 2> t2{-2, 1, 6, -10};

  auto lambda_2d = [&s2, &t2]() {
    return mhlo::compare<Tensor2D<long, 2, 2>, std::greater_equal>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {true, true, false, true}));

  Tensor3D<long, 2, 1, 2> s3{3, 1, 4, 9};
  Tensor3D<long, 2, 1, 2> t3{-2, 1, 6, -10};

  auto lambda_3d = [&s3, &t3]() {
    return mhlo::compare<Tensor3D<long, 2, 1, 2>, std::greater_equal>(s3, t3);
  };

  EXPECT_THAT(lambda_3d(), Pointwise(Eq(), {true, true, false, true}));

  Tensor4D<int, 2, 1, 2, 2> s4{3, 1, 4, 9, 10, 12, -4, 8};
  Tensor4D<int, 2, 1, 2, 2> t4{-2, 1, 6, -10, 9, 13, 4, 10};

  auto lambda_4d = [&s4, &t4]() {
    return mhlo::compare<Tensor4D<int, 2, 1, 2, 2>, std::greater_equal>(s4, t4);
  };

  EXPECT_THAT(lambda_4d(), Pointwise(Eq(), {true, true, false, true, true,
                                            false, false, false}));
}

// Slice ops

TEST(mhlo, slice) {
  Tensor1D<float, 5> s1{0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  auto t1 =
      mhlo::slice<Tensor1D<float, 2>, Tensor1D<float, 5>>(s1, {2}, {4}, {1});
  EXPECT_THAT(t1, Pointwise(FloatEq(), {2.0f, 3.0f}));

  auto t1_strided =
      mhlo::slice<Tensor1D<float, 2>, Tensor1D<float, 5>>(s1, {1}, {4}, {2});
  EXPECT_THAT(t1_strided, Pointwise(FloatEq(), {1.0f, 3.0f}));

  Tensor2D<float, 4, 3> s2{0.0f, 1.0f, 2.0f, 3.0f, 4.0f,  5.0f,
                           6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  auto t2 = mhlo::slice<Tensor2D<float, 2, 2>, Tensor2D<float, 4, 3>>(
      s2, {2, 1}, {4, 3}, {1, 1});

  EXPECT_THAT(t2, Pointwise(FloatEq(), {7.0f, 8.0f, 10.0f, 11.0f}));

  auto t2_strided = mhlo::slice<Tensor2D<float, 2, 2>, Tensor2D<float, 4, 3>>(
      s2, {1, 0}, {4, 3}, {2, 2});

  EXPECT_THAT(t2_strided, Pointwise(FloatEq(), {3.0f, 5.0f, 9.0f, 11.0f}));
}

TEST(mhlo, dynamic_slice) {
  Tensor1D<float, 5> s1{0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  auto t1 =
      mhlo::dynamic_slice<Tensor1D<float, 2>, Tensor1D<float, 5>>(s1, {2}, {2});
  EXPECT_THAT(t1, Pointwise(FloatEq(), {2.0f, 3.0f}));

  Tensor2D<float, 4, 3> s2{0.0f, 1.0f, 2.0f, 3.0f, 4.0f,  5.0f,
                           6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  auto t2 = mhlo::dynamic_slice<Tensor2D<float, 2, 2>, Tensor2D<float, 4, 3>>(
      s2, {2}, {1}, {2, 2});

  EXPECT_THAT(t2, Pointwise(FloatEq(), {7.0f, 8.0f, 10.0f, 11.0f}));
}

TEST(mhlo, dynamic_update_slice) {
  Tensor1D<float, 5> s1{0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  Tensor1D<float, 2> u1{5.0f, 6.0f};
  auto t1 = mhlo::dynamic_update_slice<Tensor1D<float, 2>, Tensor1D<float, 5>>(
      s1, u1, {2});
  EXPECT_THAT(t1, Pointwise(FloatEq(), {0.0f, 1.0f, 5.0f, 6.0f, 4.0f}));

  Tensor2D<float, 4, 3> s2{0.0f, 1.0f, 2.0f, 3.0f, 4.0f,  5.0f,
                           6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  Tensor2D<float, 3, 2> u2{12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f};
  auto t2 =
      mhlo::dynamic_update_slice<Tensor2D<float, 3, 2>, Tensor2D<float, 4, 3>>(
          s2, u2, {1}, {1});

  EXPECT_THAT(t2,
              Pointwise(FloatEq(), {0.0f, 1.0f, 2.0f, 3.0f, 12.0f, 13.0f, 6.0f,
                                    14.0f, 15.0f, 9.0f, 16.0f, 17.0f}));
}

// Other ops

TEST(mhlo, batch_norm_inference) {
  Tensor<float, 4, 2> input{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  Tensor<float, 4, 2> expected_result{-3.0f, 2.0f,  1.0f, 6.0f,
                                      5.0f,  10.0f, 9.0f, 14.0f};
  Tensor<float, 4, 2> result =
      mhlo::batch_norm_inference<Tensor<float, 4, 2>, Tensor<float, 2>>(
          input, {1.0f, 2.0f}, {1.0f, 2.0f}, {2.0f, 1.0f}, {0.249f, 0.999f},
          0.001f, 1);

  EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
}

TEST(mhlo, bitcast_convert) {
  uint8_t a = 128;
  int8_t b = -128;
  EXPECT_EQ(b, mhlo::bitcast_convert<int8_t>(a));

  Tensor0D<int16_t> t0{-1};
  auto lambda_0d = [&t0]() {
    return mhlo::bitcast_convert<Tensor0D<uint16_t>>(t0);
  };
  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {65535}));

  Tensor1D<uint16_t, 2> t1{1, 2};
  auto lambda_1d = [&t1]() {
    return mhlo::bitcast_convert<Tensor1D<int16_t, 2>>(t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(Eq(), {1, 2}));

  Tensor2D<int8_t, 2, 2> t2{0, -4, 3, -12};
  auto lambda_2d = [&t2]() {
    return mhlo::bitcast_convert<Tensor2D<uint8_t, 2, 2>>(t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(DoubleEq(), {0, 252, 3, 244}));

  Tensor3D<int8_t, 2, 1, 2> t3{0, -4, 3, -12};
  auto lambda_3d = [&t3]() {
    return mhlo::bitcast_convert<Tensor3D<uint8_t, 2, 1, 2>>(t3);
  };

  EXPECT_THAT(lambda_3d(), Pointwise(DoubleEq(), {0, 252, 3, 244}));

  Tensor4D<int8_t, 2, 1, 2, 2> t4{0, -4, 3, -12, -11, 0, 2, -4};
  auto lambda_4d = [&t4]() {
    return mhlo::bitcast_convert<Tensor4D<uint8_t, 2, 1, 2, 2>>(t4);
  };

  EXPECT_THAT(lambda_4d(),
              Pointwise(DoubleEq(), {0, 252, 3, 244, 245, 0, 2, 252}));
}

TEST(mhlo, broadcast_in_dim) {
  Tensor0D<int> t0{1};
  Tensor1D<int64_t, 0> b0;

  Tensor1D<int, 4> expected_result1{1, 1, 1, 1};
  Tensor1D<int, 4> result1 = mhlo::broadcast_in_dim<Tensor1D<int, 4>>(t0, b0);
  EXPECT_THAT(result1, Pointwise(Eq(), expected_result1));

  Tensor2D<int, 2, 3> expected_result2{1, 1, 1, 1, 1, 1};
  Tensor2D<int, 2, 3> result2 =
      mhlo::broadcast_in_dim<Tensor2D<int, 2, 3>>(t0, b0);
  EXPECT_THAT(result2, Pointwise(Eq(), expected_result2));

  // TODO add more tests once higher dimensions are supported
}

TEST(mhlo, clamp) {
  Tensor<float, 2, 1> operand{-1.0f, 1.0f};
  Tensor<float, 2, 1> min{0.0f, 0.0f};
  Tensor<float, 2, 1> max{3.0f, 0.0f};
  Tensor<float, 2, 1> expected_result{0.0, 0.0f};
  Tensor<float, 2, 1> result = mhlo::clamp(min, operand, max);

  EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));

  // broadcasting
  Tensor<int32_t, 4, 2, 1> operand_b{0, 1, 2, 3, 4, 5, 6, 7};
  Tensor<int32_t> min_b{2};
  Tensor<int32_t> max_b{5};
  Tensor<int32_t, 4, 2, 1> expected_result_b{2, 2, 2, 3, 4, 5, 5, 5};
  Tensor<int32_t, 4, 2, 1> result_b = mhlo::clamp(min_b, operand_b, max_b);

  EXPECT_THAT(result_b, Pointwise(Eq(), expected_result_b));
}

TEST(mhlo, concatenate) {
  Tensor1D<int, 1> t1{1};
  Tensor1D<int, 2> t2{2, 3};
  Tensor1D<int, 3> t3{4, 5, 6};

  auto lambda_1d_1 = [&t1]() -> Tensor1D<int, 1> {
    return mhlo::concatenate<0, Tensor1D<int, 1>, Tensor1D<int, 1>>(t1);
  };

  EXPECT_THAT(lambda_1d_1(), Pointwise(Eq(), {1}));

  auto lambda_1d_2 = [&t1, &t2]() -> Tensor1D<int, 3> {
    return mhlo::concatenate<0, Tensor1D<int, 3>, Tensor1D<int, 1>,
                             Tensor1D<int, 2>>(t1, t2);
  };

  EXPECT_THAT(lambda_1d_2(), Pointwise(Eq(), {1, 2, 3}));

  auto lambda_1d_3 = [&t1, &t2, &t3]() -> Tensor1D<int, 6> {
    return mhlo::concatenate<0, Tensor1D<int, 6>, Tensor1D<int, 1>,
                             Tensor1D<int, 2>, Tensor1D<int, 3>>(t1, t2, t3);
  };

  EXPECT_THAT(lambda_1d_3(), Pointwise(Eq(), {1, 2, 3, 4, 5, 6}));

  Tensor2D<float, 1, 2> t4{1.0f, 2.0f};
  Tensor2D<float, 2, 2> t5{3.0f, 4.0f, 5.0f, 6.0f};
  Tensor2D<float, 3, 2> t6{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  auto lambda_2d_2_row = [&t4, &t5]() -> Tensor2D<float, 3, 2> {
    return mhlo::concatenate<0, Tensor2D<float, 3, 2>, Tensor2D<float, 1, 2>,
                             Tensor2D<float, 2, 2>>(t4, t5);
  };

  EXPECT_THAT(lambda_2d_2_row(),
              Pointwise(FloatEq(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));

  auto lambda_2d_2_col = [&t4, &t5]() -> Tensor2D<float, 2, 3> {
    Tensor2D<float, 2, 1> t4_reshape = mhlo::reshape<Tensor2D<float, 2, 1>>(t4);
    return mhlo::concatenate<1, Tensor2D<float, 2, 3>, Tensor2D<float, 2, 1>,
                             Tensor2D<float, 2, 2>>(t4_reshape, t5);
  };

  EXPECT_THAT(lambda_2d_2_col(),
              Pointwise(FloatEq(), {1.0f, 3.0f, 4.0f, 2.0f, 5.0f, 6.0f}));

  auto lambda_2d_3_row = [&t4, &t5, &t6]() -> Tensor2D<float, 6, 2> {
    return mhlo::concatenate<0, Tensor2D<float, 6, 2>, Tensor2D<float, 1, 2>,
                             Tensor2D<float, 2, 2>, Tensor2D<float, 3, 2>>(
        t4, t5, t6);
  };

  EXPECT_THAT(lambda_2d_3_row(),
              Pointwise(FloatEq(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                                    8.0f, 9.0f, 10.0f, 11.0f, 12.0f}));

  auto lambda_2d_3_col = [&t4, &t5, &t6]() -> Tensor2D<float, 2, 6> {
    Tensor2D<float, 2, 1> t4_reshape = mhlo::reshape<Tensor2D<float, 2, 1>>(t4);
    Tensor2D<float, 2, 3> t6_reshape = mhlo::reshape<Tensor2D<float, 2, 3>>(t6);
    return mhlo::concatenate<1, Tensor2D<float, 2, 6>, Tensor2D<float, 2, 1>,
                             Tensor2D<float, 2, 2>, Tensor2D<float, 2, 3>>(
        t4_reshape, t5, t6_reshape);
  };

  EXPECT_THAT(lambda_2d_3_col(),
              Pointwise(FloatEq(), {1.0f, 3.0f, 4.0f, 7.0f, 8.0f, 9.0f, 2.0f,
                                    5.0f, 6.0f, 10.0f, 11.0f, 12.0f}));
}

/// Taken from
/// https://github.com/google/iree/blob/efd78a0b47a46457a644f43d98617d3e279b2a79/iree/test/e2e/xla_ops/convolution.mlir#L33
TEST(mhlo, convolution) {
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
  Tensor2D<int64_t, 2, 2> padding{1, 1, 0, 1}; // {pt, pb, pl, pr}
  Tensor1D<int64_t, 2> rhs_dilation{1, 1};
  Tensor1D<int64_t, 2> window_strides{1, 1};

  ResultType result = mhlo::convolution<ResultType, InputType, WeightType>(
      input, weights, batch_group_count, input_batch_dimension,
      input_feature_dimension, input_spatial_dimensions,
      kernel_input_feature_dimension, kernel_output_feature_dimension,
      kernel_spatial_dimensions, output_batch_dimension,
      output_feature_dimension, output_spatial_dimensions, feature_group_count,
      padding, rhs_dilation, window_strides);

  EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
}

TEST(mhlo, dot) {
  Tensor2D<int, 2, 2> a2{1, 0, 0, 1};
  Tensor2D<int, 2, 2> b2{4, 1, 2, 2};

  auto lambda_2d = [&a2, &b2]() -> Tensor2D<int, 2, 2> {
    return mhlo::dot<Tensor2D<int, 2, 2>>(a2, b2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {4, 1, 2, 2}));
}

TEST(mhlo, reshape) {
  Tensor0D<int> s0{-3};
  auto t0 = mhlo::reshape<Tensor1D<int, 1>>(s0);
  auto t0_1 = mhlo::reshape<Tensor2D<int, 1, 1>>(s0);
  EXPECT_THAT(t0, Pointwise(Eq(), {-3}));
  EXPECT_THAT(t0_1, Pointwise(Eq(), {-3}));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};
  auto t1 = mhlo::reshape<Tensor2D<float, 1, 2>>(s1);

  EXPECT_THAT(t1, Pointwise(FloatEq(), {-1.3f, 2.4f}));

  Tensor2D<long, 2, 2> s2{3, 1, 4, 9};
  auto t2 = mhlo::reshape<Tensor1D<long, 4>>(s2);

  EXPECT_THAT(t2, Pointwise(Eq(), {3, 1, 4, 9}));
}

TEST(mhlo, select) {
  EXPECT_EQ(-1, mhlo::select(true, -1, 3));
  EXPECT_EQ(3, mhlo::select(false, -1, 3));

  Tensor0D<int> s0{-3};
  Tensor0D<int> t0{8};
  Tensor0D<bool> p0{true};

  auto lambda_0d = [&p0, &s0, &t0]() -> Tensor0D<int> {
    return mhlo::select<Tensor0D<int>>(p0, s0, t0);
  };

  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {-3}));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};
  Tensor1D<float, 2> t1{0.2f, -3.7f};
  Tensor1D<bool, 2> p1{true, false};

  auto lambda_1d = [&p1, &s1, &t1]() -> Tensor1D<float, 2> {
    return mhlo::select<Tensor1D<float, 2>>(p1, s1, t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(FloatEq(), {-1.3f, -3.7f}));

  Tensor2D<long, 2, 2> s2{3, 1, 4, 9};
  Tensor2D<long, 2, 2> t2{-2, 8, 6, -10};
  Tensor2D<bool, 2, 2> p2{false, true, true, false};

  auto lambda_2d = [&p2, &s2, &t2]() -> Tensor2D<long, 2, 2> {
    return mhlo::select<Tensor2D<long, 2, 2>>(p2, s2, t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {-2, 1, 4, -10}));
}

} // namespace
