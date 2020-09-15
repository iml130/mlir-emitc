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

#include "emitc_mhlo.h"
#include "emitc_tensor.h"

namespace {

using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pointwise;

const float EPSILON = 5e-4;

TEST(mhlo, abs) {
  EXPECT_EQ(1, mhlo::abs(-1));

  Tensor0D<int> t0{-1};
  Tensor1D<float, 2> t1{-1.0f, -2.0f};
  Tensor2D<long, 2, 2> t2{-2, -1, 0, 2};

  EXPECT_THAT(mhlo::abs(t0), Pointwise(Eq(), {1}));
  EXPECT_THAT(mhlo::abs(t1), Pointwise(Eq(), {1.0f, 2.0f}));
  EXPECT_THAT(mhlo::abs(t2), Pointwise(Eq(), {2, 1, 0, 2}));

  // TODO:: Test complex to real.
}

TEST(mhlo, convert) {
  uint32_t a = 1;
  uint64_t b = 1;
  EXPECT_EQ(b, mhlo::convert<uint64_t>(a));

  std::vector<uint32_t> v1 = {1, 2};
  std::vector<uint64_t> v2 = {1, 2};
  EXPECT_EQ(v2, mhlo::convert<uint64_t>(v1));
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

TEST(mhlo, sin) {
  EXPECT_NEAR(0.0f, mhlo::sin(0.0f), EPSILON);

  Tensor0D<float> t0{M_PIf32};
  Tensor1D<float, 2> t1{M_PI_2f32, -M_PI_2f32};
  Tensor2D<double, 2, 2> t2{2 * M_PIf32, 0.0f, -0.5f, 0.5f};

  EXPECT_THAT(mhlo::sin(t0), Pointwise(FloatNear(EPSILON), {0.0f}));
  EXPECT_THAT(mhlo::sin(t1), Pointwise(FloatNear(EPSILON), {1.0f, -1.0f}));
  EXPECT_THAT(mhlo::sin(t2), Pointwise(FloatNear(EPSILON),
                                       {0.0f, 0.0f, -0.479426f, 0.479426f}));
}

TEST(mhlo, sqrt) {
  EXPECT_NEAR(3.0f, mhlo::sqrt(9.0f), EPSILON);

  Tensor0D<float> t0{4.0f};
  Tensor1D<float, 2> t1{0.0f, 81.0f};
  Tensor2D<double, 2, 2> t2{2.0f, 3.0f, 10.0f, 1.0f};

  EXPECT_THAT(mhlo::sqrt(t0), Pointwise(FloatNear(EPSILON), {2.0f}));
  EXPECT_THAT(mhlo::sqrt(t1), Pointwise(FloatNear(EPSILON), {0.0f, 9.0f}));
  EXPECT_THAT(
      mhlo::sqrt(t2),
      Pointwise(FloatNear(EPSILON), {1.414213f, 1.732050f, 3.162277f, 1.0f}));
}

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
}

TEST(mhlo, div) {
  EXPECT_EQ(-3, mhlo::div(-3, 1));

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

TEST(mhlo, log) {
  EXPECT_NEAR(0.0f, mhlo::log(1.0f), EPSILON);

  Tensor0D<float> t0{M_Ef32};
  Tensor1D<float, 2> t1{M_Ef32 * M_Ef32, M_Ef32 * M_Ef32 * M_Ef32};
  Tensor2D<double, 2, 2> t2{1.0f, 2.0f, 3.0f, 4.0f};

  EXPECT_THAT(mhlo::log(t0), Pointwise(FloatNear(EPSILON), {1.0f}));
  EXPECT_THAT(mhlo::log(t1), Pointwise(FloatNear(EPSILON), {2.0f, 3.0f}));
  // clang-format off
  EXPECT_THAT(mhlo::log(t2), Pointwise(FloatNear(EPSILON), {0.0f, 0.693147f, 1.098612f, 1.386294f}));
  // clang-format on
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

TEST(mhlo, broadcast_in_dim) {
  std::vector<int> v1 = {1, 2};
  EXPECT_THAT(mhlo::broadcast_in_dim(v1, 3),
              Pointwise(Eq(), {1, 2, 1, 2, 1, 2}));
}

TEST(mhlo, concatenate) {
  std::vector<int> v1 = {1, 2};
  std::vector<int> v2 = {3, 4};
  EXPECT_THAT(mhlo::concatenate(v1, v2), Pointwise(Eq(), {1, 2, 3, 4}));
}

} // namespace
