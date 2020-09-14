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
  EXPECT_EQ(1, mhlo::cos(0));
  EXPECT_NEAR(0.8775, mhlo::cos(0.5), EPSILON);
  EXPECT_EQ(0, mhlo::cos(1));

  // TODO: Check vector
}

TEST(mhlo, sin) {
  EXPECT_EQ(0, mhlo::sin(0));
  EXPECT_NEAR(0.4795, mhlo::sin(0.5), EPSILON);
  EXPECT_NEAR(1, mhlo::sin(1.57), EPSILON);

  // TODO: Check vector
}

TEST(mhlo, sqrt) {
  EXPECT_EQ(3, mhlo::sqrt(9));
  EXPECT_EQ(2.0f, mhlo::sqrt(4.0f));

  std::vector<float> v1 = {4.0, 9.0};
  EXPECT_THAT(mhlo::sqrt(v1), Pointwise(Eq(), {2.0, 3.0}));
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
  EXPECT_EQ(4.0f, mhlo::pow(2.0f, 2));

  // TODO: Check vector
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
