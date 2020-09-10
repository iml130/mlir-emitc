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

using ::testing::ElementsAre;

TEST(mhlo, abs) {
  EXPECT_EQ(1, mhlo::abs(-1));

  Tensor0D<int> t0{-1};
  Tensor1D<float, 2> t1{-1.0f, -2.0f};
  Tensor2D<long, 2, 2> t2{-2, -1, 0, 2};

  EXPECT_THAT(mhlo::abs(t0), ElementsAre(1));
  EXPECT_THAT(mhlo::abs(t1), ElementsAre(1.0f, 2.0f));
  EXPECT_THAT(mhlo::abs(t2), ElementsAre(2, 1, 0, 2));

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
  EXPECT_NEAR(0.8775, mhlo::cos(0.5), 5e-4);
  EXPECT_EQ(0, mhlo::cos(1));

  // TODO: Check vector
}

TEST(mhlo, sin) {
  EXPECT_EQ(0, mhlo::sin(0));
  EXPECT_NEAR(0.4795, mhlo::sin(0.5), 5e-4);
  EXPECT_NEAR(1, mhlo::sin(1.57), 5e-4);

  // TODO: Check vector
}

TEST(mhlo, sqrt) {
  EXPECT_EQ(3, mhlo::sqrt(9));
  EXPECT_EQ(2.0f, mhlo::sqrt(4.0f));

  std::vector<float> v1 = {4.0, 9.0};
  EXPECT_THAT(mhlo::sqrt(v1), ElementsAre(2.0, 3.0));
}

TEST(mhlo, add) {
  EXPECT_EQ(2, mhlo::add(-1, 3));

  Tensor0D<int> s0{-3};
  Tensor0D<int> t0{8};

  auto lambda_0d = [&s0, &t0]() -> Tensor0D<int> {
    return mhlo::add<Tensor0D<int>>(s0, t0);
  };

  EXPECT_THAT(lambda_0d(), ElementsAre(5));

  Tensor1D<float, 2> s1{-1.3f, 2.4f};
  Tensor1D<float, 2> t1{0.2f, -3.7f};

  auto lambda_1d = [&s1, &t1]() -> Tensor1D<float, 2> {
    return mhlo::add<Tensor1D<float, 2>>(s1, t1);
  };

  // TODO use floating point compare
  // EXPECT_THAT(lambda_1d(), ElementsAre(-1.1f, -1.3f));

  Tensor2D<long, 2, 2> s2{3, 1, 4, 9};
  Tensor2D<long, 2, 2> t2{-2, 8, 6, -10};

  auto lambda_2d = [&s2, &t2]() -> Tensor2D<long, 2, 2> {
    return mhlo::add<Tensor2D<long, 2, 2>>(s2, t2);
  };

  EXPECT_THAT(lambda_2d(), ElementsAre(1, 9, 10, -1));
}

TEST(mhlo, div) {
  EXPECT_EQ(2.5, mhlo::div(5.0, 2.0));

  std::vector<float> v1 = {5.0f, -5.0f};
  std::vector<float> v2 = {2.0f, 2.0f};
  EXPECT_THAT(mhlo::div(v1, v2), ElementsAre(2.5f, -2.5f));
}

TEST(mhlo, max) {
  EXPECT_EQ(5.0, mhlo::max(5.0, 2.0));

  std::vector<float> v1 = {5.0f, -5.0f};
  std::vector<float> v2 = {2.0f, 2.0f};
  EXPECT_THAT(mhlo::max(v1, v2), ElementsAre(5.0f, 2.0f));
}

TEST(mhlo, min) {
  EXPECT_EQ(2.0, mhlo::min(5.0, 2.0));

  std::vector<float> v1 = {5.0f, -5.0f};
  std::vector<float> v2 = {2.0f, 2.0f};
  EXPECT_THAT(mhlo::min(v1, v2), ElementsAre(2.0f, -5.0f));
}

TEST(mhlo, mul) {
  EXPECT_EQ(1, mhlo::mul(-1, -1));

  std::vector<int> v1 = {-1, -2};
  EXPECT_THAT(mhlo::mul(v1, v1), ElementsAre(1, 4));
}

TEST(mhlo, pow) {
  EXPECT_EQ(9, mhlo::pow(3, 2));
  EXPECT_EQ(4.0f, mhlo::pow(2.0f, 2));

  // TODO: Check vector
}

TEST(mhlo, sub) {
  EXPECT_EQ(1, mhlo::sub(2, 1));

  std::vector<int> v1 = {5, -2};
  std::vector<int> v2 = {2, 2};
  EXPECT_THAT(mhlo::sub(v1, v2), ElementsAre(3, -4));
}

TEST(mhlo, broadcast_in_dim) {
  std::vector<int> v1 = {1, 2};
  EXPECT_THAT(mhlo::broadcast_in_dim(v1, 3), ElementsAre(1, 2, 1, 2, 1, 2));
}

TEST(mhlo, concatenate) {
  std::vector<int> v1 = {1, 2};
  std::vector<int> v2 = {3, 4};
  EXPECT_THAT(mhlo::concatenate(v1, v2), ElementsAre(1, 2, 3, 4));
}

} // namespace
