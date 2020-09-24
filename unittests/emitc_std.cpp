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

#include "emitc/emitc_std.h"
#include "emitc/emitc_tensor.h"

namespace {

using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::Pointwise;

TEST(std, extract_element) {
  Tensor0D<float> t0{1.0};
  Tensor1D<int32_t, 2> t1{1, 2};
  Tensor2D<uint16_t, 1, 4> t2{10, 11, 12, 13};
  Tensor2D<uint16_t, 2, 2> t3{10, 11, 12, 13};

  EXPECT_EQ(1.0, standard::extract_element(t0));
  EXPECT_EQ(2, standard::extract_element(t1, 1));
  EXPECT_EQ(12, standard::extract_element(t2, 0, 2));
  EXPECT_EQ(12, standard::extract_element(t3, 1, 0));
}

TEST(std, index_cast) {
  uint32_t a = 1;
  uint64_t b = 1;
  EXPECT_EQ(b, standard::index_cast<uint64_t>(a));

  Tensor0D<uint32_t> t0{1};
  auto lambda_0d = [&t0]() -> Tensor0D<size_t> {
    return standard::index_cast<Tensor0D<size_t>>(t0);
  };
  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {1}));

  Tensor1D<uint16_t, 2> t1{1, 2};
  auto lambda_1d = [&t1]() -> Tensor1D<size_t, 2> {
    return standard::index_cast<Tensor1D<size_t, 2>>(t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(Eq(), {1, 2}));

  Tensor2D<size_t, 2, 2> t2{1, 2, 4, 8};
  auto lambda_2d = [&t2]() -> Tensor2D<int8_t, 2, 2> {
    return standard::index_cast<Tensor2D<int8_t, 2, 2>>(t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {1, 2, 4, 8}));
}

TEST(std, splat) {
  EXPECT_THAT(standard::splat<Tensor0D<uint32_t>>(1), Pointwise(Eq(), {1}));

  auto lambda_1d = []() -> Tensor1D<int32_t, 3> {
    return standard::splat<Tensor1D<int32_t, 3>>(-1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(Eq(), {-1, -1, -1}));

  auto lambda_2d = []() -> Tensor2D<float, 2, 2> {
    return standard::splat<Tensor2D<float, 2, 2>>(1.5f);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(FloatEq(), {1.5f, 1.5f, 1.5f, 1.5f}));
}

} // namespace
