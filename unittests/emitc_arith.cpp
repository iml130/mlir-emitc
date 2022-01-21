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

#include "emitc/arith.h"
#include "emitc/types.h"

namespace {

using namespace emitc;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::Pointwise;

TEST(arith, index_cast) {
  uint32_t a = 1;
  uint64_t b = 1;
  EXPECT_EQ(b, arith::index_cast<uint64_t>(a));

  Tensor0D<uint32_t> t0{1};
  auto lambda_0d = [&t0]() -> Tensor0D<size_t> {
    return arith::index_cast<Tensor0D<size_t>>(t0);
  };
  EXPECT_THAT(lambda_0d(), Pointwise(Eq(), {1}));

  Tensor1D<uint16_t, 2> t1{1, 2};
  auto lambda_1d = [&t1]() -> Tensor1D<size_t, 2> {
    return arith::index_cast<Tensor1D<size_t, 2>>(t1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(Eq(), {1, 2}));

  Tensor2D<size_t, 2, 2> t2{1, 2, 4, 8};
  auto lambda_2d = [&t2]() -> Tensor2D<int8_t, 2, 2> {
    return arith::index_cast<Tensor2D<int8_t, 2, 2>>(t2);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {1, 2, 4, 8}));

  Tensor3D<size_t, 2, 1, 2> t3{1, 2, 4, 8};
  auto lambda_3d = [&t3]() -> Tensor3D<int8_t, 2, 1, 2> {
    return arith::index_cast<Tensor3D<int8_t, 2, 1, 2>>(t3);
  };

  EXPECT_THAT(lambda_3d(), Pointwise(Eq(), {1, 2, 4, 8}));

  Tensor4D<size_t, 2, 1, 2, 1> t4{1, 2, 4, 8};
  auto lambda_4d = [&t4]() -> Tensor4D<int8_t, 2, 1, 2, 1> {
    return arith::index_cast<Tensor4D<int8_t, 2, 1, 2, 1>>(t4);
  };

  EXPECT_THAT(lambda_4d(), Pointwise(Eq(), {1, 2, 4, 8}));
}

} // namespace
