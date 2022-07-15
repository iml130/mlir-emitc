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
//
// SPDX-License-Identifier: Apache-2.0

#include "gmock/gmock.h"

#include "emitc/arith.h"
#include "emitc/types.h"

namespace {

using namespace emitc;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::Pointwise;

TEST(arith, index_cast) {
  {
    uint32_t x = 1;
    uint64_t expected_result = 1;
    uint64_t result = arith::index_cast<uint64_t>(x);

    EXPECT_EQ(result, expected_result);
  }
  {
    Tensor0D<uint32_t> x{1};
    Tensor0D<size_t> expected_result{1};
    Tensor0D<size_t> result = arith::index_cast<Tensor0D<size_t>>(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor1D<uint16_t, 2> x{1, 2};
    Tensor1D<size_t, 2> expected_result{1, 2};
    Tensor1D<size_t, 2> result = arith::index_cast<Tensor1D<size_t, 2>>(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor2D<size_t, 2, 2> x{1, 2, 4, 8};
    Tensor2D<int8_t, 2, 2> expected_result{1, 2, 4, 8};
    Tensor2D<int8_t, 2, 2> result =
        arith::index_cast<Tensor2D<int8_t, 2, 2>>(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor3D<size_t, 2, 1, 2> x{1, 2, 4, 8};
    Tensor3D<int8_t, 2, 1, 2> expected_result{1, 2, 4, 8};
    Tensor3D<int8_t, 2, 1, 2> result =
        arith::index_cast<Tensor3D<int8_t, 2, 1, 2>>(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor4D<size_t, 2, 1, 2, 1> x{1, 2, 4, 8};
    Tensor4D<int8_t, 2, 1, 2, 1> expected_result{1, 2, 4, 8};
    Tensor4D<int8_t, 2, 1, 2, 1> result =
        arith::index_cast<Tensor4D<int8_t, 2, 1, 2, 1>>(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
}

} // namespace
