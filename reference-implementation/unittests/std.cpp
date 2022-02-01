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

#include "emitc/std.h"
#include "emitc/types.h"

namespace {

using namespace emitc;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::Pointwise;

TEST(std, splat) {
  {
    uint32_t x = 1;
    Tensor0D<uint32_t> result = standard::splat<Tensor0D<uint32_t>>(x);
    Tensor0D<uint32_t> expected_result{1};

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    int32_t x = -1;
    Tensor1D<int32_t, 3> result = standard::splat<Tensor1D<int32_t, 3>>(x);
    Tensor1D<int32_t, 3> expected_result{-1, -1, -1};

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    float x = 1.5f;
    Tensor2D<float, 2, 2> result = standard::splat<Tensor2D<float, 2, 2>>(x);
    Tensor2D<float, 2, 2> expected_result{1.5f, 1.5f, 1.5f, 1.5f};

    EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));
  }
  {
    float x = 1.2f;
    Tensor3D<float, 2, 3, 1> result =
        standard::splat<Tensor3D<float, 2, 3, 1>>(x);
    Tensor3D<float, 2, 3, 1> expected_result{1.2f, 1.2f, 1.2f,
                                             1.2f, 1.2f, 1.2f};

    EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));
  }
  {
    float x = 1.1f;
    Tensor4D<float, 2, 3, 1, 2> result =
        standard::splat<Tensor4D<float, 2, 3, 1, 2>>(x);
    Tensor4D<float, 2, 3, 1, 2> expected_result{
        1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f};

    EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));
  }
}

} // namespace
