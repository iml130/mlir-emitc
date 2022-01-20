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
  EXPECT_THAT(standard::splat<Tensor0D<uint32_t>>(1), Pointwise(Eq(), {1}));

  auto lambda_1d = []() -> Tensor1D<int32_t, 3> {
    return standard::splat<Tensor1D<int32_t, 3>>(-1);
  };

  EXPECT_THAT(lambda_1d(), Pointwise(Eq(), {-1, -1, -1}));

  auto lambda_2d = []() -> Tensor2D<float, 2, 2> {
    return standard::splat<Tensor2D<float, 2, 2>>(1.5f);
  };

  EXPECT_THAT(lambda_2d(), Pointwise(FloatEq(), {1.5f, 1.5f, 1.5f, 1.5f}));

  auto lambda_3d = []() -> Tensor3D<float, 2, 3, 1> {
    return standard::splat<Tensor3D<float, 2, 3, 1>>(1.2f);
  };

  EXPECT_THAT(lambda_3d(),
              Pointwise(FloatEq(), {1.2f, 1.2f, 1.2f, 1.2f, 1.2f, 1.2f}));

  auto lambda_4d = []() -> Tensor4D<float, 2, 3, 1, 2> {
    return standard::splat<Tensor4D<float, 2, 3, 1, 2>>(1.1f);
  };

  EXPECT_THAT(lambda_4d(),
              Pointwise(FloatEq(), {1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f,
                                    1.1f, 1.1f, 1.1f, 1.1f, 1.1f}));
}

} // namespace
