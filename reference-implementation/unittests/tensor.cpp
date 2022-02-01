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

#include "emitc/tensor.h"
#include "emitc/types.h"

namespace {

using namespace emitc;
using ::testing::Eq;

TEST(tensor, extract) {
  {
    Tensor0D<float> x{1.0};
    float expected_result = 1.0;
    float result = tensor::extract(x);

    EXPECT_EQ(result, expected_result);
  }
  {
    Tensor1D<int32_t, 2> x{1, 2};
    int32_t expected_result = 2;
    int32_t result = tensor::extract(x, 1);

    EXPECT_EQ(result, expected_result);
  }
  {
    Tensor2D<uint16_t, 1, 4> x{10, 11, 12, 13};
    uint16_t expected_result = 12;
    uint16_t result = tensor::extract(x, 0, 2);

    EXPECT_EQ(result, expected_result);
  }
  {
    Tensor3D<uint16_t, 2, 1, 2> x{10, 11, 12, 13};
    uint16_t expected_result = 12;
    uint16_t result = tensor::extract(x, 1, 0, 0);

    EXPECT_EQ(result, expected_result);
  }
  {
    Tensor4D<uint32_t, 1, 2, 2, 2> x{10, 11, 12, 13, 14, 15, 16, 17};
    uint32_t expected_result = 15;
    uint32_t result = tensor::extract(x, 0, 1, 0, 1);

    EXPECT_EQ(result, expected_result);
  }
}

} // namespace
