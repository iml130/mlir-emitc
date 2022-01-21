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
  Tensor0D<float> t0{1.0};
  Tensor1D<int32_t, 2> t1{1, 2};
  Tensor2D<uint16_t, 1, 4> t2{10, 11, 12, 13};
  Tensor2D<uint16_t, 2, 2> t3{10, 11, 12, 13};
  Tensor3D<uint32_t, 2, 1, 2> t4{10, 11, 12, 13};

  EXPECT_EQ(1.0, tensor::extract(t0));
  EXPECT_EQ(2, tensor::extract(t1, 1));
  EXPECT_EQ(12, tensor::extract(t2, 0, 2));
  EXPECT_EQ(12, tensor::extract(t3, 1, 0));
  EXPECT_EQ(12, tensor::extract(t4, 1, 0, 0));
}

} // namespace
