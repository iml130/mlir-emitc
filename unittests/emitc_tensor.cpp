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

#include "emitc_tensor.h"

#include "gmock/gmock.h"

using ::testing::ElementsAre;

TEST(tensor, default_constructor_1d) {
  Tensor1D<float, 2> tensor;

  // multi dimensional indexing
  EXPECT_EQ(0.0, tensor(0));
  EXPECT_EQ(0.0, tensor(1));

  // flat indexing
  EXPECT_EQ(0.0, tensor[0]);
  EXPECT_EQ(0.0, tensor[1]);
}

TEST(tensor, initializer_list_1d) {
  Tensor1D<float, 2> tensor{1.0, 2.0};

  // multi dimensional indexing
  EXPECT_EQ(1.0, tensor(0));
  EXPECT_EQ(2.0, tensor(1));

  // flat indexing
  EXPECT_EQ(1.0, tensor[0]);
  EXPECT_EQ(2.0, tensor[1]);
}

TEST(tensor, default_constructor_2d) {
  Tensor2D<float, 2, 2> tensor;

  // multi dimensional indexing
  EXPECT_EQ(0.0, tensor(0, 0));
  EXPECT_EQ(0.0, tensor(0, 1));
  EXPECT_EQ(0.0, tensor(1, 0));
  EXPECT_EQ(0.0, tensor(1, 1));

  // flat indexing
  EXPECT_EQ(0.0, tensor[0]);
  EXPECT_EQ(0.0, tensor[1]);
  EXPECT_EQ(0.0, tensor[2]);
  EXPECT_EQ(0.0, tensor[3]);
}

TEST(tensor, initializer_list_2d) {
  Tensor2D<int32_t, 2, 2> tensor{1, 2, 3, 4};

  // multi dimensional indexing
  EXPECT_EQ(1, tensor(0, 0));
  EXPECT_EQ(2, tensor(0, 1));
  EXPECT_EQ(3, tensor(1, 0));
  EXPECT_EQ(4, tensor(1, 1));

  // flat indexing (row major)
  EXPECT_EQ(1, tensor[0]);
  EXPECT_EQ(2, tensor[1]);
  EXPECT_EQ(3, tensor[2]);
  EXPECT_EQ(4, tensor[3]);
}

TEST(tensor, dimension_1d) {
  Tensor1D<float, 2> tensor;

  EXPECT_EQ(2, tensor.dimX);

  Tensor1D<int32_t, 13> tensor2;

  EXPECT_EQ(13, tensor2.dimX);
}

TEST(tensor, dimension_2d) {
  Tensor2D<float, 4, 12> tensor;

  EXPECT_EQ(4, tensor.dimX);
  EXPECT_EQ(12, tensor.dimY);

  Tensor2D<int8_t, 64, 16> tensor2;

  EXPECT_EQ(64, tensor2.dimX);
  EXPECT_EQ(16, tensor2.dimY);
}
