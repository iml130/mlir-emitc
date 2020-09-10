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

namespace {

TEST(tensor, default_constructor_0d) {
  Tensor0D<float> tensor;

  // multi dimensional indexing
  EXPECT_EQ(0.0, tensor());

  // flat indexing
  EXPECT_EQ(0.0, tensor[0]);
}

TEST(tensor, default_constructor_1d) {
  Tensor1D<float, 2> tensor;

  // multi dimensional indexing
  EXPECT_EQ(0.0, tensor(0));
  EXPECT_EQ(0.0, tensor(1));

  // flat indexing
  EXPECT_EQ(0.0, tensor[0]);
  EXPECT_EQ(0.0, tensor[1]);
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

TEST(tensor, initializer_list_0d) {
  Tensor0D<float> tensor{1.0};

  // multi dimensional indexing
  EXPECT_EQ(1.0, tensor());

  // flat indexing
  EXPECT_EQ(1.0, tensor[0]);
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

TEST(tensor, wrong_size_initializer_list) {
  auto lambda_0d = []() -> void { Tensor0D<float> t{0.0, 1.0}; };
  EXPECT_DEATH(lambda_0d();, "");

  auto lambda_1d = []() -> void { Tensor1D<uint16_t, 1> t{0, 1, 3}; };
  EXPECT_DEATH(lambda_1d();, "");

  auto lambda_2d = []() -> void {
    Tensor2D<int8_t, 2, 3> t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  };
  EXPECT_DEATH(lambda_2d();, "");
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

TEST(tensor, size_0d) {
  Tensor0D<float> tensor;

  EXPECT_EQ(1, tensor.size_);

  Tensor0D<int32_t> tensor2;

  EXPECT_EQ(1, tensor2.size_);
}

TEST(tensor, size_1d) {
  Tensor1D<float, 2> tensor;

  EXPECT_EQ(2, tensor.size_);

  Tensor1D<int32_t, 13> tensor2;

  EXPECT_EQ(13, tensor2.size_);
}

TEST(tensor, size_2d) {
  Tensor2D<float, 4, 12> tensor;

  EXPECT_EQ(48, tensor.size_);

  Tensor2D<int8_t, 64, 16> tensor2;

  EXPECT_EQ(1024, tensor2.size_);
}

TEST(tensor, meta_is_scalar) {
  using s0 = float;
  using s1 = int32_t;
  using t0 = Tensor0D<uint16_t>;
  using t1 = Tensor1D<int8_t, 12>;
  using t2 = Tensor2D<double, 3, 7>;

  EXPECT_TRUE(is_scalar<s0>::value);
  EXPECT_TRUE(is_scalar<s1>::value);
  EXPECT_FALSE(is_scalar<t0>::value);
  EXPECT_FALSE(is_scalar<t1>::value);
  EXPECT_FALSE(is_scalar<t2>::value);
}

TEST(tensor, meta_is_tensor) {
  using s0 = float;
  using s1 = int32_t;
  using t0 = Tensor0D<uint16_t>;
  using t1 = Tensor1D<int8_t, 12>;
  using t2 = Tensor2D<double, 3, 7>;

  EXPECT_FALSE(is_tensor<s0>::value);
  EXPECT_FALSE(is_tensor<s1>::value);
  EXPECT_TRUE(is_tensor<t0>::value);
  EXPECT_TRUE(is_tensor<t1>::value);
  EXPECT_TRUE(is_tensor<t2>::value);
}

TEST(tensor, meta_is_tensor_0d) {
  using s0 = float;
  using s1 = int32_t;
  using t0 = Tensor0D<uint16_t>;
  using t1 = Tensor1D<int8_t, 12>;
  using t2 = Tensor2D<double, 3, 7>;

  EXPECT_FALSE(is_tensor_0d<s0>::value);
  EXPECT_FALSE(is_tensor_0d<s1>::value);
  EXPECT_TRUE(is_tensor_0d<t0>::value);
  EXPECT_FALSE(is_tensor_0d<t1>::value);
  EXPECT_FALSE(is_tensor_0d<t2>::value);
}

TEST(tensor, meta_is_tensor_1d) {
  using s0 = float;
  using s1 = int32_t;
  using t0 = Tensor0D<uint16_t>;
  using t1 = Tensor1D<int8_t, 12>;
  using t2 = Tensor2D<double, 3, 7>;

  EXPECT_FALSE(is_tensor_1d<s0>::value);
  EXPECT_FALSE(is_tensor_1d<s1>::value);
  EXPECT_FALSE(is_tensor_1d<t0>::value);
  EXPECT_TRUE(is_tensor_1d<t1>::value);
  EXPECT_FALSE(is_tensor_1d<t2>::value);
}

TEST(tensor, meta_is_tensor_2d) {
  using s0 = float;
  using s1 = int32_t;
  using t0 = Tensor0D<uint16_t>;
  using t1 = Tensor1D<int8_t, 12>;
  using t2 = Tensor2D<double, 3, 7>;

  EXPECT_FALSE(is_tensor_2d<s0>::value);
  EXPECT_FALSE(is_tensor_2d<s1>::value);
  EXPECT_FALSE(is_tensor_2d<t0>::value);
  EXPECT_FALSE(is_tensor_2d<t1>::value);
  EXPECT_TRUE(is_tensor_2d<t2>::value);
}

} // namespace
