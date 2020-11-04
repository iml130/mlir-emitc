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

#include "emitc/emitc_tensor.h"

namespace {

using ::testing::Eq;
using ::testing::Pointwise;

TEST(tensor, tensor_of_bool) {
  Tensor0D<bool> t0{false};
  Tensor1D<bool, 2> t1{false, true};
  Tensor2D<bool, 2, 2> t2{false, true, false, true};

  EXPECT_FALSE(t0[0]);
  EXPECT_FALSE(t0());

  EXPECT_FALSE(t1[0]);
  EXPECT_FALSE(t1(0));
  EXPECT_TRUE(t1[1]);
  EXPECT_TRUE(t1(1));

  EXPECT_FALSE(t2[0]);
  EXPECT_FALSE(t2(0, 0));
  EXPECT_TRUE(t2[1]);
  EXPECT_TRUE(t2(0, 1));
  EXPECT_FALSE(t2[2]);
  EXPECT_FALSE(t2(1, 0));
  EXPECT_TRUE(t2[3]);
  EXPECT_TRUE(t2(1, 1));

  t0[0] = !t0[0];
  t1[0] = !t1[0];
  t1[1] = !t1[1];
  t2[0] = !t2[0];
  t2[1] = !t2[1];
  t2[2] = !t2[2];
  t2[3] = !t2[3];

  EXPECT_TRUE(t0[0]);
  EXPECT_TRUE(t0());

  EXPECT_TRUE(t1[0]);
  EXPECT_TRUE(t1(0));
  EXPECT_FALSE(t1[1]);
  EXPECT_FALSE(t1(1));

  EXPECT_TRUE(t2[0]);
  EXPECT_TRUE(t2(0, 0));
  EXPECT_FALSE(t2[1]);
  EXPECT_FALSE(t2(0, 1));
  EXPECT_TRUE(t2[2]);
  EXPECT_TRUE(t2(1, 0));
  EXPECT_FALSE(t2[3]);
  EXPECT_FALSE(t2(1, 1));

  t0() = !t0();
  t1(0) = !t1(0);
  t1(1) = !t1(1);
  t2(0, 0) = !t2(0, 0);
  t2(0, 1) = !t2(0, 1);
  t2(1, 0) = !t2(1, 0);
  t2(1, 1) = !t2(1, 1);

  EXPECT_FALSE(t0[0]);
  EXPECT_FALSE(t0());

  EXPECT_FALSE(t1[0]);
  EXPECT_FALSE(t1(0));
  EXPECT_TRUE(t1[1]);
  EXPECT_TRUE(t1(1));

  EXPECT_FALSE(t2[0]);
  EXPECT_FALSE(t2(0, 0));
  EXPECT_TRUE(t2[1]);
  EXPECT_TRUE(t2(0, 1));
  EXPECT_FALSE(t2[2]);
  EXPECT_FALSE(t2(1, 0));
  EXPECT_TRUE(t2[3]);
  EXPECT_TRUE(t2(1, 1));
}

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

TEST(tensor, tensor_default_constructor_3d) {
  Tensor3D<float, 2, 1, 2> tensor;

  // multi dimensional indexing
  EXPECT_EQ(0.0, tensor(0, 0, 0));
  EXPECT_EQ(0.0, tensor(0, 0, 1));
  EXPECT_EQ(0.0, tensor(1, 0, 0));
  EXPECT_EQ(0.0, tensor(1, 0, 1));

  // flat indexing
  EXPECT_EQ(0.0, tensor[0]);
  EXPECT_EQ(0.0, tensor[1]);
  EXPECT_EQ(0.0, tensor[2]);
  EXPECT_EQ(0.0, tensor[3]);
}

TEST(tensor, tensor_default_constructor_4d) {
  Tensor4D<float, 2, 1, 2, 2> tensor;

  // multi dimensional indexing
  EXPECT_EQ(0.0, tensor(0, 0, 0, 0));
  EXPECT_EQ(0.0, tensor(0, 0, 0, 1));
  EXPECT_EQ(0.0, tensor(0, 0, 1, 0));
  EXPECT_EQ(0.0, tensor(0, 0, 1, 1));
  EXPECT_EQ(0.0, tensor(1, 0, 0, 0));
  EXPECT_EQ(0.0, tensor(1, 0, 0, 1));
  EXPECT_EQ(0.0, tensor(1, 0, 1, 0));
  EXPECT_EQ(0.0, tensor(1, 0, 1, 1));

  // flat indexing
  EXPECT_EQ(0.0, tensor[0]);
  EXPECT_EQ(0.0, tensor[1]);
  EXPECT_EQ(0.0, tensor[2]);
  EXPECT_EQ(0.0, tensor[3]);
  EXPECT_EQ(0.0, tensor[4]);
  EXPECT_EQ(0.0, tensor[5]);
  EXPECT_EQ(0.0, tensor[6]);
  EXPECT_EQ(0.0, tensor[7]);
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

TEST(tensor, initializer_list_3d) {
  Tensor3D<int32_t, 2, 1, 2> tensor{1, 2, 3, 4};

  // multi dimensional indexing
  EXPECT_EQ(1, tensor(0, 0, 0));
  EXPECT_EQ(2, tensor(0, 0, 1));
  EXPECT_EQ(3, tensor(1, 0, 0));
  EXPECT_EQ(4, tensor(1, 0, 1));

  // flat indexing (row major)
  EXPECT_EQ(1, tensor[0]);
  EXPECT_EQ(2, tensor[1]);
  EXPECT_EQ(3, tensor[2]);
  EXPECT_EQ(4, tensor[3]);
}

TEST(tensor, initializer_list_4d) {
  Tensor4D<int32_t, 2, 1, 2, 2> tensor{1, 2, 3, 4, 5, 6, 7, 8};

  // multi dimensional indexing
  EXPECT_EQ(1, tensor(0, 0, 0, 0));
  EXPECT_EQ(2, tensor(0, 0, 0, 1));
  EXPECT_EQ(3, tensor(0, 0, 1, 0));
  EXPECT_EQ(4, tensor(0, 0, 1, 1));
  EXPECT_EQ(5, tensor(1, 0, 0, 0));
  EXPECT_EQ(6, tensor(1, 0, 0, 1));
  EXPECT_EQ(7, tensor(1, 0, 1, 0));
  EXPECT_EQ(8, tensor(1, 0, 1, 1));

  // flat indexing (row major)
  EXPECT_EQ(1, tensor[0]);
  EXPECT_EQ(2, tensor[1]);
  EXPECT_EQ(3, tensor[2]);
  EXPECT_EQ(4, tensor[3]);
  EXPECT_EQ(5, tensor[4]);
  EXPECT_EQ(6, tensor[5]);
  EXPECT_EQ(7, tensor[6]);
  EXPECT_EQ(8, tensor[7]);
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

  auto lambda_3d = []() -> void {
    Tensor3D<int8_t, 2, 3, 1> t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  };
  EXPECT_DEATH(lambda_3d();, "");

  auto lambda_4d = []() -> void {
    Tensor4D<int8_t, 2, 3, 1, 2> t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  };
  EXPECT_DEATH(lambda_4d();, "");
}

TEST(tensor, dimension_1d) {
  Tensor1D<float, 2> tensor;

  EXPECT_EQ(2, tensor.dim(0));

  Tensor1D<int32_t, 13> tensor2;

  EXPECT_EQ(13, tensor2.dim(0));
}

TEST(tensor, dimension_2d) {
  Tensor2D<float, 4, 12> tensor;

  EXPECT_EQ(4, tensor.dim(0));
  EXPECT_EQ(12, tensor.dim(1));

  Tensor2D<int8_t, 64, 16> tensor2;

  EXPECT_EQ(64, tensor2.dim(0));
  EXPECT_EQ(16, tensor2.dim(1));
}

TEST(tensor, dimension_3d) {
  Tensor3D<float, 2, 1, 7> tensor;

  EXPECT_EQ(2, tensor.dim(0));
  EXPECT_EQ(1, tensor.dim(1));
  EXPECT_EQ(7, tensor.dim(2));

  Tensor3D<int32_t, 13, 9, 24> tensor2;

  EXPECT_EQ(13, tensor2.dim(0));
  EXPECT_EQ(9, tensor2.dim(1));
  EXPECT_EQ(24, tensor2.dim(2));
}

TEST(tensor, dimension_4d) {
  Tensor4D<float, 2, 1, 4, 5> tensor;

  EXPECT_EQ(2, tensor.dim(0));
  EXPECT_EQ(1, tensor.dim(1));
  EXPECT_EQ(4, tensor.dim(2));
  EXPECT_EQ(5, tensor.dim(3));

  Tensor4D<int32_t, 13, 6, 9, 8> tensor2;

  EXPECT_EQ(13, tensor2.dim(0));
  EXPECT_EQ(6, tensor2.dim(1));
  EXPECT_EQ(9, tensor2.dim(2));
  EXPECT_EQ(8, tensor2.dim(3));
}

TEST(tensor, size_0d) {
  Tensor0D<float> tensor;

  EXPECT_EQ(1, tensor.size());

  Tensor0D<int32_t> tensor2;

  EXPECT_EQ(1, tensor2.size());
}

TEST(tensor, size_1d) {
  Tensor1D<float, 2> tensor;

  EXPECT_EQ(2, tensor.size());

  Tensor1D<int32_t, 13> tensor2;

  EXPECT_EQ(13, tensor2.size());
}

TEST(tensor, size_2d) {
  Tensor2D<float, 4, 12> tensor;

  EXPECT_EQ(48, tensor.size());

  Tensor2D<int8_t, 64, 16> tensor2;

  EXPECT_EQ(1024, tensor2.size());
}

TEST(tensor, size_3d) {
  Tensor3D<float, 4, 3, 5> tensor;

  EXPECT_EQ(60, tensor.size());

  Tensor3D<int8_t, 8, 8, 8> tensor2;

  EXPECT_EQ(512, tensor2.size());
}

TEST(tensor, size_4d) {
  Tensor4D<float, 4, 2, 1, 3> tensor;

  EXPECT_EQ(24, tensor.size());

  Tensor4D<int8_t, 1, 5, 6, 2> tensor2;

  EXPECT_EQ(60, tensor2.size());
}

TEST(tensor, meta_get_element_type) {
  using s0 = float;
  using s1 = int32_t;
  using t0 = Tensor0D<uint16_t>;
  using t1 = Tensor1D<int8_t, 12>;
  using t2 = Tensor2D<double, 3, 7>;
  using t3 = Tensor3D<float, 1, 4, 6>;
  using t4 = Tensor4D<uint8_t, 2, 1, 1, 9>;

  const bool check_s0 = std::is_same<float, get_element_type<s0>::type>::value;
  const bool check_s1 =
      std::is_same<int32_t, get_element_type<s1>::type>::value;
  const bool check_t0 =
      std::is_same<uint16_t, get_element_type<t0>::type>::value;
  const bool check_t1 = std::is_same<int8_t, get_element_type<t1>::type>::value;
  const bool check_t2 = std::is_same<double, get_element_type<t2>::type>::value;
  const bool check_t3 = std::is_same<float, get_element_type<t3>::type>::value;
  const bool check_t4 =
      std::is_same<uint8_t, get_element_type<t4>::type>::value;

  EXPECT_TRUE(check_s0);
  EXPECT_TRUE(check_s1);
  EXPECT_TRUE(check_t0);
  EXPECT_TRUE(check_t1);
  EXPECT_TRUE(check_t2);
  EXPECT_TRUE(check_t3);
  EXPECT_TRUE(check_t4);
}

TEST(tensor, meta_is_scalar) {
  using s0 = float;
  using s1 = int32_t;
  using t0 = Tensor0D<uint16_t>;
  using t1 = Tensor1D<int8_t, 12>;
  using t2 = Tensor2D<double, 3, 7>;
  using t3 = Tensor3D<float, 1, 4, 6>;
  using t4 = Tensor4D<uint8_t, 2, 1, 1, 9>;

  EXPECT_TRUE(is_scalar<s0>::value);
  EXPECT_TRUE(is_scalar<s1>::value);
  EXPECT_FALSE(is_scalar<t0>::value);
  EXPECT_FALSE(is_scalar<t1>::value);
  EXPECT_FALSE(is_scalar<t2>::value);
  EXPECT_FALSE(is_scalar<t3>::value);
  EXPECT_FALSE(is_scalar<t4>::value);
}

TEST(tensor, meta_is_tensor) {
  using s0 = float;
  using s1 = int32_t;
  using t0 = Tensor0D<uint16_t>;
  using t1 = Tensor1D<int8_t, 12>;
  using t2 = Tensor2D<double, 3, 7>;
  using t3 = Tensor3D<float, 1, 4, 6>;
  using t4 = Tensor4D<uint8_t, 2, 1, 1, 9>;

  EXPECT_FALSE(is_tensor<s0>::value);
  EXPECT_FALSE(is_tensor<s1>::value);
  EXPECT_TRUE(is_tensor<t0>::value);
  EXPECT_TRUE(is_tensor<t1>::value);
  EXPECT_TRUE(is_tensor<t2>::value);
  EXPECT_TRUE(is_tensor<t3>::value);
  EXPECT_TRUE(is_tensor<t4>::value);
}

TEST(tensor, meta_is_tensor_0d) {
  using s0 = float;
  using s1 = int32_t;
  using t0 = Tensor0D<uint16_t>;
  using t1 = Tensor1D<int8_t, 12>;
  using t2 = Tensor2D<double, 3, 7>;
  using t3 = Tensor3D<float, 1, 4, 6>;
  using t4 = Tensor4D<uint8_t, 2, 1, 1, 9>;

  const bool b0 = is_tensor_of_dim<0, t0>::value;
  const bool b1 = is_tensor_of_dim<0, t1>::value;
  const bool b2 = is_tensor_of_dim<0, t2>::value;
  const bool b3 = is_tensor_of_dim<0, t3>::value;
  const bool b4 = is_tensor_of_dim<0, t4>::value;

  EXPECT_FALSE((is_tensor_of_dim<0, s0>::value));
  EXPECT_FALSE((is_tensor_of_dim<0, s1>::value));
  EXPECT_TRUE(b0);
  EXPECT_FALSE(b1);
  EXPECT_FALSE(b2);
  EXPECT_FALSE(b3);
  EXPECT_FALSE(b4);
}

TEST(tensor, meta_is_tensor_1d) {
  using s0 = float;
  using s1 = int32_t;
  using t0 = Tensor0D<uint16_t>;
  using t1 = Tensor1D<int8_t, 12>;
  using t2 = Tensor2D<double, 3, 7>;
  using t3 = Tensor3D<float, 1, 4, 6>;
  using t4 = Tensor4D<uint8_t, 2, 1, 1, 9>;

  const bool b0 = is_tensor_of_dim<1, t0>::value;
  const bool b1 = is_tensor_of_dim<1, t1>::value;
  const bool b2 = is_tensor_of_dim<1, t2>::value;
  const bool b3 = is_tensor_of_dim<1, t3>::value;
  const bool b4 = is_tensor_of_dim<1, t4>::value;

  EXPECT_FALSE((is_tensor_of_dim<1, s0>::value));
  EXPECT_FALSE((is_tensor_of_dim<1, s1>::value));
  EXPECT_FALSE(b0);
  EXPECT_TRUE(b1);
  EXPECT_FALSE(b2);
  EXPECT_FALSE(b3);
  EXPECT_FALSE(b4);
}

TEST(tensor, meta_is_tensor_2d) {
  using s0 = float;
  using s1 = int32_t;
  using t0 = Tensor0D<uint16_t>;
  using t1 = Tensor1D<int8_t, 12>;
  using t2 = Tensor2D<double, 3, 7>;
  using t3 = Tensor3D<float, 1, 4, 6>;
  using t4 = Tensor4D<uint8_t, 2, 1, 1, 9>;

  const bool b0 = is_tensor_of_dim<2, t0>::value;
  const bool b1 = is_tensor_of_dim<2, t1>::value;
  const bool b2 = is_tensor_of_dim<2, t2>::value;
  const bool b3 = is_tensor_of_dim<2, t3>::value;
  const bool b4 = is_tensor_of_dim<2, t4>::value;

  EXPECT_FALSE((is_tensor_of_dim<2, s0>::value));
  EXPECT_FALSE((is_tensor_of_dim<2, s1>::value));
  EXPECT_FALSE(b0);
  EXPECT_FALSE(b1);
  EXPECT_TRUE(b2);
  EXPECT_FALSE(b3);
  EXPECT_FALSE(b4);
}

TEST(tensor, meta_is_tensor_3d) {
  using s0 = float;
  using s1 = int32_t;
  using t0 = Tensor0D<uint16_t>;
  using t1 = Tensor1D<int8_t, 12>;
  using t2 = Tensor2D<double, 3, 7>;
  using t3 = Tensor3D<float, 1, 4, 6>;
  using t4 = Tensor4D<uint8_t, 2, 1, 1, 9>;

  const bool b0 = is_tensor_of_dim<3, t0>::value;
  const bool b1 = is_tensor_of_dim<3, t1>::value;
  const bool b2 = is_tensor_of_dim<3, t2>::value;
  const bool b3 = is_tensor_of_dim<3, t3>::value;
  const bool b4 = is_tensor_of_dim<3, t4>::value;

  EXPECT_FALSE((is_tensor_of_dim<3, s0>::value));
  EXPECT_FALSE((is_tensor_of_dim<3, s1>::value));
  EXPECT_FALSE(b0);
  EXPECT_FALSE(b1);
  EXPECT_FALSE(b2);
  EXPECT_TRUE(b3);
  EXPECT_FALSE(b4);
}

TEST(tensor, meta_is_tensor_4d) {
  using s0 = float;
  using s1 = int32_t;
  using t0 = Tensor0D<uint16_t>;
  using t1 = Tensor1D<int8_t, 12>;
  using t2 = Tensor2D<double, 3, 7>;
  using t3 = Tensor3D<float, 1, 4, 6>;
  using t4 = Tensor4D<uint8_t, 2, 1, 1, 9>;

  const bool b0 = is_tensor_of_dim<4, t0>::value;
  const bool b1 = is_tensor_of_dim<4, t1>::value;
  const bool b2 = is_tensor_of_dim<4, t2>::value;
  const bool b3 = is_tensor_of_dim<4, t3>::value;
  const bool b4 = is_tensor_of_dim<4, t4>::value;

  EXPECT_FALSE((is_tensor_of_dim<4, s0>::value));
  EXPECT_FALSE((is_tensor_of_dim<4, s1>::value));
  EXPECT_FALSE(b0);
  EXPECT_FALSE(b1);
  EXPECT_FALSE(b2);
  EXPECT_FALSE(b3);
  EXPECT_TRUE(b4);
}

TEST(tensor, meta_replace_element_type) {
  using s0 = float;
  using s1 = int32_t;
  using t0 = Tensor0D<uint16_t>;
  using t1 = Tensor1D<int8_t, 12>;
  using t2 = Tensor2D<double, 3, 7>;
  using t3 = Tensor3D<float, 1, 4, 6>;
  using t4 = Tensor4D<uint8_t, 2, 1, 1, 9>;

  const bool check_s0 =
      std::is_same<int32_t, replace_element_type<int32_t, s0>::type>::value;
  const bool check_s1 =
      std::is_same<uint8_t, replace_element_type<uint8_t, s1>::type>::value;
  const bool check_t0 =
      std::is_same<Tensor0D<int32_t>,
                   replace_element_type<int32_t, t0>::type>::value;
  const bool check_t1 =
      std::is_same<Tensor1D<uint16_t, 12>,
                   replace_element_type<uint16_t, t1>::type>::value;
  const bool check_t2 =
      std::is_same<Tensor2D<int32_t, 3, 7>,
                   replace_element_type<int32_t, t2>::type>::value;
  const bool check_t3 =
      std::is_same<Tensor3D<uint8_t, 1, 4, 6>,
                   replace_element_type<uint8_t, t3>::type>::value;
  const bool check_t4 =
      std::is_same<Tensor4D<float, 2, 1, 1, 9>,
                   replace_element_type<float, t4>::type>::value;

  EXPECT_TRUE(check_s0);
  EXPECT_TRUE(check_s1);
  EXPECT_TRUE(check_t0);
  EXPECT_TRUE(check_t1);
  EXPECT_TRUE(check_t2);
  EXPECT_TRUE(check_t3);
  EXPECT_TRUE(check_t4);
}

TEST(tensor, ravel_index) {
  Tensor0D<uint16_t> t0;
  Tensor1D<int8_t, 12> t1;
  Tensor2D<double, 3, 7> t2;
  Tensor3D<float, 2, 4, 6> t3;
  Tensor4D<uint8_t, 2, 3, 4, 9> t4;

  EXPECT_EQ(t0.ravel_index(), 0);
  EXPECT_EQ(t1.ravel_index(0), 0);
  EXPECT_EQ(t1.ravel_index(7), 7);
  EXPECT_EQ(t2.ravel_index(0, 2), 2);
  EXPECT_EQ(t2.ravel_index(2, 3), 17);
  EXPECT_EQ(t3.ravel_index(0, 2, 0), 12);
  EXPECT_EQ(t3.ravel_index(1, 3, 4), 46);
  EXPECT_EQ(t4.ravel_index(1, 0, 0, 3), 111);
  EXPECT_EQ(t4.ravel_index(1, 2, 3, 4), 211);
}

TEST(tensor, unravel_index) {
  Tensor0D<uint16_t> t0;
  Tensor1D<int8_t, 12> t1;
  Tensor2D<double, 3, 7> t2;
  Tensor3D<float, 2, 4, 6> t3;
  Tensor4D<uint8_t, 2, 3, 4, 9> t4;

  EXPECT_THAT(t1.unravel_index(0), Pointwise(Eq(), {0}));
  EXPECT_THAT(t1.unravel_index(7), Pointwise(Eq(), {7}));
  EXPECT_THAT(t2.unravel_index(2), Pointwise(Eq(), {0, 2}));
  EXPECT_THAT(t2.unravel_index(17), Pointwise(Eq(), {2, 3}));
  EXPECT_THAT(t3.unravel_index(12), Pointwise(Eq(), {0, 2, 0}));
  EXPECT_THAT(t3.unravel_index(46), Pointwise(Eq(), {1, 3, 4}));
  EXPECT_THAT(t4.unravel_index(111), Pointwise(Eq(), {1, 0, 0, 3}));
  EXPECT_THAT(t4.unravel_index(211), Pointwise(Eq(), {1, 2, 3, 4}));
}

} // namespace
