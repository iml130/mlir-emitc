// Copyright Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "gmock/gmock.h"

#include "emitc/tensor.h"
#include "emitc/types.h"

namespace {

using namespace emitc;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::Pointwise;

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

TEST(tensor, splat) {
  {
    uint32_t x = 1;
    Tensor0D<uint32_t> result = tensor::splat<Tensor0D<uint32_t>>(x);
    Tensor0D<uint32_t> expected_result{1};

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    int32_t x = -1;
    Tensor1D<int32_t, 3> result = tensor::splat<Tensor1D<int32_t, 3>>(x);
    Tensor1D<int32_t, 3> expected_result{-1, -1, -1};

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    float x = 1.5f;
    Tensor2D<float, 2, 2> result = tensor::splat<Tensor2D<float, 2, 2>>(x);
    Tensor2D<float, 2, 2> expected_result{1.5f, 1.5f, 1.5f, 1.5f};

    EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));
  }
  {
    float x = 1.2f;
    Tensor3D<float, 2, 3, 1> result =
        tensor::splat<Tensor3D<float, 2, 3, 1>>(x);
    Tensor3D<float, 2, 3, 1> expected_result{1.2f, 1.2f, 1.2f,
                                             1.2f, 1.2f, 1.2f};

    EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));
  }
  {
    float x = 1.1f;
    Tensor4D<float, 2, 3, 1, 2> result =
        tensor::splat<Tensor4D<float, 2, 3, 1, 2>>(x);
    Tensor4D<float, 2, 3, 1, 2> expected_result{
        1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f};

    EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));
  }
}

} // namespace
