// Copyright Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "gmock/gmock.h"

#include "emitc/tosa.h"
#include "emitc/types.h"

namespace {

using namespace emitc;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pointwise;

const float EPSILON = 5e-4;

// Other ops
TEST(tosa, conv2d) {
  {
    // strides = 1
    using InputType = Tensor4D<float, 1, 4, 5, 2>;  // N H W C
    using WeightType = Tensor4D<float, 1, 3, 2, 2>; // COUT KH KW CIN
    using ResultType = Tensor4D<float, 1, 4, 5, 1>; // N H W C
    InputType input{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
    WeightType weights{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    ResultType expected_result{600,  736,  872,  1008, 476,  1310, 1466,
                               1622, 1778, 805,  2090, 2246, 2402, 2558,
                               1135, 1080, 1152, 1224, 1296, 524};

    Tensor1D<int64_t, 4> padding{1, 1, 0, 1}; // {pt, pb, pl, pr}
    Tensor1D<int64_t, 2> dilation{1, 1};
    Tensor1D<int64_t, 2> stride{1, 1};

    ResultType result =
        tosa::conv2d<ResultType>(input, weights, padding, stride, dilation);
    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    // Strided convolution
    using InputType = Tensor4D<float, 1, 4, 4, 1>;  // N H W C
    using WeightType = Tensor4D<float, 1, 2, 2, 1>; // COUT KH KW CIN
    using ResultType = Tensor4D<float, 1, 2, 2, 1>; // N H W C
    // clang-format off
    InputType input{1,  2,  3,  4,
                    5,  6,  7,  8,
                    9,  10, 11, 12,
                    13, 14, 15, 16};
    WeightType weights{1, 2,
                       3, 4};
    ResultType expected_result{44,  64,
                              124, 144};
    // clang-format on
    Tensor1D<int64_t, 4> padding{0, 0, 0, 1}; // {pt, pb, pl, pr}
    Tensor1D<int64_t, 2> dilation{1, 1};
    Tensor1D<int64_t, 2> stride{2, 2};

    ResultType result =
        tosa::conv2d<ResultType>(input, weights, padding, stride, dilation);
    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
}

} // namespace
