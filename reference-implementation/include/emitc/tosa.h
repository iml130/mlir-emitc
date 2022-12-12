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

// This file defines functions emitted by TosaToEmitC.

#ifndef EMITC_TOSA_H
#define EMITC_TOSA_H

#include <limits>

#include "emitc/core_ops.h"
#include "emitc/tensor.h"

#ifdef EMITC_TOSA_USE_EIGEN
#include "emitc/tosa_eigen.h"
#endif

namespace emitc {
namespace tosa {

/// Functions for unary elementwise TOSA ops.
// AbsOp
template <typename Src>
inline Src abs(Src x) {
  return emitc::abs<Src>(x);
}

// CastOp
template <typename Dest, typename Src>
inline Dest cast(Src x) {
  return emitc::convert<Dest>(x);
}

// CeilOp
template <typename Src>
inline Src ceil(Src x) {
  return emitc::ceil<Src>(x);
}

// ClampOp
template <typename Src>
inline Src clamp(Src operand, typename Src::value_type min_value,
                 typename Src::value_type max_value) {
  Tensor<typename Src::value_type> min{min_value};
  Tensor<typename Src::value_type> max{max_value};
  return emitc::clamp(min, operand, max);
}

// ClzOp
template <typename Src>
inline Src clz(Src x) {
  using ET_Src = typename get_element_type<Src>::type;
  static_assert(std::is_same<ET_Src, int32_t>::value,
                "Expected tensor of type int32_t");
  auto f = [](ET_Src element) {
    ET_Src count = 32;
    while (element != 0 && count > 0) {
      count--;
      element >>= 1;
    }
    return count;
  };
  return unary<Src>(x, f);
}

// ExpOp
template <typename Src>
inline Src exp(Src x) {
  return emitc::exp<Src>(x);
}

// FloorOp
template <typename Src>
inline Src floor(Src x) {
  return emitc::floor<Src>(x);
}

// LogOp
template <typename Src>
inline Src log(Src x) {
  return emitc::log<Src>(x);
}

// NegateOp
template <typename Src>
inline Src negate(Src x) {
  return emitc::negate(x);
}

// ReciprocalOp
template <typename Src>
inline Src reciprocal(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = [](ET_Src element) { return (static_cast<ET_Src>(1.0) / element); };

  return unary<Src>(x, f);
}

// RescaleOp
template <typename Dest, size_t Dim, typename Src>
inline Dest rescale(Src x, typename get_element_type<Src>::type in_zp,
                    typename get_element_type<Dest>::type out_zp,
                    Tensor1D<int64_t, Dim> mult, Tensor1D<int64_t, Dim> shift,
                    bool scale32, bool double_round, bool per_channel) {
  using ET_Dest = typename get_element_type<Dest>::type;
  using Dest_I32 = typename replace_element_type<int32_t, Dest>::type;

  assert(!(!scale32 && double_round) &&
         "Invalid combination of `scale32` and `double_round` arguments.");

  auto apply_scale = [=](int64_t element, int64_t mult, int64_t shift) {
    int64_t round = 1 << (shift - 1);
    if (double_round && shift > 31) {
      if (element >= 0)
        round += 1 << 30;
      else
        round -= 1 << 30;
    }

    int64_t result = (element * mult + round) >> shift;
    return static_cast<int32_t>(result);
  };

  Dest_I32 result;
  for (size_t i = 0; i < x.size(); ++i) {
    size_t index = per_channel ? x.unravel_index(i)[x.rank() - 1] : 0;
    int64_t element = x[i] - in_zp;
    int32_t scaled_element = apply_scale(element, mult[index], shift[index]);
    result[i] = scaled_element + out_zp;
  }

  Tensor0D<int32_t> min{
      static_cast<int32_t>(std::numeric_limits<ET_Dest>::min())};
  Tensor0D<int32_t> max{
      static_cast<int32_t>(std::numeric_limits<ET_Dest>::max())};

  return cast<Dest>(emitc::clamp(min, result, max));
}

// TanhOp
template <typename Src>
inline Src tanh(Src x) {
  return emitc::tanh<Src>(x);
}

/// Functions for binary elementwise TOSA ops.
// AddOp
template <typename Src>
inline Src add(Src x, Src y) {
  return emitc::add<Src>(x, y);
}

// ArithmeticRightShiftOp
template <typename Src>
inline Src arithmetic_right_shift(Src x, Src y, bool round) {
  using ET_Src = typename get_element_type<Src>::type;
  std::function<ET_Src(ET_Src, ET_Src)> f;
  if (round) {
    f = [](ET_Src left, ET_Src right) {
      ET_Src result = left >> right;
      if (right > 0 && ((left >> (right - 1)) & 1) != 0) {
        result++;
      }
      return result;
    };
  } else {
    f = [](ET_Src left, ET_Src right) { return left >> right; };
  }
  return binary<Src>(x, y, f);
}

// EqualOp
template <typename Dest, typename Src>
inline Dest equal(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;
  auto f = [](ET_Src left, ET_Src right) { return left == right; };
  return binary<Dest, Src>(x, y, f);
}

// LogicalLeftShiftOp
template <typename Src>
inline Src logical_left_shift(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;
  auto f = [](ET_Src left, ET_Src right) { return left << right; };
  return binary<Src>(x, y, f);
}

// MulOp
template <typename Src>
inline Src mul(Src x, Src y) {
  return emitc::mul(x, y);
}

// MaxOp
template <typename Src>
inline Src maximum(Src x, Src y) {
  return emitc::max(x, y);
}

// MinOp
template <typename Src>
inline Src minimum(Src x, Src y) {
  return emitc::min(x, y);
}

template <typename Src, IsTensorOfType<Src, int32_t> = true>
inline Src mul(Src x, Src y, const int32_t shift) {
  // Adopted from
  // https://git.mlplatform.org/tosa/reference_model.git/tree/reference_model/src/ops/ewise_binary.cc?id=df8626976df6c779bb30df9c5ceef689462109c0#n436
  if (shift > 0) {
    auto f = [&shift](int32_t x, int32_t y) -> int32_t {
      int64_t result;
      int64_t round = 1L << (shift - 1);
      result = x * y + round;
      result = result >> shift;
      return static_cast<int32_t>(result);
    };
    return binary<Src>(x, y, f);
  } else {
    return emitc::mul(x, y);
  }
}

// PowOp
template <typename Src>
inline Src pow(Src x, Src y) {
  return emitc::pow(x, y);
}

// SubOp
template <typename Src>
inline Src sub(Src x, Src y) {
  return emitc::sub<Src>(x, y);
}

// TableOp int8_t
template <size_t... Shape>
inline Tensor<int8_t, Shape...> table(Tensor<int8_t, Shape...> x,
                                      Tensor1D<int8_t, 256> table) {
  auto f = [&table](int8_t element) {
    return table(static_cast<int16_t>(element) + 128);
  };
  return unary<Tensor<int8_t, Shape...>>(x, f);
}

// TableOp int16_t
template <size_t... Shape>
inline Tensor<int32_t, Shape...> table(Tensor<int16_t, Shape...> x,
                                       Tensor1D<int16_t, 513> table) {
  auto f = [&table](int16_t element) {
    int32_t integer = (element >> 7) + 0x100; // 9 bit integer part
    int32_t fractional = element & 0x7F;      // 7 bit fractional part
    int32_t result_integer = table(integer);  // 16 bit integer part
    int32_t result_fractional = (table(integer + 1) - table(integer)) *
                                fractional; // 7 bit fractional part
    return (result_integer << 7) + result_fractional;
  };
  return unary<Tensor<int32_t, Shape...>>(x, f);
}

/// Functions for other TOSA ops.
// Disable Conv2DOp if Eigen implementation is used
#ifndef EMITC_TOSA_USE_EIGEN
// Conv2DOp
template <typename Dest, typename Src, typename Weights>
Dest conv2d(Src input, Weights weights, Tensor1D<int64_t, 4> padding,
            Tensor1D<int64_t, 2> stride, Tensor1D<int64_t, 2> dilation) {
  // This implementation is taken from emitc_mhlo.c (convolution) and slightly
  // adapted to fit the memory layout of tosa. Input is [N,IH,IW,IC], weights
  // are [OC,KH,KW,IC] and output is [N,H,W,OC].
  static_assert(is_tensor_of_dim<4, Src>::value,
                "Expected 4 dimensional input");
  static_assert(is_tensor_of_dim<4, Dest>::value,
                "Expected 4 dimensional output");
  static_assert(is_tensor_of_dim<4, Weights>::value,
                "Expected 4 dimensional weights");

  assert(stride[0] > 0);
  assert(stride[1] > 0);

  assert(dilation[0] == 1);
  assert(dilation[1] == 1);

  const int N = input.dim(0);
  const int H_IN = input.dim(1);
  const int W_IN = input.dim(2);
  const int C_IN = input.dim(3);

  Dest output;

  const int C_OUT = output.dim(3);

  const int K_H = weights.dim(1);
  const int K_W = weights.dim(2);

  const int S_H = stride[0];
  const int S_W = stride[1];

  const int pt = padding[0];
  const int pb = padding[1];
  const int pl = padding[2];
  const int pr = padding[3];

  const int H_PAD = pt + H_IN + pb;
  const int W_PAD = pl + W_IN + pr;

  // Convolution
  for (int n = 0; n < N; n++) {
    for (int h_pad = 0; h_pad < H_PAD - K_H + 1; h_pad += S_H) {
      for (int w_pad = 0; w_pad < W_PAD - K_W + 1; w_pad += S_W) {
        for (int kh = 0; kh < K_H; kh++) {
          for (int kw = 0; kw < K_W; kw++) {
            for (int c_in = 0; c_in < C_IN; c_in++) {
              for (int c_out = 0; c_out < C_OUT; c_out++) {
                const int h_out = h_pad / S_H;
                const int w_out = w_pad / S_W;
                const int h_in = h_pad - pt + kh;
                const int w_in = w_pad - pl + kw;

                if (h_in < 0 || h_in >= H_IN || w_in < 0 || w_in >= W_IN)
                  continue;

                output(n, h_out, w_out, c_out) +=
                    input(n, h_in, w_in, c_in) * weights(c_out, kh, kw, c_in);
              }
            }
          }
        }
      }
    }
  }

  return output;
}
#endif

// DepthwiseConv2DOp
template <typename Dest, typename Src, typename Weights>
Dest depthwise_conv2d(Src input, Weights weights, Tensor1D<int64_t, 4> padding,
                      Tensor1D<int64_t, 2> stride,
                      Tensor1D<int64_t, 2> dilation) {
  // Input is [N,H_IN,W_IN,C_IN], weights
  // are [K_H,K_W,C_IN,M] and output is [N,H,W,C_IN*M].
  static_assert(is_tensor_of_dim<4, Src>::value,
                "Expected 4 dimensional input");
  static_assert(is_tensor_of_dim<4, Dest>::value,
                "Expected 4 dimensional output");
  static_assert(is_tensor_of_dim<4, Weights>::value,
                "Expected 4 dimensional weights");

  // Check dimensions
  static_assert(Src::dim(3) == Weights::dim(2),
                "Input channels must equal weights channels");
  static_assert(Src::dim(0) == Dest::dim(0), "Batch sizes must be equal");
  static_assert(Dest::dim(3) % Src::dim(3) == 0,
                "Output channels need to be a multiple of input channels");
  static_assert(
      Dest::dim(3) == Src::dim(3) * Weights::dim(3),
      "Output channels size must be input channels times channel multiplier");

  assert(stride[0] > 0);
  assert(stride[1] > 0);

  assert(dilation[0] == 1);
  assert(dilation[1] == 1);

  const int N = input.dim(0);
  const int H_IN = input.dim(1);
  const int W_IN = input.dim(2);
  const int C_IN = input.dim(3);

  Dest output;

  const int K_H = weights.dim(0);
  const int K_W = weights.dim(1);
  const int M = weights.dim(3);

  const int S_H = stride[0];
  const int S_W = stride[1];

  const int pt = padding[0];
  const int pb = padding[1];
  const int pl = padding[2];
  const int pr = padding[3];

  const int H_PAD = pt + H_IN + pb;
  const int W_PAD = pl + W_IN + pr;

  // Convolution
  for (int n = 0; n < N; ++n) {
    for (int h_pad = 0; h_pad < H_PAD - K_H + 1; h_pad += S_H) {
      for (int w_pad = 0; w_pad < W_PAD - K_W + 1; w_pad += S_W) {
        for (int kh = 0; kh < K_H; ++kh) {
          for (int kw = 0; kw < K_W; ++kw) {
            for (int c_in = 0; c_in < C_IN; ++c_in) {
              for (int m = 0; m < M; ++m) {
                const int h_out = h_pad / S_H;
                const int w_out = w_pad / S_W;
                const int c_out = c_in * M + m;
                const int h_in = h_pad - pt + kh;
                const int w_in = w_pad - pl + kw;

                if (h_in < 0 || h_in >= H_IN || w_in < 0 || w_in >= W_IN)
                  continue;

                // For depthwise convolution we interpret weights as a tensor
                // with shape [filter_height, filter_width, 1, in_channels *
                // channel_multiplier]. So we need to calculate the index
                // using these dimensions.
                const size_t weights_index = emitc::utility::ravel_index<
                    Weights::dim(0), Weights::dim(1), 1,
                    Weights::dim(2) * Weights::dim(3)>(kh, kw, 0, c_out);

                output(n, h_out, w_out, c_out) +=
                    input(n, h_in, w_in, c_in) * weights[weights_index];
              }
            }
          }
        }
      }
    }
  }

  return output;
}

// FullyConnectedOp
template <typename Dest, typename Src, typename Weights, typename Bias>
Dest fully_connected(Src input, Weights weights, Bias bias) {
  static_assert(is_tensor_of_dim<2, Src>::value,
                "Expected 2 dimensional input");
  static_assert(is_tensor_of_dim<2, Dest>::value,
                "Expected 2 dimensional output");
  static_assert(is_tensor_of_dim<2, Weights>::value,
                "Expected 2 dimensional weights");
  static_assert(is_tensor_of_dim<1, Bias>::value,
                "Expected 1 dimensional bias");

  Dest output;
  static_assert(input.dim(0) == output.dim(0),
                "Output and input batch dimension do not match.");
  static_assert(input.dim(1) == weights.dim(1),
                "Input and weights dimensions do not match.");
  static_assert(output.dim(1) == weights.dim(0),
                "Output and weights dimensions do not match.");
  static_assert(weights.dim(0) == bias.dim(0),
                "Bias and weights dimensions do not match.");

  const size_t N = input.dim(0);
  const size_t C_IN = input.dim(1);
  const size_t C_OUT = weights.dim(0);

  for (size_t n = 0; n < N; ++n) {
    for (size_t c_out = 0; c_out < C_OUT; ++c_out) {
      for (size_t c_in = 0; c_in < C_IN; ++c_in) {
        auto in = input(n, c_in);
        auto weight = weights(c_out, c_in);
        output(n, c_out) += in * weight;
      }
      output(n, c_out) += bias(c_out);
    }
  }
  return output;
}

// MatMulOp
template <typename T, size_t B, size_t M, size_t K, size_t N>
Tensor3D<T, B, M, N> matmul(Tensor3D<T, B, M, K> a, Tensor3D<T, B, K, N> b) {
  return emitc::batch_matmul<Tensor3D<T, B, M, N>>(a, b);
}

namespace {
// Common reduce function used by specialized TOSA reduce ops.
template <typename Dest, typename Src, typename Computation>
inline Dest reduce(Src operand, typename get_element_type<Src>::type initValue,
                   int64_t dimension, Computation computation) {
  static_assert(is_tensor<Src>::value, "Expected tensor argument");
  static_assert(is_tensor<Dest>::value, "Expected tensor result");

  using ET_Src = typename get_element_type<Src>::type;
  using ET_Dest = typename get_element_type<Dest>::type;

  static_assert(std::is_same<ET_Src, ET_Dest>::value, "Element type mismatch");

  static_assert(Src::rank() == Dest::rank() + 1,
                "source rank must equal dest rank + 1");

  std::vector<size_t> retainedDimensions(Src::rank());
  std::iota(retainedDimensions.begin(), retainedDimensions.end(), 0);
  retainedDimensions.erase(retainedDimensions.begin() + dimension);

  assert(retainedDimensions.size() == Dest::rank());

  Dest result;
  std::fill(result.begin(), result.end(), initValue);

  for (size_t i = 0; i < operand.size(); ++i) {
    auto value = operand[i];
    auto index = operand.unravel_index(i);

    std::array<size_t, Dest::rank()> reducedIndex;
    size_t j = 0;
    for (size_t dim : retainedDimensions) {
      reducedIndex[j++] = index[dim];
    }

    auto reductionValue = result[result.ravel_index(reducedIndex)];
    result[result.ravel_index(reducedIndex)] =
        computation(reductionValue, value);
  }

  return result;
}
} // namespace

// ArgMaxOp
template <typename Dest, typename Src>
inline Dest argmax(Src operand, int64_t dimension) {
  static_assert(is_tensor<Src>::value, "Expected tensor argument");
  static_assert(is_tensor<Dest>::value, "Expected tensor result");

  using ET_Src = typename get_element_type<Src>::type;

  static_assert(Src::rank() == Dest::rank() + 1,
                "source rank must equal dest rank + 1");

  std::vector<size_t> retainedDimensions(Src::rank());
  std::iota(retainedDimensions.begin(), retainedDimensions.end(), 0);
  retainedDimensions.erase(retainedDimensions.begin() + dimension);

  assert(retainedDimensions.size() == Dest::rank());

  Dest result;
  typename replace_element_type<ET_Src, Dest>::type maxValues;

  std::fill(maxValues.begin(), maxValues.end(),
            std::numeric_limits<ET_Src>::min());

  for (size_t i = 0; i < operand.size(); ++i) {
    auto value = operand[i];
    auto index = operand.unravel_index(i);

    std::array<size_t, Dest::rank()> reducedIndex;
    size_t j = 0;
    for (size_t dim : retainedDimensions) {
      reducedIndex[j++] = index[dim];
    }

    auto destIndex = result.ravel_index(reducedIndex);

    if (value > maxValues[destIndex]) {
      maxValues[destIndex] = value;
      result[destIndex] = index[dimension];
    }
  }

  return result;
}

// ReduceAllOp
template <typename Dest, typename Src>
inline Dest reduce_all(Src input, int64_t dimension) {
  // ReduceAllOp takes only tensors with datatype bool according to the
  // TOSA specifications.
  using ET_Src = typename get_element_type<Src>::type;
  using ET_Dest = typename get_element_type<Dest>::type;

  static_assert(std::is_same<ET_Src, bool>::value,
                "Src tensor type must be bool");
  static_assert(std::is_same<ET_Dest, bool>::value,
                "Dest tensor type must be bool");

  auto and_ = [](ET_Src a, ET_Src b) { return (a && b); };

  return tosa::reduce<Dest, Src>(input, true, dimension, and_);
}

// ReduceAnyOp
template <typename Dest, typename Src>
inline Dest reduce_any(Src input, int64_t dimension) {
  // ReduceAnyOp takes only tensors with datatype bool according to the
  // TOSA specifications.
  using ET_Src = typename get_element_type<Src>::type;
  using ET_Dest = typename get_element_type<Dest>::type;

  static_assert(std::is_same<ET_Src, bool>::value,
                "Src tensor type must be bool");
  static_assert(std::is_same<ET_Dest, bool>::value,
                "Dest tensor type must be bool");

  auto or_ = [](ET_Src a, ET_Src b) { return a || b; };

  return tosa::reduce<Dest, Src>(input, false, dimension, or_);
}

// ReduceMaxOp
template <typename Dest, typename Src>
inline Dest reduce_max(Src input, int64_t dimension) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f =
      static_cast<const ET_Src &(*)(const ET_Src &, const ET_Src &)>(std::max);

  return tosa::reduce<Dest, Src>(input, std::numeric_limits<ET_Src>::min(),
                                 dimension, f);
}

// ReduceMinOp
template <typename Dest, typename Src>
inline Dest reduce_min(Src input, int64_t dimension) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f =
      static_cast<const ET_Src &(*)(const ET_Src &, const ET_Src &)>(std::min);

  return tosa::reduce<Dest, Src>(input, std::numeric_limits<ET_Src>::max(),
                                 dimension, f);
}

// ReduceProdOp
template <typename Dest, typename Src>
inline Dest reduce_prod(Src input, int64_t dimension) {
  using ET_Src = typename get_element_type<Src>::type;

  return tosa::reduce<Dest, Src>(input, 1, dimension,
                                 std::multiplies<ET_Src>{});
}

// ReduceSumOp
template <typename Dest, typename Src>
inline Dest reduce_sum(Src input, int64_t dimension) {
  using ET_Src = typename get_element_type<Src>::type;

  return tosa::reduce<Dest, Src>(input, 0, dimension, std::plus<ET_Src>{});
}

// ReshapeOp
template <typename Dest, typename Src>
inline Dest reshape(Src x) {
  return emitc::reshape<Dest>(x);
}

// SliceOp
template <typename Dest, typename Src>
Dest slice(Src x, Tensor<int64_t, Src::rank()> start_indices,
           Tensor<int64_t, Src::rank()> slice_sizes) {
  Tensor<int64_t, Src::rank()> limit_indices =
      emitc::add(start_indices, slice_sizes);
  Tensor<int64_t, Src::rank()> strides =
      emitc::tensor::splat<Tensor<int64_t, Src::rank()>>(1);
  return emitc::slice<Dest, Src>(x, start_indices, limit_indices, strides);
}

// PadOp
template <typename Dest, typename Src, typename Padding>
inline Dest pad(Src operand, Padding padding,
                Tensor0D<typename get_element_type<Src>::type> pad_const =
                    Tensor0D<typename get_element_type<Src>::type>{0}) {
  using ET_Padding = typename get_element_type<Padding>::type;

  static_assert(is_tensor<Dest>::value, "Expected tensor result");
  static_assert(is_tensor<Src>::value, "Expected tensor argument");
  static_assert(is_tensor<Padding>::value, "Expected tensor argument");

  static_assert(Padding::rank() == 2, "Padding must have rank 2");
  static_assert(Padding::dim(0) == Src::rank(),
                "Dimension 1 of padding must equal source rank");
  static_assert(Padding::dim(1) == 2, "Dimension 2 of padding is must be 2");

  // This check is not needed in a conversion pipeline since this would be
  // already illegal IR. Might be helpful for unittests, etc.
  static_assert(std::is_same<ET_Padding, int32_t>::value ||
                    std::is_same<ET_Padding, int64_t>::value,
                "Padding element type must be i32 or i64");

  // Create arguments for emitc::pad
  Tensor<int64_t, Src::rank()> edge_padding_low;
  Tensor<int64_t, Src::rank()> edge_padding_high;

  for (unsigned int i = 0; i < padding.dim(0); ++i) {
    edge_padding_low(i) = padding(i, 0);
    edge_padding_high(i) = padding(i, 1);
  }

  // Fill with zeros
  Tensor<int64_t, Src::rank()> interior_padding;
  std::fill(interior_padding.begin(), interior_padding.end(), 0);

  return emitc::pad<Dest>(operand, pad_const, edge_padding_low,
                          edge_padding_high, interior_padding);
}

// TileOp
// Overload for 1d case
template <typename Dest, typename Src, IsTensorOfDim<1, Dest> = true>
Dest tile(Src input, Tensor1D<int32_t, 1> multiples) {
  Dest result;
  auto it = result.begin();
  for (int32_t i = 0; i < multiples[0]; i++) {
    it = std::copy(input.begin(), input.end(), it);
  }
  return result;
}

// Overload for 2d case
template <typename Dest, typename Src, IsTensorOfDim<2, Src> = true>
Dest tile(Src input, Tensor1D<int32_t, 2> multiples) {
  Dest result;
  auto it = result.begin();
  for (int32_t i = 0; i < multiples[0]; i++) {
    for (int32_t j = 0; j < Src::dim(0); j++) {
      for (int32_t k = 0; k < multiples[1]; k++) {
        auto start = input.begin() + j * Src::dim(1);
        auto end = start + Src::dim(1);
        it = std::copy(start, end, it);
      }
    }
  }
  return result;
}

// Overload for 3d case
template <typename Dest, typename Src, IsTensorOfDim<3, Src> = true>
Dest tile(Src input, Tensor1D<int32_t, 3> multiples) {
  Dest result;
  auto it = result.begin();
  for (int32_t m0 = 0; m0 < multiples[0]; m0++) {
    for (int32_t d0 = 0; d0 < Src::dim(0); d0++) {
      for (int32_t m1 = 0; m1 < multiples[1]; m1++) {
        for (int32_t d1 = 0; d1 < Src::dim(1); d1++) {
          for (int32_t m2 = 0; m2 < multiples[2]; m2++) {
            auto start = input.begin() + (d0 * Src::dim(1) + d1) * Src::dim(2);
            auto end = start + Src::dim(2);
            it = std::copy(start, end, it);
          }
        }
      }
    }
  }
  return result;
}

// Overload for 4d case
template <typename Dest, typename Src, IsTensorOfDim<4, Src> = true>
Dest tile(Src input, Tensor1D<int32_t, 4> multiples) {
  Dest result;
  auto it = result.begin();
  for (int32_t m0 = 0; m0 < multiples[0]; m0++) {
    for (int32_t d0 = 0; d0 < Src::dim(0); d0++) {
      for (int32_t m1 = 0; m1 < multiples[1]; m1++) {
        for (int32_t d1 = 0; d1 < Src::dim(1); d1++) {
          for (int32_t m2 = 0; m2 < multiples[2]; m2++) {
            for (int32_t d2 = 0; d2 < Src::dim(2); d2++) {
              for (int32_t m3 = 0; m3 < multiples[3]; m3++) {
                auto start =
                    input.begin() +
                    ((d0 * Src::dim(1) + d1) * Src::dim(2) + d2) * Src::dim(3);
                auto end = start + Src::dim(3);
                it = std::copy(start, end, it);
              }
            }
          }
        }
      }
    }
  }
  return result;
}

// TransposeOp
// Maps the perms dimension from Dest to Src.
template <typename Dest, typename Src>
inline Dest transpose(Src operand, Tensor1D<int64_t, Src::rank()> perms) {
  static_assert(is_tensor<Src>::value, "Expected tensor argument");
  static_assert(is_tensor<Dest>::value, "Expected tensor result");

  // Since emitc::broadcast_in_dim maps the dimensions (argument
  // "broadcast_dimensions") from Src to Dest and tosa::transpose maps the
  // dimensions (argument "perms") from Dest to Src, we have to invert the
  // mapping.
  Tensor1D<int64_t, Src::rank()> broadcast_dimensions;
  for (size_t i = 0; i < perms.size(); ++i) {
    auto pos = std::find(perms.begin(), perms.end(), i);
    assert(pos != std::end(perms));
    int64_t index = std::distance(perms.begin(), pos);
    broadcast_dimensions[i] = index;
  }
  return emitc::broadcast_in_dim<Dest>(operand, broadcast_dimensions);
}

// TransposeOp allows perms to be of type int32_t or int64_t.
template <typename Dest, typename Src>
inline Dest transpose(Src input, Tensor1D<int32_t, Src::rank()> perms) {
  Tensor1D<int64_t, Src::rank()> permsInt64;
  for (size_t i = 0; i < perms.size(); ++i) {
    permsInt64[i] = static_cast<int64_t>(perms[i]);
  }
  return tosa::transpose<Dest>(input, permsInt64);
}

} // namespace tosa
} // namespace emitc

#endif // EMITC_TOSA_H
