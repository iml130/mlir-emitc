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

// This file defines functions emitted by TosaToEmitC

#ifndef EMITC_EMITC_TOSA_H
#define EMITC_EMITC_TOSA_H

#include <limits>

#include "emitc_core_ops.h"

namespace tosa {

/// Unary elementwise ops
// AbsOp
template <typename Src>
inline Src abs(Src x) {
  return emitc::abs<Src>(x);
}

// CeilOp
template <typename Src>
inline Src ceil(Src x) {
  return emitc::ceil<Src>(x);
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

// ReciprocalOp
template <typename Src>
inline Src reciprocal(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = [](ET_Src element) { return (static_cast<ET_Src>(1.0) / element); };

  return unary<Src>(x, f);
}

// TanhOp
template <typename Src>
inline Src tanh(Src x) {
  return emitc::tanh<Src>(x);
}

/// Binary elementwise ops
// AddOp
template <typename Src>
inline Src add(Src x, Src y) {
  return emitc::add<Src>(x, y);
}

// MulOp
template <typename Src>
inline Src mul(Src x, Src y) {
  return emitc::mul(x, y);
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

/// Other ops
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

/// Reduce ops
// ReduceOp
template <typename Dest, typename Src, typename Computation>
inline Dest reduce(Src operand, typename get_element_type<Src>::type initValue,
                   int64_t dimension, Computation computation) {
  static_assert(is_tensor<Src>::value, "Expected tensor argument");
  static_assert(is_tensor<Dest>::value, "Expected tensor result");

  using ET_Src = typename get_element_type<Src>::type;
  using ET_Dest = typename get_element_type<Src>::type;

  static_assert(std::is_same<ET_Src, ET_Dest>::value, "Element type mismatch");

  static_assert(Src::rank() == Dest::rank() + 1,
                "source rank must equal dest rank + 1");

  std::vector<size_t> retainedDimensions(Src::rank());
  std::iota(retainedDimensions.begin(), retainedDimensions.end(), 0);
  retainedDimensions.erase(retainedDimensions.begin() + dimension);

  assert(retainedDimensions.size() == Dest::rank());

  Dest result;
  std::fill(result.begin(), result.end(), initValue);

  for (size_t i = 0; i < operand.size(); i++) {
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

// ReduceAllOp
template <typename Dest, typename Src>
inline Dest reduce_all(Src input, int64_t dimension) {
  using ET_Src = typename get_element_type<Src>::type;
  using ET_Dest = typename get_element_type<Src>::type;

  static_assert(std::is_same<ET_Src, bool>::value,
                "Src tensor type must be bool");
  static_assert(std::is_same<ET_Dest, bool>::value,
                "Dest tensor type must be bool");

  auto and_ = [](ET_Src a, ET_Src b) { return (a && b); };

  return tosa::reduce<Dest, Src>(input, true, dimension, and_);
}

// reduce_any
template <typename Dest, typename Src>
inline Dest reduce_any(Src input, int64_t dimension) {
  using ET_Src = typename get_element_type<Src>::type;
  using ET_Dest = typename get_element_type<Src>::type;

  static_assert(std::is_same<ET_Src, bool>::value,
                "Src tensor type must be bool");
  static_assert(std::is_same<ET_Dest, bool>::value,
                "Dest tensor type must be bool");

  auto or_ = [](ET_Src a, ET_Src b) { return a || b; };

  return tosa::reduce<Dest, Src>(input, false, dimension, or_);
}

// reduce_max
template <typename Dest, typename Src>
inline Dest reduce_max(Src input, int64_t dimension) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = [](ET_Src a, ET_Src b) { return std::max(a, b); };

  return tosa::reduce<Dest, Src>(input, std::numeric_limits<ET_Src>::min(),
                                 dimension, f);
}

// reduce_min
template <typename Dest, typename Src>
inline Dest reduce_min(Src input, int64_t dimension) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = [](ET_Src a, ET_Src b) { return std::min(a, b); };

  return tosa::reduce<Dest, Src>(input, std::numeric_limits<ET_Src>::max(),
                                 dimension, f);
}

// reduce_prod
template <typename Dest, typename Src>
inline Dest reduce_prod(Src input, int64_t dimension) {
  using ET_Src = typename get_element_type<Src>::type;

  return tosa::reduce<Dest, Src>(input, 1, dimension,
                                 std::multiplies<ET_Src>{});
}

// reduce_sum
template <typename Dest, typename Src>
inline Dest reduce_sum(Src input, int64_t dimension) {
  using ET_Src = typename get_element_type<Src>::type;

  return tosa::reduce<Dest, Src>(input, 0, dimension, std::plus<ET_Src>{});
}

} // namespace tosa

#endif // EMITC_EMITC_TOSA_H
