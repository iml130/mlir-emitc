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

// This file defines alternative implementations for the functions in
// emitc_tosa.h utilizing Eigen.

#ifndef EMITC_EMITC_TOSA_EIGEN_H
#define EMITC_EMITC_TOSA_EIGEN_H

#include "emitc_std.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace {

// A view on an emitc tensor as Eigen tensor in row-major order
template <typename T, size_t... Shape>
inline auto as_eigen(Tensor<T, Shape...> &t) {
  return Eigen::TensorMap<Eigen::Tensor<T, sizeof...(Shape), Eigen::RowMajor>>(
      &*t.begin(), Shape...);
}

} // namespace

namespace emitc {
namespace tosa {

// Conv2DOp
template <typename Dest, typename Src, typename Weights>
Dest conv2d(Src input, Weights weights, Tensor1D<int64_t, 4> padding,
            Tensor1D<int64_t, 2> stride, Tensor1D<int64_t, 2> dilation) {
  // Input is [N,IH,IW,IC], weights are [OC,KH,KW,IC] and output is [N,H,W,OC]
  static_assert(is_tensor_of_dim<4, Src>::value,
                "Expected 4 dimensional input");
  static_assert(is_tensor_of_dim<4, Dest>::value,
                "Expected 4 dimensional output");
  static_assert(is_tensor_of_dim<4, Weights>::value,
                "Expected 4 dimensional weights");

  constexpr Eigen::Index N = Src::dim(0);
  constexpr Eigen::Index IC = Src::dim(3);
  constexpr Eigen::Index KF = Weights::dim(0);
  constexpr Eigen::Index KH = Weights::dim(1);
  constexpr Eigen::Index KW = Weights::dim(2);
  constexpr Eigen::Index KC = Weights::dim(3);
  constexpr Eigen::Index ON = Dest::dim(0);
  constexpr Eigen::Index H = Dest::dim(1);
  constexpr Eigen::Index W = Dest::dim(2);
  constexpr Eigen::Index OC = Dest::dim(3);

  static_assert(N == ON, "Expected input batch size to match output");
  static_assert(IC == KC, "Expected input channels to match weights");
  static_assert(OC == KF, "Expected output channels to match weights");

  const int64_t pt = padding[0];
  const int64_t pb = padding[1];
  const int64_t pl = padding[2];
  const int64_t pr = padding[3];
  const int64_t SH = stride[0];
  const int64_t SW = stride[1];
  const int64_t DH = dilation[0];
  const int64_t DW = dilation[1];

  Dest output;
  // [N,IH,IW,IC]
  auto e_input = as_eigen(input);

  // [KH,KW,IC,OC]
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
  auto e_weight =
      as_eigen(weights).shuffle(Eigen::array<Eigen::Index, 4>({1, 2, 3, 0}));
#else
  Eigen::Tensor<typename Weights::value_type, 4, Eigen::RowMajor> e_weight =
      as_eigen(weights).shuffle(Eigen::array<Eigen::Index, 4>({1, 2, 3, 0}));
#endif

  // [N,H,W,OC]
  auto e_output = as_eigen(output);

  // apply padding to input [N,IH+pt+pb,IW+pl+pr,IC]
  auto input_pad = e_input.pad(Eigen::array<std::pair<int64_t, int64_t>, 4>{
      std::make_pair(0, 0), std::make_pair(pt, pb), std::make_pair(pl, pr),
      std::make_pair(0, 0)});

  // create tensor containing input patches [N,H*W,KH,KW,IC]
  auto patches = input_pad.extract_image_patches(KW, KH, SW, SH, DW, DH,
                                                 Eigen::PADDING_VALID);

  // create 2d tensor from patches [N*H*W,KH*KW*IC]
  auto patches_m =
      patches.reshape(Eigen::DSizes<Eigen::Index, 2>{N * H * W, KH * KW * IC});

  // create 2d tensor from weights [KH*KW*IC,OC]
  auto weight_m =
      e_weight.reshape(Eigen::DSizes<Eigen::Index, 2>{KH * KW * IC, OC});

  // multiply [N*H*W,OC]
  auto contr = patches_m.contract(
      weight_m, Eigen::array<Eigen::IndexPair<Eigen::Index>, 1>{
                    Eigen::IndexPair<Eigen::Index>(1, 0)});

  // reshape result to output [N,H,W,OC]
  e_output = contr.reshape(Eigen::DSizes<Eigen::Index, 4>{N, H, W, OC});

  return output;
}

} // namespace tosa
} // namespace emitc

#endif // EMITC_EMITC_TOSA_EIGEN_H
