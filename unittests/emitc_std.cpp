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

#include "emitc_std.h"
#include "gmock/gmock.h"

namespace {

TEST(std, extract_element) {
  std::vector<int> v1 = {1, 2, 3};
  EXPECT_EQ(1, standard::extract_element(v1));
  EXPECT_EQ(2, standard::extract_element(v1, 1));
}

TEST(std, index_cast) {
  uint32_t a = 1;
  uint64_t b = 1;
  EXPECT_EQ(b, standard::index_cast<uint64_t>(a));

  std::vector<uint32_t> v1 = {1, 2};
  std::vector<uint64_t> v2 = {1, 2};
  EXPECT_EQ(v2, standard::index_cast<uint64_t>(v1));
}

} // namespace
