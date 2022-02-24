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

#include "emitc/tosa.h"
#include "emitc/types.h"

namespace {

using namespace emitc;
using ::testing::DoubleEq;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pointwise;

const float EPSILON = 5e-4;

// Unary elementwise ops
TEST(tosa, clamp) {
  {
    Tensor0D<int32_t> operand{-1};
    int32_t min_value = -3;
    int32_t max_value = -3;
    Tensor0D<int32_t> expected_result{-3};
    Tensor0D<int32_t> result = tosa::clamp(operand, min_value, max_value);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor1D<double, 2> operand{-4.7, 1.3};
    double min_value = -3.25;
    double max_value = -3.25;
    Tensor1D<double, 2> expected_result{-3.25, -3.25};
    Tensor1D<double, 2> result = tosa::clamp(operand, min_value, max_value);

    EXPECT_THAT(result, Pointwise(DoubleEq(), expected_result));
  }
  {
    Tensor2D<float, 1, 2> operand{-1.5f, 5.0f};
    float min_value = -1.0f;
    float max_value = 3.0f;
    Tensor2D<float, 1, 2> expected_result{-1.0, 3.0f};
    Tensor2D<float, 1, 2> result = tosa::clamp(operand, min_value, max_value);

    EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));
  }
  {
    Tensor3D<int64_t, 4, 2, 1> operand{-2, 2, -2, 3, 4, -5, 5, 5};
    int64_t min_value = 1;
    int64_t max_value = 3;
    Tensor3D<int64_t, 4, 2, 1> expected_result{1, 2, 1, 3, 3, 1, 3, 3};
    Tensor3D<int64_t, 4, 2, 1> result =
        tosa::clamp(operand, min_value, max_value);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor4D<float, 1, 2, 3, 2> operand{-1.0, -0.5, 0.0, 2.0625, 2.875, 2.1875,
                                        2.5,  -0.0, 2.0, 3.0,    -99.0, 99.0};
    float min_value = 2.125;
    float max_value = 2.875;
    Tensor4D<float, 1, 2, 3, 2> expected_result{2.125, 2.125,  2.125, 2.125,
                                                2.875, 2.1875, 2.5,   2.125,
                                                2.125, 2.875,  2.125, 2.875};
    Tensor4D<float, 1, 2, 3, 2> result =
        tosa::clamp(operand, min_value, max_value);

    EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));
  }
}

TEST(tosa, clz) {
  {
    Tensor0D<int32_t> x{0};
    Tensor0D<int32_t> expected_result{32};
    Tensor0D<int32_t> result = tosa::clz(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor1D<int32_t, 2> x{1, -1};
    Tensor1D<int32_t, 2> expected_result{31, 0};
    Tensor1D<int32_t, 2> result = tosa::clz(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor2D<int32_t, 3, 2> x{-1328632289, 1459158945, -1912283137,
                              627316066,   59808247,   42};
    Tensor2D<int32_t, 3, 2> expected_result{0, 1, 0, 2, 6, 26};
    Tensor2D<int32_t, 3, 2> result = tosa::clz(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor3D<int32_t, 2, 1, 2> x{0xC000, -0x7FFFFFFF, 132486298, -2104906602};
    Tensor3D<int32_t, 2, 1, 2> expected_result{16, 0, 5, 0};
    Tensor3D<int32_t, 2, 1, 2> result = tosa::clz(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor4D<int32_t, 1, 2, 2, 2> x{21845, 10922, 5461, 2730,
                                    1365,  682,   341,  170};
    Tensor4D<int32_t, 1, 2, 2, 2> expected_result{17, 18, 19, 20,
                                                  21, 22, 23, 24};
    Tensor4D<int32_t, 1, 2, 2, 2> result = tosa::clz(x);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
}

TEST(tosa, reciprocal) {
  {
    Tensor0D<float> x{1.0f};
    Tensor0D<float> expected_result{1.0f};
    Tensor0D<float> result = tosa::reciprocal(x);

    EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));
  }
  {
    Tensor1D<double, 2> x{6.312247e+64, -9.053782e-32};
    Tensor1D<double, 2> expected_result{1.5842219102009158e-65,
                                        -1.1045108000170537e+31};
    Tensor1D<double, 2> result = tosa::reciprocal(x);

    EXPECT_THAT(result, Pointwise(DoubleEq(), expected_result));
  }
  {
    Tensor2D<float, 3, 2> x{1.393225e+27f, -1.151362e-12f, -5.340778e+5f,
                            1.346074e+6f,  1.373985f,      9.198730e+7f};
    Tensor2D<float, 3, 2> expected_result{7.177592e-28f, -8.685366e+11f,
                                          -1.872386e-6f, 7.429012e-7f,
                                          7.278100e-1f,  1.087107e-8f};
    Tensor2D<float, 3, 2> result = tosa::reciprocal(x);

    EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));
  }
  {
    Tensor3D<double, 2, 1, 2> x{-1.857135e-3, 3.523054e-5, 1.704234e+59,
                                -7.043905e-21};
    Tensor3D<double, 2, 1, 2> expected_result{
        -5.384638165776855e+2, 2.838446416092402e+4, 5.867738819903839e-60,
        -1.4196670738745057e+20};
    Tensor3D<double, 2, 1, 2> result = tosa::reciprocal(x);

    EXPECT_THAT(result, Pointwise(DoubleEq(), expected_result));
  }
  {
    Tensor4D<float, 1, 2, 2, 2> x{-2.524463e+22f, -5.496311e-5f, -1.025806e-2f,
                                  2.648090e-10f,  7.170789f,     2.227768e-26f,
                                  2.188774e+17f,  5.150893f};
    Tensor4D<float, 1, 2, 2, 2> expected_result{
        -3.961238e-23f, -1.819402e+4f, -9.748432e+1f, 3.776307e+9f,
        1.394547e-1f,   4.488798e+25f, 4.568768e-18f, 1.941411e-1f};
    Tensor4D<float, 1, 2, 2, 2> result = tosa::reciprocal(x);

    EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));
  }
}

TEST(tosa, reluN) {
  {
    Tensor0D<int32_t> operand{0};
    int32_t max_value = 0;
    Tensor0D<int32_t> expected_result{0};
    Tensor0D<int32_t> result = tosa::reluN(operand, max_value);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor1D<double, 2> operand{-4.7, 1.3};
    double max_value = 1.4;
    Tensor1D<double, 2> expected_result{0, 1.3};
    Tensor1D<double, 2> result = tosa::reluN(operand, max_value);

    EXPECT_THAT(result, Pointwise(DoubleEq(), expected_result));
  }
  {
    Tensor2D<float, 2, 2> operand{0.0f, -9.9f, 4.4f, 8.8f};
    float max_value = 5.5f;
    Tensor2D<float, 2, 2> expected_result{0.0f, 0.0f, 4.4f, 5.5f};
    Tensor2D<float, 2, 2> result = tosa::reluN(operand, max_value);

    EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));
  }
  {
    Tensor3D<int64_t, 3, 2, 1> operand{4, 1, -1, 3, 0, 2};
    int64_t max_value = 3;
    Tensor3D<int64_t, 3, 2, 1> expected_result{3, 1, 0, 3, 0, 2};
    Tensor3D<int64_t, 3, 2, 1> result = tosa::reluN(operand, max_value);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor4D<int16_t, 1, 2, 3, 2> operand{7812,  15481,  -30284, 30996,
                                          18736, 6699,   31903,  26229,
                                          15931, -18954, -27643, 19133};
    int16_t max_value = 20000;
    Tensor4D<int16_t, 1, 2, 3, 2> expected_result{
        7812, 15481, 0, 20000, 18736, 6699, 20000, 20000, 15931, 0, 0, 19133};
    Tensor4D<int16_t, 1, 2, 3, 2> result = tosa::reluN(operand, max_value);

    EXPECT_THAT(result, Pointwise(FloatEq(), expected_result));
  }
}

// Binary elementwise ops
TEST(tosa, arithmetic_right_shift) {
  {
    Tensor1D<int16_t, 5> in1{0b10, 0b10, -0b10, 0b1, -0b1};
    Tensor1D<int16_t, 5> in2{0, 1, 1, 1, 1};
    bool round = false;
    Tensor1D<int16_t, 5> expected_result{0b10, 0b1, -0b1, 0b0, -0b1};
    Tensor1D<int16_t, 5> result = tosa::arithmetic_right_shift(in1, in2, round);
    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor1D<int16_t, 4> in1{0b1, 0b1, 0b10, 0b110};
    Tensor1D<int16_t, 4> in2{0, 1, 1, 2};
    bool round = true;
    Tensor1D<int16_t, 4> expected_result{0b1, 0b1, 0b1, 0b10};
    Tensor1D<int16_t, 4> result = tosa::arithmetic_right_shift(in1, in2, round);
    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
}

TEST(tosa, logical_left_shift) {
  Tensor1D<int16_t, 4> s0{0b1, 0b1, -0b1, 0b101};
  Tensor1D<int16_t, 4> t0{0, 1, 1, 2};
  Tensor1D<int16_t, 4> expected_result{0b1, 0b10, -0b10, 0b10100};
  Tensor1D<int16_t, 4> result = tosa::logical_left_shift(s0, t0);
  EXPECT_THAT(result, Pointwise(Eq(), expected_result));
}

TEST(tosa, mul) {
  // no shift
  Tensor2D<long, 2, 2> s0{3, 1, 4, 9};
  Tensor2D<long, 2, 2> t0{-2, 8, 6, -10};

  auto lambda_2d = [&s0, &t0]() -> Tensor2D<long, 2, 2> {
    return tosa::mul(s0, t0);
  };
  EXPECT_THAT(lambda_2d(), Pointwise(Eq(), {-6, 8, 24, -90}));

  Tensor1D<int32_t, 1> s1{3};
  Tensor1D<int32_t, 1> t1{2};

  auto lambda_1d_int = [&s1, &t1]() -> Tensor1D<int32_t, 1> {
    return tosa::mul(s1, t1);
  };

  EXPECT_THAT(lambda_1d_int(), Pointwise(Eq(), {6}));

  // shift
  Tensor2D<int32_t, 2, 2> s2{1, 2, 3, 4};
  Tensor2D<int32_t, 2, 2> t2{1, 2, 3, 4};

  auto lambda_1d_int_shift = [&s2, &t2]() -> Tensor2D<int32_t, 2, 2> {
    int32_t shift{2};
    return tosa::mul(s2, t2, shift);
  };

  EXPECT_THAT(lambda_1d_int_shift(), Pointwise(Eq(), {0, 1, 2, 4}));
}

TEST(tosa, table) {
  // table tensors
  // f(n) = n
  Tensor1D<int8_t, 256> table_linear8{
      -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117,
      -116, -115, -114, -113, -112, -111, -110, -109, -108, -107, -106, -105,
      -104, -103, -102, -101, -100, -99,  -98,  -97,  -96,  -95,  -94,  -93,
      -92,  -91,  -90,  -89,  -88,  -87,  -86,  -85,  -84,  -83,  -82,  -81,
      -80,  -79,  -78,  -77,  -76,  -75,  -74,  -73,  -72,  -71,  -70,  -69,
      -68,  -67,  -66,  -65,  -64,  -63,  -62,  -61,  -60,  -59,  -58,  -57,
      -56,  -55,  -54,  -53,  -52,  -51,  -50,  -49,  -48,  -47,  -46,  -45,
      -44,  -43,  -42,  -41,  -40,  -39,  -38,  -37,  -36,  -35,  -34,  -33,
      -32,  -31,  -30,  -29,  -28,  -27,  -26,  -25,  -24,  -23,  -22,  -21,
      -20,  -19,  -18,  -17,  -16,  -15,  -14,  -13,  -12,  -11,  -10,  -9,
      -8,   -7,   -6,   -5,   -4,   -3,   -2,   -1,   0,    1,    2,    3,
      4,    5,    6,    7,    8,    9,    10,   11,   12,   13,   14,   15,
      16,   17,   18,   19,   20,   21,   22,   23,   24,   25,   26,   27,
      28,   29,   30,   31,   32,   33,   34,   35,   36,   37,   38,   39,
      40,   41,   42,   43,   44,   45,   46,   47,   48,   49,   50,   51,
      52,   53,   54,   55,   56,   57,   58,   59,   60,   61,   62,   63,
      64,   65,   66,   67,   68,   69,   70,   71,   72,   73,   74,   75,
      76,   77,   78,   79,   80,   81,   82,   83,   84,   85,   86,   87,
      88,   89,   90,   91,   92,   93,   94,   95,   96,   97,   98,   99,
      100,  101,  102,  103,  104,  105,  106,  107,  108,  109,  110,  111,
      112,  113,  114,  115,  116,  117,  118,  119,  120,  121,  122,  123,
      124,  125,  126,  127};
  // f(n) = n
  Tensor1D<int16_t, 513> table_linear16{
      -256, -255, -254, -253, -252, -251, -250, -249, -248, -247, -246, -245,
      -244, -243, -242, -241, -240, -239, -238, -237, -236, -235, -234, -233,
      -232, -231, -230, -229, -228, -227, -226, -225, -224, -223, -222, -221,
      -220, -219, -218, -217, -216, -215, -214, -213, -212, -211, -210, -209,
      -208, -207, -206, -205, -204, -203, -202, -201, -200, -199, -198, -197,
      -196, -195, -194, -193, -192, -191, -190, -189, -188, -187, -186, -185,
      -184, -183, -182, -181, -180, -179, -178, -177, -176, -175, -174, -173,
      -172, -171, -170, -169, -168, -167, -166, -165, -164, -163, -162, -161,
      -160, -159, -158, -157, -156, -155, -154, -153, -152, -151, -150, -149,
      -148, -147, -146, -145, -144, -143, -142, -141, -140, -139, -138, -137,
      -136, -135, -134, -133, -132, -131, -130, -129, -128, -127, -126, -125,
      -124, -123, -122, -121, -120, -119, -118, -117, -116, -115, -114, -113,
      -112, -111, -110, -109, -108, -107, -106, -105, -104, -103, -102, -101,
      -100, -99,  -98,  -97,  -96,  -95,  -94,  -93,  -92,  -91,  -90,  -89,
      -88,  -87,  -86,  -85,  -84,  -83,  -82,  -81,  -80,  -79,  -78,  -77,
      -76,  -75,  -74,  -73,  -72,  -71,  -70,  -69,  -68,  -67,  -66,  -65,
      -64,  -63,  -62,  -61,  -60,  -59,  -58,  -57,  -56,  -55,  -54,  -53,
      -52,  -51,  -50,  -49,  -48,  -47,  -46,  -45,  -44,  -43,  -42,  -41,
      -40,  -39,  -38,  -37,  -36,  -35,  -34,  -33,  -32,  -31,  -30,  -29,
      -28,  -27,  -26,  -25,  -24,  -23,  -22,  -21,  -20,  -19,  -18,  -17,
      -16,  -15,  -14,  -13,  -12,  -11,  -10,  -9,   -8,   -7,   -6,   -5,
      -4,   -3,   -2,   -1,   0,    1,    2,    3,    4,    5,    6,    7,
      8,    9,    10,   11,   12,   13,   14,   15,   16,   17,   18,   19,
      20,   21,   22,   23,   24,   25,   26,   27,   28,   29,   30,   31,
      32,   33,   34,   35,   36,   37,   38,   39,   40,   41,   42,   43,
      44,   45,   46,   47,   48,   49,   50,   51,   52,   53,   54,   55,
      56,   57,   58,   59,   60,   61,   62,   63,   64,   65,   66,   67,
      68,   69,   70,   71,   72,   73,   74,   75,   76,   77,   78,   79,
      80,   81,   82,   83,   84,   85,   86,   87,   88,   89,   90,   91,
      92,   93,   94,   95,   96,   97,   98,   99,   100,  101,  102,  103,
      104,  105,  106,  107,  108,  109,  110,  111,  112,  113,  114,  115,
      116,  117,  118,  119,  120,  121,  122,  123,  124,  125,  126,  127,
      128,  129,  130,  131,  132,  133,  134,  135,  136,  137,  138,  139,
      140,  141,  142,  143,  144,  145,  146,  147,  148,  149,  150,  151,
      152,  153,  154,  155,  156,  157,  158,  159,  160,  161,  162,  163,
      164,  165,  166,  167,  168,  169,  170,  171,  172,  173,  174,  175,
      176,  177,  178,  179,  180,  181,  182,  183,  184,  185,  186,  187,
      188,  189,  190,  191,  192,  193,  194,  195,  196,  197,  198,  199,
      200,  201,  202,  203,  204,  205,  206,  207,  208,  209,  210,  211,
      212,  213,  214,  215,  216,  217,  218,  219,  220,  221,  222,  223,
      224,  225,  226,  227,  228,  229,  230,  231,  232,  233,  234,  235,
      236,  237,  238,  239,  240,  241,  242,  243,  244,  245,  246,  247,
      248,  249,  250,  251,  252,  253,  254,  255,  256};
  // f(n) = 256 / (1 + exp(-0.05 * n)) - 128
  Tensor1D<int8_t, 256> table_logistic8{
      -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
      -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126,
      -126, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125,
      -125, -125, -125, -125, -124, -124, -124, -124, -124, -124, -123, -123,
      -123, -123, -122, -122, -122, -122, -121, -121, -121, -120, -120, -120,
      -119, -119, -118, -118, -117, -117, -116, -116, -115, -115, -114, -114,
      -113, -112, -111, -111, -110, -109, -108, -107, -106, -105, -104, -103,
      -102, -101, -100, -98,  -97,  -96,  -94,  -93,  -91,  -90,  -88,  -86,
      -84,  -83,  -81,  -79,  -77,  -75,  -73,  -70,  -68,  -66,  -64,  -61,
      -59,  -56,  -54,  -51,  -48,  -45,  -43,  -40,  -37,  -34,  -31,  -28,
      -25,  -22,  -19,  -15,  -12,  -9,   -6,   -3,   0,    3,    6,    9,
      12,   15,   19,   22,   25,   28,   31,   34,   37,   40,   43,   45,
      48,   51,   54,   56,   59,   61,   64,   66,   68,   70,   73,   75,
      77,   79,   81,   83,   84,   86,   88,   90,   91,   93,   94,   96,
      97,   98,   100,  101,  102,  103,  104,  105,  106,  107,  108,  109,
      110,  111,  111,  112,  113,  114,  114,  115,  115,  116,  116,  117,
      117,  118,  118,  119,  119,  120,  120,  120,  121,  121,  121,  122,
      122,  122,  122,  123,  123,  123,  123,  124,  124,  124,  124,  124,
      124,  125,  125,  125,  125,  125,  125,  125,  125,  126,  126,  126,
      126,  126,  126,  126,  126,  126,  126,  126,  126,  126,  126,  127,
      127,  127,  127,  127,  127,  127,  127,  127,  127,  127,  127,  127,
      127,  127,  127,  127};
  // f(n) = 32768 / (1 + exp(-0.03 * n)) - 16384
  Tensor1D<int16_t, 513> table_logistic16{
      -16368, -16368, -16367, -16367, -16366, -16366, -16365, -16365, -16364,
      -16364, -16363, -16362, -16362, -16361, -16360, -16360, -16359, -16358,
      -16358, -16357, -16356, -16355, -16354, -16353, -16352, -16351, -16351,
      -16350, -16348, -16347, -16346, -16345, -16344, -16343, -16342, -16340,
      -16339, -16338, -16336, -16335, -16333, -16332, -16330, -16329, -16327,
      -16325, -16323, -16322, -16320, -16318, -16316, -16314, -16312, -16309,
      -16307, -16305, -16302, -16300, -16297, -16295, -16292, -16289, -16287,
      -16284, -16281, -16277, -16274, -16271, -16267, -16264, -16260, -16257,
      -16253, -16249, -16245, -16241, -16236, -16232, -16227, -16222, -16217,
      -16212, -16207, -16202, -16196, -16191, -16185, -16179, -16173, -16166,
      -16160, -16153, -16146, -16139, -16131, -16124, -16116, -16108, -16100,
      -16091, -16082, -16073, -16064, -16054, -16044, -16034, -16023, -16013,
      -16001, -15990, -15978, -15966, -15953, -15940, -15927, -15913, -15899,
      -15885, -15870, -15855, -15839, -15822, -15806, -15788, -15771, -15752,
      -15733, -15714, -15694, -15673, -15652, -15631, -15608, -15585, -15561,
      -15537, -15512, -15486, -15460, -15432, -15404, -15375, -15346, -15315,
      -15284, -15251, -15218, -15184, -15149, -15112, -15075, -15037, -14998,
      -14957, -14916, -14873, -14829, -14784, -14738, -14691, -14642, -14592,
      -14540, -14487, -14433, -14377, -14320, -14261, -14201, -14139, -14075,
      -14010, -13943, -13875, -13804, -13732, -13658, -13582, -13504, -13425,
      -13343, -13259, -13173, -13085, -12995, -12903, -12809, -12712, -12613,
      -12512, -12408, -12302, -12194, -12083, -11970, -11854, -11735, -11614,
      -11491, -11365, -11236, -11104, -10970, -10833, -10693, -10551, -10406,
      -10258, -10107, -9953,  -9797,  -9638,  -9475,  -9310,  -9143,  -8972,
      -8799,  -8622,  -8443,  -8261,  -8077,  -7889,  -7699,  -7506,  -7311,
      -7113,  -6912,  -6709,  -6503,  -6295,  -6084,  -5871,  -5655,  -5438,
      -5218,  -4996,  -4772,  -4546,  -4319,  -4089,  -3858,  -3625,  -3390,
      -3154,  -2917,  -2679,  -2439,  -2198,  -1956,  -1714,  -1470,  -1226,
      -981,   -736,   -491,   -245,   0,      245,    491,    736,    981,
      1226,   1470,   1714,   1956,   2198,   2439,   2679,   2917,   3154,
      3390,   3625,   3858,   4089,   4319,   4546,   4772,   4996,   5218,
      5438,   5655,   5871,   6084,   6295,   6503,   6709,   6912,   7113,
      7311,   7506,   7699,   7889,   8077,   8261,   8443,   8622,   8799,
      8972,   9143,   9310,   9475,   9638,   9797,   9953,   10107,  10258,
      10406,  10551,  10693,  10833,  10970,  11104,  11236,  11365,  11491,
      11614,  11735,  11854,  11970,  12083,  12194,  12302,  12408,  12512,
      12613,  12712,  12809,  12903,  12995,  13085,  13173,  13259,  13343,
      13425,  13504,  13582,  13658,  13732,  13804,  13875,  13943,  14010,
      14075,  14139,  14201,  14261,  14320,  14377,  14433,  14487,  14540,
      14592,  14642,  14691,  14738,  14784,  14829,  14873,  14916,  14957,
      14998,  15037,  15075,  15112,  15149,  15184,  15218,  15251,  15284,
      15315,  15346,  15375,  15404,  15432,  15460,  15486,  15512,  15537,
      15561,  15585,  15608,  15631,  15652,  15673,  15694,  15714,  15733,
      15752,  15771,  15788,  15806,  15822,  15839,  15855,  15870,  15885,
      15899,  15913,  15927,  15940,  15953,  15966,  15978,  15990,  16001,
      16013,  16023,  16034,  16044,  16054,  16064,  16073,  16082,  16091,
      16100,  16108,  16116,  16124,  16131,  16139,  16146,  16153,  16160,
      16166,  16173,  16179,  16185,  16191,  16196,  16202,  16207,  16212,
      16217,  16222,  16227,  16232,  16236,  16241,  16245,  16249,  16253,
      16257,  16260,  16264,  16267,  16271,  16274,  16277,  16281,  16284,
      16287,  16289,  16292,  16295,  16297,  16300,  16302,  16305,  16307,
      16309,  16312,  16314,  16316,  16318,  16320,  16322,  16323,  16325,
      16327,  16329,  16330,  16332,  16333,  16335,  16336,  16338,  16339,
      16340,  16342,  16343,  16344,  16345,  16346,  16347,  16348,  16350,
      16351,  16351,  16352,  16353,  16354,  16355,  16356,  16357,  16358,
      16358,  16359,  16360,  16360,  16361,  16362,  16362,  16363,  16364,
      16364,  16365,  16365,  16366,  16366,  16367,  16367,  16368,  16368};

  {
    Tensor0D<int8_t> x{0};
    Tensor0D<int8_t> expected_result{0};
    Tensor0D<int8_t> result = tosa::table(x, table_linear8);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor0D<int16_t> x{0};
    Tensor0D<int32_t> expected_result{0};
    Tensor0D<int32_t> result = tosa::table(x, table_linear16);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor1D<int8_t, 2> x{-128, 127};
    Tensor1D<int8_t, 2> expected_result{-128, 127};
    Tensor1D<int8_t, 2> result = tosa::table(x, table_linear8);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    //                     -256    255
    Tensor1D<int16_t, 2> x{-32768, 32640};
    Tensor1D<int32_t, 2> expected_result{-32768, 32640};
    Tensor1D<int32_t, 2> result = tosa::table(x, table_linear16);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor2D<int8_t, 2, 2> x{-128, -2, 1, 127};
    Tensor2D<int8_t, 2, 2> expected_result{-127, -6, 3, 127};
    Tensor2D<int8_t, 2, 2> result = tosa::table(x, table_logistic8);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    //                        -256    -2    1    255
    Tensor2D<int16_t, 2, 2> x{-32768, -256, 128, 32640};
    //                                      -16368    -491    245    16368
    Tensor2D<int32_t, 2, 2> expected_result{-2095104, -62848, 31360, 2095104};
    Tensor2D<int32_t, 2, 2> result = tosa::table(x, table_logistic16);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor3D<int8_t, 2, 1, 2> x{-85, -65, 7, 42};
    Tensor3D<int8_t, 2, 1, 2> expected_result{-85, -65, 7, 42};
    Tensor3D<int8_t, 2, 1, 2> result = tosa::table(x, table_linear8);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    //                           -85.5  -65.125 7.875 42.0078125
    Tensor3D<int16_t, 2, 1, 2> x{-10944, -8336, 1008, 5377};
    Tensor3D<int32_t, 2, 1, 2> expected_result{-10944, -8336, 1008, 5377};
    Tensor3D<int32_t, 2, 1, 2> result = tosa::table(x, table_linear16);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    Tensor4D<int8_t, 1, 1, 2, 2> x{106, 98, -94, -23};
    Tensor4D<int8_t, 1, 1, 2, 2> expected_result{126, 126, -125, -66};
    Tensor4D<int8_t, 1, 1, 2, 2> result = tosa::table(x, table_logistic8);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    //                             -253.5  255.25 -152.125 49.625
    Tensor4D<int16_t, 1, 1, 2, 2> x{-32448, 32672, -19472, 6352};
    //                                          -16367 16368 -16045.25 10350.5
    Tensor4D<int32_t, 1, 1, 2, 2> expected_result{-2094976, 2095104, -2053792,
                                                  1324864};
    Tensor4D<int32_t, 1, 1, 2, 2> result = tosa::table(x, table_logistic16);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
}

TEST(tosa, broadcastable_op) {
  // In the CallOpBroadcastableConversion ops where the tensor shape of the
  // operands don't match, a broadcast_in_dim op is inserted. This unittest
  // verifies that procedure.

  // %0 = "tosa.add"(%arg0, %arg1) : (tensor<2x1x3xf32>, tensor<1x1x3xf32>) ->
  // tensor<2x1x3xf32>
  Tensor<float, 2, 1, 3> t0_arg0{3, 3, 3, 3, 3, 3};
  Tensor<float, 1, 1, 3> t0_arg1{1, 2, 3};
  Tensor<float, 2, 1, 3> t0_arg1_broadcasted =
      emitc::broadcast_in_dim<Tensor<float, 2, 1, 3>>(t0_arg1, {0, 1, 2});
  EXPECT_THAT(t0_arg1_broadcasted, Pointwise(Eq(), {1, 2, 3, 1, 2, 3}));
  tosa::add(t0_arg0,
            t0_arg1_broadcasted); // Just make sure it compiles in this test

  // %0 = "tosa.add"(%arg0, %arg1) : (tensor<2x1x3xf32>, tensor<3xf32>) ->
  // tensor<2x1x3xf32>
  Tensor<float, 2, 1, 3> t1_arg0{4, 4, 4, 4, 4, 4};
  Tensor<float, 3> t1_arg1{1, 2, 3};
  Tensor<float, 2, 1, 3> t1_arg1_broadcasted =
      emitc::broadcast_in_dim<Tensor<float, 2, 1, 3>>(t1_arg1, {2});
  EXPECT_THAT(t1_arg1_broadcasted, Pointwise(Eq(), {1, 2, 3, 1, 2, 3}));
  tosa::add(t1_arg0,
            t1_arg1_broadcasted); // Just make sure it compiles in this test
}

// Other ops
TEST(tosa, depthwise_conv2d) {
  {
    // test for channel_multiplier=1
    using InputType = Tensor4D<float, 1, 4, 5, 2>;  // N H W C
    using WeightType = Tensor4D<float, 2, 2, 2, 1>; // KH KW CIN M
    using ResultType = Tensor4D<float, 1, 3, 4, 2>; // N H W CXM
    InputType input{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
    WeightType weights{1, 2, 3, 4, 5, 6, 7, 8};
    ResultType expected_result{156, 204, 188, 244, 220, 284, 252, 324,
                               316, 404, 348, 444, 380, 484, 412, 524,
                               476, 604, 508, 644, 540, 684, 572, 724};

    Tensor1D<int64_t, 4> padding{0, 0, 0, 0}; // {pt, pb, pl, pr}
    Tensor1D<int64_t, 2> dilation{1, 1};
    Tensor1D<int64_t, 2> stride{1, 1};

    ResultType result = tosa::depthwise_conv2d<ResultType>(
        input, weights, padding, stride, dilation);
    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    // test for channel_multiplier=2
    using InputType = Tensor4D<float, 1, 4, 5, 2>;  // N H W C
    using WeightType = Tensor4D<float, 2, 2, 2, 2>; // KH KW CIN M
    using ResultType = Tensor4D<float, 1, 3, 4, 4>; // N H W CXM
    InputType input{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40};
    WeightType weights{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    ResultType expected_result{
        284, 312,  376,  408,  340, 376,  448,  488,  396,  440,  520,  568,
        452, 504,  592,  648,  564, 632,  736,  808,  620,  696,  808,  888,
        676, 760,  880,  968,  732, 824,  952,  1048, 844,  952,  1096, 1208,
        900, 1016, 1168, 1288, 956, 1080, 1240, 1368, 1012, 1144, 1312, 1448};

    Tensor1D<int64_t, 4> padding{0, 0, 0, 0}; // {pt, pb, pl, pr}
    Tensor1D<int64_t, 2> dilation{1, 1};
    Tensor1D<int64_t, 2> stride{1, 1};

    ResultType result = tosa::depthwise_conv2d<ResultType>(
        input, weights, padding, stride, dilation);
    EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
  }
}

TEST(tosa, fully_connected) {
  using InputType = Tensor2D<float, 2, 5>;  // N CIN
  using WeightType = Tensor2D<float, 2, 5>; // COUT CIN
  using BiasType = Tensor1D<float, 2>;      // COUT
  using ResultType = Tensor2D<float, 2, 2>; // N COUT
  InputType input{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  WeightType weights{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  BiasType bias{100, 200};
  ResultType expected_result{155, 330, 230, 530};
  ResultType result = tosa::fully_connected<ResultType>(input, weights, bias);

  EXPECT_THAT(result, Pointwise(FloatNear(EPSILON), expected_result));
}

TEST(tosa, matmul) {
  {
    using AType = Tensor3D<float, 1, 3, 1>; // M K
    using BType = Tensor3D<float, 1, 1, 2>; // K N
    using CType = Tensor3D<float, 1, 3, 2>; // M N
    AType a{1, 2, 3};
    BType b{1, 2};
    CType c = tosa::matmul(a, b);

    CType expected_result{1, 2, 2, 4, 3, 6};
    EXPECT_THAT(c, Pointwise(FloatNear(EPSILON), expected_result));
  }
  {
    using AType = Tensor3D<float, 1, 3, 2>; // M K
    using BType = Tensor3D<float, 1, 2, 2>; // K N
    using CType = Tensor3D<float, 1, 3, 2>; // M N
    AType a{1, 2, 3, 4, 5, 6};
    BType b{7, 8, 9, 10};
    CType c = tosa::matmul(a, b);

    CType expected_result{25, 28, 57, 64, 89, 100};
    EXPECT_THAT(c, Pointwise(FloatNear(EPSILON), expected_result));
  }
}

TEST(tosa, reduce_all) {
  Tensor<bool, 2, 3> t0{true, true, true, false, true, false};

  Tensor<bool, 3> expected_result0_0{false, true, false};
  Tensor<bool, 2> expected_result0_1{true, false};

  Tensor<bool, 3> result0_0 = tosa::reduce_all<Tensor<bool, 3>>(t0, 0);
  Tensor<bool, 2> result0_1 = tosa::reduce_all<Tensor<bool, 2>>(t0, 1);

  EXPECT_THAT(result0_0, Pointwise(Eq(), expected_result0_0));
  EXPECT_THAT(result0_1, Pointwise(Eq(), expected_result0_1));
}

TEST(tosa, reduce_any) {
  Tensor<bool, 2, 3> t0{true, true, false, true, false, false};
  Tensor<bool, 3> t1{false, false, false};

  Tensor<bool, 3> expected_result0_0{true, true, false};
  Tensor<bool, 2> expected_result0_1{true, true};
  Tensor<bool> expected_result1{false};

  Tensor<bool, 3> result0_0 = tosa::reduce_any<Tensor<bool, 3>>(t0, 0);
  Tensor<bool, 2> result0_1 = tosa::reduce_any<Tensor<bool, 2>>(t0, 1);
  Tensor<bool> result1 = tosa::reduce_any<Tensor<bool>>(t1, 0);

  EXPECT_THAT(result0_0, Pointwise(Eq(), expected_result0_0));
  EXPECT_THAT(result0_1, Pointwise(Eq(), expected_result0_1));
  EXPECT_THAT(result1, Pointwise(Eq(), expected_result1));
}

TEST(tosa, reduce_max) {
  Tensor<int32_t, 2, 3> t0{1, 2, 3, 4, 5, 6};

  Tensor<int32_t, 3> expected_result0_0{4, 5, 6};
  Tensor<int32_t, 2> expected_result0_1{3, 6};

  Tensor<int32_t, 3> result0_0 = tosa::reduce_max<Tensor<int32_t, 3>>(t0, 0);
  Tensor<int32_t, 2> result0_1 = tosa::reduce_max<Tensor<int32_t, 2>>(t0, 1);

  EXPECT_THAT(result0_0, Pointwise(Eq(), expected_result0_0));
  EXPECT_THAT(result0_1, Pointwise(Eq(), expected_result0_1));
}

TEST(tosa, reduce_min) {
  Tensor<int32_t, 2, 3> t0{1, 2, 3, 4, 5, 6};

  Tensor<int32_t, 3> expected_result0_0{1, 2, 3};
  Tensor<int32_t, 2> expected_result0_1{1, 4};

  Tensor<int32_t, 3> result0_0 = tosa::reduce_min<Tensor<int32_t, 3>>(t0, 0);
  Tensor<int32_t, 2> result0_1 = tosa::reduce_min<Tensor<int32_t, 2>>(t0, 1);

  EXPECT_THAT(result0_0, Pointwise(Eq(), expected_result0_0));
  EXPECT_THAT(result0_1, Pointwise(Eq(), expected_result0_1));
}

TEST(tosa, reduce_prod) {
  Tensor<int32_t, 2, 3> t0{1, 2, 3, 4, 5, 6};

  Tensor<int32_t, 3> expected_result0_0{4, 10, 18};
  Tensor<int32_t, 2> expected_result0_1{6, 120};

  Tensor<int32_t, 3> result0_0 = tosa::reduce_prod<Tensor<int32_t, 3>>(t0, 0);
  Tensor<int32_t, 2> result0_1 = tosa::reduce_prod<Tensor<int32_t, 2>>(t0, 1);

  EXPECT_THAT(result0_0, Pointwise(Eq(), expected_result0_0));
  EXPECT_THAT(result0_1, Pointwise(Eq(), expected_result0_1));
}

TEST(tosa, reduce_sum) {
  Tensor<int32_t, 2, 3> t0{1, 2, 3, 4, 5, 6};
  Tensor<int32_t, 4, 2, 3> t1{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
                              1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
  Tensor<int32_t, 3> expected_result0_0{5, 7, 9};
  Tensor<int32_t, 2> expected_result0_1{6, 15};
  Tensor<int32_t, 2, 3> expected_result1_0{4, 8, 12, 16, 20, 24};
  Tensor<int32_t, 4, 3> expected_result1_1{5, 7, 9, 5, 7, 9, 5, 7, 9, 5, 7, 9};
  Tensor<int32_t, 4, 2> expected_result1_2{6, 15, 6, 15, 6, 15, 6, 15};

  Tensor<int32_t, 3> result0_0 = tosa::reduce_sum<Tensor<int32_t, 3>>(t0, 0);
  Tensor<int32_t, 2> result0_1 = tosa::reduce_sum<Tensor<int32_t, 2>>(t0, 1);
  Tensor<int32_t, 2, 3> result1_0 =
      tosa::reduce_sum<Tensor<int32_t, 2, 3>>(t1, 0);
  Tensor<int32_t, 4, 3> result1_1 =
      tosa::reduce_sum<Tensor<int32_t, 4, 3>>(t1, 1);
  Tensor<int32_t, 4, 2> result1_2 =
      tosa::reduce_sum<Tensor<int32_t, 4, 2>>(t1, 2);

  EXPECT_THAT(result0_0, Pointwise(Eq(), expected_result0_0));
  EXPECT_THAT(result0_1, Pointwise(Eq(), expected_result0_1));
  EXPECT_THAT(result1_0, Pointwise(Eq(), expected_result1_0));
  EXPECT_THAT(result1_1, Pointwise(Eq(), expected_result1_1));
  EXPECT_THAT(result1_2, Pointwise(Eq(), expected_result1_2));
}

TEST(tosa, reshape) {
  Tensor2D<int, 1, 2> t0 = {1, 2};
  Tensor3D<int, 1, 1, 2> s0 = tosa::reshape<Tensor3D<int, 1, 1, 2>>(t0);
  EXPECT_THAT(s0, Pointwise(Eq(), t0));

  Tensor3D<int, 2, 1, 2> t1 = {1, 2, 3, 4};
  Tensor3D<int, 1, 2, 2> s1 = tosa::reshape<Tensor3D<int, 1, 2, 2>>(t1);
  EXPECT_THAT(s1, Pointwise(Eq(), t1));

  Tensor1D<int, 10> t2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  Tensor2D<int, 2, 5> s2 = tosa::reshape<Tensor2D<int, 2, 5>>(t2);
  EXPECT_THAT(s2, Pointwise(Eq(), t2));

  Tensor3D<int, 2, 2, 2> t3 = {1, 2, 3, 4, 5, 6, 7, 8};
  Tensor1D<int, 8> s3 = tosa::reshape<Tensor1D<int, 8>>(t3);
  EXPECT_THAT(s3, Pointwise(Eq(), t3));
}

TEST(tosa, slice) {
  // Slice Tensor1D
  Tensor1D<float, 5> s1{0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  auto t1 = tosa::slice<Tensor1D<float, 2>>(s1, {2}, {2});
  EXPECT_THAT(t1, Pointwise(FloatEq(), {2.0f, 3.0f}));

  // Slice Tensor2D
  Tensor2D<float, 4, 3> s2{0.0f, 1.0f, 2.0f, 3.0f, 4.0f,  5.0f,
                           6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  auto t2 = tosa::slice<Tensor2D<float, 2, 2>>(s2, {2, 1}, {2, 2});

  EXPECT_THAT(t2, Pointwise(FloatEq(), {7.0f, 8.0f, 10.0f, 11.0f}));

  // Slice Tensor3D
  Tensor3D<float, 4, 3, 2> s3{0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
                              6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
                              12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
                              18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f};
  auto t3 = tosa::slice<Tensor3D<float, 2, 2, 2>>(s3, {2, 1, 0}, {2, 2, 2});
  EXPECT_THAT(t3, Pointwise(FloatEq(), {14.0f, 15.0f, 16.0f, 17.0f, 20.0f,
                                        21.0f, 22.0f, 23.0f}));

  // Slice Tensor4D
  Tensor4D<float, 4, 3, 1, 2> s4{0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
                                 6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
                                 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
                                 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f};
  auto t4 =
      tosa::slice<Tensor4D<float, 2, 2, 1, 2>>(s4, {2, 1, 0, 0}, {2, 2, 1, 2});
  EXPECT_THAT(t4, Pointwise(FloatEq(), {14.0f, 15.0f, 16.0f, 17.0f, 20.0f,
                                        21.0f, 22.0f, 23.0f}));

  auto t4_2 =
      tosa::slice<Tensor4D<float, 4, 3, 1, 2>>(s4, {0, 0, 0, 0}, {4, 3, 1, 2});
  EXPECT_THAT(t4_2, Pointwise(FloatEq(), s4));
}

TEST(tosa, pad) {
  // clang-format off
  Tensor2D<int32_t, 2, 3> operand0{1, 2, 3,
                                   4, 5, 6};
  Tensor3D<int32_t, 2, 2, 3> operand1{1, 2, 3,  4,  5,  6,
                                      7, 8, 9, 10, 11, 12};
  // clang-format on

  {
    // clang-format off
    Tensor2D<int32_t, 2, 2> padding{0, 1,
                                    1, 2};
    Tensor2D<int32_t, 3, 6> expected_result{0, 1, 2, 3, 0, 0,
                                            0, 4, 5, 6, 0, 0,
                                            0, 0, 0, 0, 0, 0};
    // clang-format on
    Tensor2D<int32_t, 3, 6> result =
        tosa::pad<Tensor2D<int32_t, 3, 6>>(operand0, padding);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  { // explicit value
    // clang-format off
    Tensor2D<int32_t, 2, 2> padding{0, 1,
                                    1, 2};
    Tensor0D<int32_t> pad_const{1};
    Tensor2D<int32_t, 3, 6> expected_result{1, 1, 2, 3, 1, 1,
                                            1, 4, 5, 6, 1, 1,
                                            1, 1, 1, 1, 1, 1};
    // clang-format on
    Tensor2D<int32_t, 3, 6> result =
        tosa::pad<Tensor2D<int32_t, 3, 6>>(operand0, padding, pad_const);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    // clang-format off
    Tensor2D<int32_t, 3, 2> padding{0, 0,
                                    0, 0,
                                    0, 0};
    Tensor3D<int32_t, 2, 2, 3> expected_result{1, 2, 3,  4,  5,  6,
                                               7, 8, 9, 10, 11, 12};
    // clang-format on
    Tensor3D<int32_t, 2, 2, 3> result =
        tosa::pad<Tensor3D<int32_t, 2, 2, 3>>(operand1, padding);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  { // explicit value
    // clang-format off
    Tensor2D<int32_t, 3, 2> padding{0, 0,
                                    0, 0,
                                    0, 0};
    Tensor0D<int32_t> pad_const{1};
    Tensor3D<int32_t, 2, 2, 3> expected_result{1, 2, 3,  4,  5,  6,
                                               7, 8, 9, 10, 11, 12};
    // clang-format on
    Tensor3D<int32_t, 2, 2, 3> result =
        tosa::pad<Tensor3D<int32_t, 2, 2, 3>>(operand1, padding, pad_const);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    // clang-format off
    Tensor2D<int32_t, 3, 2> padding{1, 1,
                                    1, 1,
                                    1, 1};
    Tensor3D<int32_t, 4, 4, 5> expected_result{
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0,  4,  5,  6,  0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 7, 8, 9, 0, 0, 10, 11, 12,  0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0, 0, 0, 0, 0, 0};
    // clang-format on
    Tensor3D<int32_t, 4, 4, 5> result =
        tosa::pad<Tensor3D<int32_t, 4, 4, 5>>(operand1, padding);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  { // explicit value
    // clang-format off
    Tensor2D<int32_t, 3, 2> padding{1, 1,
                                    1, 1,
                                    1, 1};
    Tensor0D<int32_t> pad_const{-1};
    Tensor3D<int32_t, 4, 4, 5> expected_result{
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1,  -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1,  1,  2,  3, -1, -1,   4,   5,   6,  -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1,  7,  8,  9, -1, -1,  10,  11,  12,  -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1,  -1, -1, -1, -1, -1, -1};
    // clang-format on
    Tensor3D<int32_t, 4, 4, 5> result =
        tosa::pad<Tensor3D<int32_t, 4, 4, 5>>(operand1, padding, pad_const);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  {
    // clang-format off
    Tensor2D<int32_t, 3, 2> padding{1, 0,
                                    0, 1,
                                    1, 0};
    Tensor3D<int32_t, 3, 3, 4> expected_result{
        0, 0, 0, 0, 0,  0,  0,  0, 0, 0, 0, 0,
        0, 1, 2, 3, 0,  4,  5,  6, 0, 0, 0, 0,
        0, 7, 8, 9, 0, 10, 11, 12, 0, 0, 0, 0};
    // clang-format on
    Tensor3D<int32_t, 3, 3, 4> result =
        tosa::pad<Tensor3D<int32_t, 3, 3, 4>>(operand1, padding);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
  { // explicit value
    // clang-format off
    Tensor2D<int32_t, 3, 2> padding{1, 0,
                                    0, 1,
                                    1, 0};
    Tensor0D<int32_t> pad_const{3};
    Tensor3D<int32_t, 3, 3, 4> expected_result{
        3, 3, 3, 3, 3,  3,  3,  3, 3, 3, 3, 3,
        3, 1, 2, 3, 3,  4,  5,  6, 3, 3, 3, 3,
        3, 7, 8, 9, 3, 10, 11, 12, 3, 3, 3, 3};
    // clang-format on
    Tensor3D<int32_t, 3, 3, 4> result =
        tosa::pad<Tensor3D<int32_t, 3, 3, 4>>(operand1, padding, pad_const);

    EXPECT_THAT(result, Pointwise(Eq(), expected_result));
  }
}

TEST(tosa, transpose) {
  // clang-format off
  Tensor2D<float, 3, 2> t0 = {1, 2,
                              3, 4,
                              5, 6};
  Tensor1D<int32_t, 2> perms_i32 = {1, 0};
  Tensor1D<int64_t, 2> perms_i64 = {1, 0};
  Tensor1D<int64_t, 2> no_perms =  {0, 1};
  Tensor2D<float, 2, 3> expected_result0 = {1, 3, 5,
                                            2, 4, 6};
  // clang-format on
  Tensor2D<float, 2, 3> s0 =
      tosa::transpose<Tensor2D<float, 2, 3>>(t0, perms_i32);
  Tensor2D<float, 2, 3> s0_2 =
      tosa::transpose<Tensor2D<float, 2, 3>>(t0, perms_i64);
  Tensor2D<float, 3, 2> s0_3 =
      tosa::transpose<Tensor2D<float, 3, 2>>(t0, no_perms);
  EXPECT_THAT(s0, Pointwise(Eq(), expected_result0));
  EXPECT_THAT(s0_2, Pointwise(Eq(), expected_result0));
  EXPECT_THAT(s0_3, Pointwise(Eq(), t0));

  // clang-format off
  Tensor3D<float, 1, 3, 2> t1 = {1, 2,
                                 3, 4,
                                 5, 6};
  Tensor1D<int32_t, 3> perms1 = {2, 0, 1};
  Tensor3D<float, 2, 1, 3> expected_result1 = {1, 3, 5,
                                               2, 4, 6};
  // clang-format on
  Tensor3D<float, 2, 1, 3> s1 =
      tosa::transpose<Tensor3D<float, 2, 1, 3>>(t1, perms1);
  EXPECT_THAT(s1, Pointwise(Eq(), expected_result1));

  // clang-format off
  Tensor3D<float, 2, 3, 4> t2 = {1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
                                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  Tensor1D<int32_t, 3> perms2 = {2, 0, 1};
  Tensor3D<float, 4, 2, 3> expected_result2 = {1, 5,  9, 13, 17, 21,
                                               2, 6, 10, 14, 18, 22,
                                               3, 7, 11, 15, 19, 23,
                                               4, 8, 12, 16, 20, 24};
  // clang-format on
  Tensor3D<float, 4, 2, 3> s2 =
      tosa::transpose<Tensor3D<float, 4, 2, 3>>(t2, perms2);
  EXPECT_THAT(s2, Pointwise(Eq(), expected_result2));
}

} // namespace
