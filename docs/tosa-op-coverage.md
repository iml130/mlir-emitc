# TOSA Op Coverage

The table below shows the supported TOSA ops.

| op                     | supported          | comment |
| :--------------------- |:------------------:| :------ |
| **Data node ops**
| const                  | :heavy_check_mark: | |
| **Unary elementwise ops**
| abs                    | :heavy_check_mark: | |
| cast                   | :heavy_check_mark: | |
| ceil                   | :heavy_check_mark: | |
| clamp                  | :heavy_check_mark: | |
| clz                    | :heavy_check_mark: | |
| exp                    | :heavy_check_mark: | |
| floor                  | :heavy_check_mark: | |
| log                    | :heavy_check_mark: | |
| negate                 | :heavy_check_mark: | |
| reciprocal             | :heavy_check_mark: | |
| rescale                | :heavy_check_mark: | |
| rsqrt                  | :heavy_check_mark: | |
| tanh                   | :heavy_check_mark: | |
| **Binary elementwise ops**
| add                    | :heavy_check_mark: | |
| arithmetic_right_shift | :heavy_check_mark: | |
| equal                  | :heavy_check_mark: | |
| logical_left_shift     | :heavy_check_mark: | |
| maximum                | :heavy_check_mark: | |
| minimum                | :heavy_check_mark: | |
| mul                    | :heavy_check_mark: | |
| pow                    | :heavy_check_mark: | |
| sub                    | :heavy_check_mark: | |
| table                  | :heavy_check_mark: | |
| **Ternary elementwise ops**
| select                 | :heavy_check_mark: | |
| **Other ops**
| argmax                 | :heavy_check_mark: | |
| concat                 | :heavy_check_mark: | |
| conv2d                 | :white_check_mark: | Quantization and dilation not supported |
| depthwise_conv2d       | :white_check_mark: | Quantization and dilation not supported |
| gather                 | :heavy_check_mark: | |
| fully_connected        | :white_check_mark: | Quantization not supported |
| matmul                 | :white_check_mark: | Quantization not supported |
| reduce_all             | :heavy_check_mark: | |
| reduce_any             | :heavy_check_mark: | |
| reduce_max             | :heavy_check_mark: | |
| reduce_min             | :heavy_check_mark: | |
| reduce_prod            | :heavy_check_mark: | |
| reduce_sum             | :heavy_check_mark: | |
| reshape                | :heavy_check_mark: | |
| slice                  | :white_check_mark: | Only for 1D to 4D inputs |
| pad                    | :white_check_mark: | Quantization not supported |
| tile                   | :heavy_check_mark: | |
| transpose              | :heavy_check_mark: | |
