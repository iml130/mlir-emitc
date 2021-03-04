# TOSA Op Coverage

The table below shows the supported TOSA ops.

| op                    | supported          | comment |
| :-------------------- |:------------------:| :------ |
| **Unary elementwise ops**
| abs                   | :heavy_check_mark: | |
| ceil                  | :heavy_check_mark: | |
| exp                   | :heavy_check_mark: | |
| floor                 | :heavy_check_mark: | |
| log                   | :heavy_check_mark: | |
| reciprocal            | :heavy_check_mark: | |
| rsqrt                 | :heavy_check_mark: | |
| tanh                 | :heavy_check_mark: | |
| **Binary elementwise ops**
| add                   | :heavy_check_mark: | |
| mul                   | :heavy_check_mark: | |
| **Other ops**
| fully_connected       | :white_check_mark: | Quantization not supported |
| reduce_all            | :white_check_mark: | |
| reduce_any            | :white_check_mark: | |
| reduce_max            | :white_check_mark: | |
| reduce_min            | :white_check_mark: | |
| reduce_prod           | :white_check_mark: | |
| reduce_sum            | :white_check_mark: | |
