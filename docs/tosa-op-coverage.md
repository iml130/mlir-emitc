# TOSA Op Coverage

The table below shows the supported TOSA ops.

| op                    | supported          | comment |
| :-------------------- |:------------------:| :------ |
| **Data node ops**
| const                 | :heavy_check_mark: | |
| **Unary elementwise ops**
| abs                   | :heavy_check_mark: | |
| ceil                  | :heavy_check_mark: | |
| exp                   | :heavy_check_mark: | |
| floor                 | :heavy_check_mark: | |
| log                   | :heavy_check_mark: | |
| reciprocal            | :heavy_check_mark: | |
| rsqrt                 | :heavy_check_mark: | |
| tanh                  | :heavy_check_mark: | |
| **Binary elementwise ops**
| add                   | :heavy_check_mark: | |
| mul                   | :heavy_check_mark: | |
| sub                   | :heavy_check_mark: | |
| **Other ops**
| conv2d                | :white_check_mark: | Quantization not supported |
| fully_connected       | :white_check_mark: | Quantization not supported |
| matmul                | :white_check_mark: | Quantization not supported |
