# MHLO HLO Op Coverage

The table below shows the supported MHLO ops.

| op                    | supported          | comment |
| :-------------------- |:------------------:| :------ |
| **Unary elementwise ops**
| abs                   | :heavy_check_mark: | |
| ceil                  | :heavy_check_mark: | |
| convert               | :heavy_check_mark: | |
| cosine                | :heavy_check_mark: | |
| exponential           | :heavy_check_mark: | |
| exponential_minus_one | :white_check_mark: | `--preprocess-mhlo-for-emitc`|
| floor                 | :heavy_check_mark: | |
| is_finite             | :heavy_check_mark: | |
| log                   | :heavy_check_mark: | |
| log_plus_one          | :white_check_mark: | `--preprocess-mhlo-for-emitc`|
| negate                | :heavy_check_mark: | |
| round_nearest_afz     | :heavy_check_mark: | |
| sine                  | :heavy_check_mark: | |
| sqrt                  | :heavy_check_mark: | |
| tanh                  | :heavy_check_mark: | |
| **Binary elementwise ops**
| add                   | :heavy_check_mark: | |
| atan2                 | :heavy_check_mark: | |
| divide                | :heavy_check_mark: | |
| maximum               | :heavy_check_mark: | |
| minimum               | :heavy_check_mark: | |
| multiply              | :heavy_check_mark: | |
| power                 | :heavy_check_mark: | |
| shift_left            | :heavy_check_mark: | |
| shift_right_logical   | :heavy_check_mark: | |
| subtract              | :heavy_check_mark: | |
| **Control flow ops**
| if                    | :white_check_mark: | `--mhlo-control-flow-to-scf`, `--convert-scf-to-emitc`|
| while                 | :white_check_mark: | `--mhlo-control-flow-to-scf`, `--convert-scf-to-emitc`|
| **Binary logical elementwise ops**
| or                    | :heavy_check_mark: | |
| xor                   | :heavy_check_mark: | |
| **Tuple ops**
| get_tuple_element     | :heavy_check_mark: | |
| tuple                 | :heavy_check_mark: | |
| compare               | :heavy_check_mark: | |
| **Slice ops**
| slice                 | :white_check_mark: | Only for 1D or 2D inputs |
| dynamic-slice         | :white_check_mark: | Only for 1D or 2D inputs |
| dynamic-update-slice  | :white_check_mark: | Only for 1D or 2D inputs |
| **Other ops**
| batch_norm_inference  | :heavy_check_mark: | |
| bitcast_convert       | :heavy_check_mark: | |
| broadcast_in_dim      | :white_check_mark: | Only for 1D inputs |
| clamp                 | :heavy_check_mark: | |
| concatenate           | :heavy_check_mark: | |
| convolution           | :white_check_mark: | No support for dilated convolutions, only for `NHWC` layout. Untested for grouped convolutions except for the depthwise case |
| dot                   | :white_check_mark: | Only the `Matrix times Matrix` case |
| pad                   | :white_check_mark: | No support for negative edge padding |
| reshape               | :heavy_check_mark: | |
| select                | :heavy_check_mark: | |
