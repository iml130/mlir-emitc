<!--
SPDX-FileCopyrightText: Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->
# Func Op Coverage

The table below shows the func ops, supported with the `--convert-func-to-emitc` **upstream** conversion.

| op                    | supported          | comment |
| :-------------------- |:------------------:| :------ |
| call                  | :heavy_check_mark: | |
| func                  | :heavy_check_mark: | |
| return                | :heavy_check_mark: | |

The table below shows the func ops, supported **upstream** via `--mlir-to-cpp`

| op                    | supported          | comment |
| :-------------------- |:------------------:| :------ |
| call                  | :white_check_mark: | via `emitc-translate` |
| func                  | :white_check_mark: | via `emitc-translate` |
| return                | :white_check_mark: | via `emitc-translate` |
