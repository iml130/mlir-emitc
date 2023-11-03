<!--
SPDX-FileCopyrightText: Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->
# SCF Op Coverage

The table below shows the SCF ops, supported via `--convert-scf-to-emitc` **upstream** conversions.

| op                    | supported          | comment |
| :-------------------- |:------------------:| :------ |
| for                   | :heavy_check_mark: | |
| if                    | :heavy_check_mark: | |
| yield                 | :white_check_mark: | only as part of lowering `for` and `if` ops |
