// Copyright 2019 kurosawa. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

package cublas

// #cgo CFLAGS: -I/usr/local/cuda/include/
// #include "cublas.h"
import "C"

type Handle_t C.cublasHandle_t
type Status_t C.cublasStatus_t
type Operation_t C.cublasOperation_t

// cublasStatus_t
const (
	CUBLAS_STATUS_SUCCESS          = C.CUBLAS_STATUS_SUCCESS
	CUBLAS_STATUS_NOT_INITIALIZED  = C.CUBLAS_STATUS_NOT_INITIALIZED
	CUBLAS_STATUS_ALLOC_FAILED     = C.CUBLAS_STATUS_ALLOC_FAILED
	CUBLAS_STATUS_INVALID_VALUE    = C.CUBLAS_STATUS_INVALID_VALUE
	CUBLAS_STATUS_ARCH_MISMATCH    = C.CUBLAS_STATUS_ARCH_MISMATCH
	CUBLAS_STATUS_MAPPING_ERROR    = C.CUBLAS_STATUS_MAPPING_ERROR
	CUBLAS_STATUS_EXECUTION_FAILED = C.CUBLAS_STATUS_EXECUTION_FAILED
	CUBLAS_STATUS_INTERNAL_ERROR   = C.CUBLAS_STATUS_INTERNAL_ERROR
	CUBLAS_STATUS_NOT_SUPPORTED    = C.CUBLAS_STATUS_NOT_SUPPORTED
	CUBLAS_STATUS_LICENSE_ERROR    = C.CUBLAS_STATUS_LICENSE_ERROR
)

// cublasOperation_t
const (
	CUBLAS_OP_N        = C.CUBLAS_OP_N
	CUBLAS_OP_T        = C.CUBLAS_OP_T
	CUBLAS_OP_C        = C.CUBLAS_OP_C
	CUBLAS_OP_HERMITAN = C.CUBLAS_OP_HERMITAN
	CUBLAS_OP_CONJG    = C.CUBLAS_OP_CONJG
)
