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
// #include "cublas_v2.h"
// #cgo LDFLAGS: -L/usr/lib/x86_64-linux-gnu/ -lcublas
import "C"
import (
	"github.com/kuroko1t/gocuda"
	"unsafe"
)

func Create() (Handle_t, Status_t) {
	var handle C.cublasHandle_t
	status := C.cublasCreate(&handle)
	return Handle_t(handle), Status_t(status)
}

func Destroy(handle Handle_t) Status_t {
	status := C.cublasDestroy(C.cublasHandle_t(handle))
	return Status_t(status)
}

func GetErrorString(status Status_t) string {
	switch status {
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS"
	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED"
	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED"
	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE"
	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH"
	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR"
	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED"
	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR"
	case CUBLAS_STATUS_NOT_SUPPORTED:
		return "CUBLAS_STATUS_NOT_SUPPORTED"
	case CUBLAS_STATUS_LICENSE_ERROR:
		return "CUBLAS_STATUS_LICENSE_ERROR"
	default:
		return "NOT_MATCH_ERROR_CODE"
	}
}

func Sgemm(handle Handle_t, transa, transb Operation_t, m, n, k int,
	alpha *float32, A cu.CUdeviceptr, lda int, B cu.CUdeviceptr, ldb int, beta *float32, C cu.CUdeviceptr, ldc int) Status_t {
	calpha := C.float(*alpha)
	cbeta := C.float(*beta)
	cA := (*C.float)(unsafe.Pointer(uintptr(A)))
	cB := (*C.float)(unsafe.Pointer(uintptr(B)))
	cC := (*C.float)(unsafe.Pointer(uintptr(C)))
	status := C.cublasSgemm(C.cublasHandle_t(handle), C.cublasOperation_t(transa),
		C.cublasOperation_t(transb), C.int(m), C.int(n), C.int(k), &calpha, cA, C.int(lda),
		cB, C.int(ldb), &cbeta, cC, C.int(ldc))
	return Status_t(status)
}

func Sgeam(handle Handle_t, transa, transb Operation_t, m, n int, alpha *float32, A cu.CUdeviceptr, lda int, beta *float32, B cu.CUdeviceptr, ldb int, C cu.CUdeviceptr, ldc int) Status_t {
	calpha := C.float(*alpha)
	cbeta := C.float(*beta)

	cA := (*C.float)(unsafe.Pointer(uintptr(A)))
	cB := (*C.float)(unsafe.Pointer(uintptr(B)))
	cC := (*C.float)(unsafe.Pointer(uintptr(C)))

	status := C.cublasSgeam(C.cublasHandle_t(handle), C.cublasOperation_t(transa), C.cublasOperation_t(transb), C.int(m), C.int(n), &calpha, cA, C.int(lda), &cbeta, cB, C.int(ldb), cC, C.int(ldc))
	return Status_t(status)
}

func Saxpy(handle Handle_t, n int, alpha *float32, x cu.CUdeviceptr, incx int, y cu.CUdeviceptr, incy int) Status_t {
	calpha := C.float(*alpha)

	cx := (*C.float)(unsafe.Pointer(uintptr(x)))
	cy := (*C.float)(unsafe.Pointer(uintptr(y)))

	status := C.cublasSaxpy(C.cublasHandle_t(handle), C.int(n), &calpha, cx, C.int(incx), cy, C.int(incy))
	return Status_t(status)
}

func Sasum(handle Handle_t, n int, x cu.CUdeviceptr, incx int, result *float32) Status_t {
	cx := (*C.float)(unsafe.Pointer(uintptr(x)))
	status := C.cublasSasum(C.cublasHandle_t(handle), C.int(n), cx, C.int(incx), (*C.float)(result))
	return Status_t(status)
}
