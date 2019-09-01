package main

import (
	"fmt"
	"github.com/kuroko1t/gocuda"
	"github.com/kuroko1t/gocublas"
	"unsafe"
)

func cuErrCheck(status cu.CUresult) {
	if status != cu.CUDA_SUCCESS {
		fmt.Println(cu.GetErrorString(status))
	}
}

func cublasErrCheck(status cublas.Status_t) {
	if status != cublas.CUBLAS_STATUS_SUCCESS {
		fmt.Println(cublas.GetErrorString(status))
	}
}

func main() {
	// Init Device
	cuErrCheck(cu.Setup(0))
	handle, cublaserr := cublas.Create()
	cublasErrCheck(cublaserr)
	//// input data
	A := [][]float32{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	B := [][]float32{
		{8, 7},
		{5, 4},
		{2, 1},
	}

	C := [][]float32{
		{0, 0},
		{0, 0},
		{0, 0},
	}
	// A = (3, 3)
	// B = (3, 2)
	// C = (3, 2)
	//// Malloc DevicePtr
	devA, err := cu.MemAlloc(uint32(unsafe.Sizeof(A[0][0]))*uint32(9))
	cuErrCheck(err)
	devB, err := cu.MemAlloc(uint32(unsafe.Sizeof(B[0][0]))*uint32(6))
	cuErrCheck(err)
	devC, err := cu.MemAlloc(uint32(unsafe.Sizeof(C[0][0]))*uint32(6))
	cuErrCheck(err)
	//
	//// transfoer H to D
	cuErrCheck(cu.MemcpyHtoD(devA, &A[0][0], uint32(unsafe.Sizeof(A[0][0]))*uint32(9)))
	cuErrCheck(cu.MemcpyHtoD(devB, &B[0][0], uint32(unsafe.Sizeof(B[0][0]))*uint32(6)))
	cuErrCheck(cu.MemcpyHtoD(devC, &C[0][0], uint32(unsafe.Sizeof(C[0][0]))*uint32(6)))

	var alpha float32 = 1.0
	var beta float32 = 1.0
	m := len(A)
	n := len(B[0])
	k := len(A[0])
	lda := m
	ldb := k
	ldc := m
	cublaserr = cublas.Sgemm(handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N, m, n, k,
		&alpha, devA, lda, devB, ldb, &beta, devC, ldc)
	cublasErrCheck(cublaserr)
	//// transfoer D to H
	cuErrCheck(cu.MemcpyDtoH(&C[0][0], devC, uint32(unsafe.Sizeof(C[0][0]))*uint32(6)))

	fmt.Println(B, "dot", A, "=", C)
	cuErrCheck(cu.MemFree(devA))
	cuErrCheck(cu.MemFree(devB))
	cuErrCheck(cu.MemFree(devC))
	cublaserr = cublas.Destroy(handle)
	cublasErrCheck(cublaserr)
	cuErrCheck(cu.Teardown(0))
}
