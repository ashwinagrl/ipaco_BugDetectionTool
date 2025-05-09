%%writefile datarace.cu
#include <stdio.h>

#define N 2 // For simplicity, just two threads in the block

// Test Case 1: Intra Data Race
__global__ void test_case_1(int* Array) {
    Array[0] = threadIdx.x;  // Thread 0 writes 0, Thread 1 writes 1
    __syncthreads();         // Synchronization
    Array[0] = threadIdx.x;  // Thread 0 writes 0, Thread 1 writes 1
}

// Test Case 2: Both Inter and Intra Data Race (No Synchronization Between Threads)
__global__ void test_case_2(int* Array) {
    // Thread 0 and thread 1 both write to Array[0], no synchronization
    Array[0] = threadIdx.x;  // Thread 0 writes 0, Thread 1 writes 1
    Array[0] = threadIdx.x;  // Thread 0 writes 0, Thread 1 writes 1
}

// Test Case 3: No Data Race (Same Thread Writing to Array[tid] Twice, No Synchronization)
__global__ void test_case_3(int* Array) {
    // Thread 0 writes to Array[tid] twice
    Array[threadIdx.x] = threadIdx.x;  // Thread 0 writes 0 to Array[0], Thread 1 writes 1 to Array[1]
    Array[threadIdx.x] = threadIdx.x;  // Thread 0 writes 0 to Array[0] again
}

int main() {
  // Dummy main function
    return 0;
}
