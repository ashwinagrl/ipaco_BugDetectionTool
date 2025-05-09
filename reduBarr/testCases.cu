#include <stdio.h>

#define N 2 // For simplicity, just two threads in the block

// Test Case 1: Redundant Barrier
__global__ void test_case_1(int* Array) {
    Array[threadIdx.x] = threadIdx.x;  // Thread 0 writes 0 to Array[0], Thread 1 writes 1 to Array[1]
    __syncthreads(); 
    Array[threadIdx.x] = threadIdx.x;  // Thread 0 writes 0 to Array[0] again
}

// Test Case 2: Non Redundant Barrier
__global__ void test_case_2(int* Array) {
    Array[threadIdx.x] = threadIdx.x;  
    __syncthreads();
    Array[threadIdx.x+1] = threadIdx.x; 
}

int main() {
  // Dummy main function
    return 0;
}
