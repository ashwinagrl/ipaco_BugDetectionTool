#include <stdio.h>

#define N 4 // Assume block size for tid range

// Case: Divergent (Thread-Dependent Condition)
__global__ void barrier_divergent_tid(int* A) {
  int tid = threadIdx.x;
  if (tid < (N / 2)) { // Condition depends on tid (True for 0, 1; False for 2, 3 if N=4)
    A[tid] = tid;
    __syncthreads(); // DIVERGENT
    A[tid] = tid * 10;
  } else {
    A[tid] = tid * -1;
  }
}

// Case : Not Divergent (Barrier Outside Condition)
__global__ void barrier_outside_divergence(int* A) {

  int tid = threadIdx.x;

  if (tid < (N / 2)) { // Divergent execution path
    A[tid] = tid;
  } else {
    A[tid] = tid * 100;
  }
  // Barrier is outside the conditional block. All threads reach here.
  __syncthreads(); // NOT DIVERGENT
  A[tid] = A[tid] + 1;
}

// Case: Not Divergent (Because all threads executing)
__global__ void barrier_NotDivergent_allThds(int* A) {
    int tid = threadIdx.x;
    if (1) { //
      A[tid] = tid;
      __syncthreads(); // Not DIVERGENT
      A[tid] = tid * 10;
    } else {
      A[tid] = tid * -1;
    }
  }

// Dummy main

int main() { return 0; }