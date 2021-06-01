#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define NO_THREADS 256
//Error check-----
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}
//Error check-----

__global__ void d_cycles3(int* adj, int* xadj, int* results, int nnz, int nov) {
    int v1 = threadIdx.x + blockDim.x * blockIdx.x;
    if (v1 < nov) {
        int count = 0;
        int n1 = xadj[v1+1] - xadj[v1];
        for (int i1 = 0; i1 < n1; i1++) {
            int v2 = adj[xadj[v1]+i1];
            int n2 = xadj[v2+1] - xadj[v2];
            for (int i2 = 0; i2 < n2; i2++) {
                int v3 = adj[xadj[v2]+i2];
                if (v3 == v1)
                    continue;
                int n3 = xadj[v3+1] - xadj[v3];
                for (int i3 = 0; i3 < n3; i3++) {
                    int v4 = adj[xadj[v3]+i3];
                    if (v4 == v1) {
                        count++;
                    }
                }
            }
        }
        results[v1] = count;
    }
}
__global__ void d_cycles4(int* adj, int* xadj, int* results, int nnz, int nov) {
    int v1 = threadIdx.x + blockDim.x * blockIdx.x;
    if (v1 < nov) {
        int count = 0;
        int n1 = xadj[v1+1] - xadj[v1];
        for (int i1 = 0; i1 < n1; i1++) {
            int v2 = adj[xadj[v1]+i1];
            int n2 = xadj[v2+1] - xadj[v2];
            for (int i2 = 0; i2 < n2; i2++) {
                int v3 = adj[xadj[v2]+i2];
                if (v3 == v1)
                    continue;
                int n3 = xadj[v3+1] - xadj[v3];
                for (int i3 = 0; i3 < n3; i3++) {
                    int v4 = adj[xadj[v3]+i3];
                    if (v4 == v3 || v4 == v2 || v4 == v1)
                        continue;
                    int n4 = xadj[v4+1] - xadj[v4];
                    for (int i4 = 0; i4 < n4; i4++) {
                        int v5 = adj[xadj[v4]+i4];
                        if (v5 == v1) {
                            count++;
                        }
                    }
                }
            }
        }
        results[v1]=count;
    }
}

__global__ void d_cycles5(int* adj, int* xadj, int* results, int nnz, int nov) { 
    int v1 = threadIdx.x + blockDim.x * blockIdx.x;
    if (v1 < nov) {
        int count = 0;
        int n1 = xadj[v1+1] - xadj[v1];
        for (int i1 = 0; i1 < n1; i1++) {
            int v2 = adj[xadj[v1]+i1];
            int n2 = xadj[v2+1] - xadj[v2];
            for (int i2 = 0; i2 < n2; i2++) {
            int v3 = adj[xadj[v2]+i2];
            if (v3 == v1)
                continue;
            int n3 = xadj[v3+1] - xadj[v3];
            for (int i3 = 0; i3 < n3; i3++) {
                int v4 = adj[xadj[v3]+i3];
                if (v4 == v3 || v4 == v2 || v4 == v1)
                continue;
                int n4 = xadj[v4+1] - xadj[v4];
                for (int i4 = 0; i4 < n4; i4++) {
                    int v5 = adj[xadj[v4]+i4];
                    if (v5 == v3 || v5 == v4 || v5 == v2 || v5 == v1)
                        continue;
                    int n5 = xadj[v5+1] - xadj[v5];
                    for (int i5 = 0; i5 < n5; i5++) {
                        int v6 = adj[xadj[v5]+i5];
                        if (v6 == v1) {
                        count++;
                        }
                    }
                }
            }
            }
        }
        results[v1]=count;
    }
}
void wrapper(int* adj, int* xadj, int* results, int nnz, int nov, int k) {
    cudaEvent_t start, stop;
  
    int *d_adj, *d_xadj;
    cudaMalloc( (void**) &d_adj,  nnz * sizeof(int));
    cudaMalloc( (void**) &d_xadj, (nov+1) * sizeof(int));
    cudaMemcpy( d_adj,  adj,  nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_xadj, xadj, (nov+1) * sizeof(int), cudaMemcpyHostToDevice);
    int *d_results;
    cudaMalloc( (void**) &d_results,  (nov) * sizeof(int));

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    int NO_BLOCKS = ceil((nov+NO_THREADS-1)/NO_THREADS);
    if (k == 3) {
        d_cycles3<<<NO_BLOCKS,NO_THREADS>>>(d_adj, d_xadj, d_results, nnz, nov);
    } else if (k == 4) {
        d_cycles4<<<NO_BLOCKS,NO_THREADS>>>(d_adj, d_xadj, d_results, nnz, nov);
    } else if (k == 5) {
        d_cycles5<<<NO_BLOCKS,NO_THREADS>>>(d_adj, d_xadj, d_results, nnz, nov);
    } else {
        printf("ERROR: Invalid k value.\n");
        return;
    }

    

    gpuErrchk( cudaDeviceSynchronize() );
  
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaMemcpy( results, d_results, (nov) * sizeof(int), cudaMemcpyDeviceToHost);
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU scale took: %f s\n", elapsedTime/1000);
}