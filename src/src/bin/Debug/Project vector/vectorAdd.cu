#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define N 1000000 //~1GB
#define THREADS 2
#define BLOCKS 2

__global__ void VectorAdd(int *a, int *b, int *c, int n)
{
	int i=blockDim.x * blockIdx.x +threadIdx.x;

	while(i<n)
	{
		c[i] = (a[i] + b[i]);
        i+=blockDim.x;
	}
}

int main( void ) {

    clock_t start = clock();
    
    int *a, *b, *c;               // The arrays on the host CPU machine
    int *dev_a, *dev_b, *dev_c;   // The arrays for the GPU device

    // 2.a allocate the memory on the CPU
    a = (int*)malloc( N * sizeof(int) );
    b = (int*)malloc( N * sizeof(int) );
    c = (int*)malloc( N * sizeof(int) );

    // 2.b. fill the arrays 'a' and 'b' on the CPU with dummy values
    for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = i;
    }

    // 2.c. allocate the memory on the GPU
     cudaMalloc( (void**)&dev_a, N * sizeof(int) );
     cudaMalloc( (void**)&dev_b, N * sizeof(int) );
     cudaMalloc( (void**)&dev_c, N * sizeof(int) );

    // 2.d. copy the arrays 'a' and 'b' to the GPU
     cudaMemcpy( dev_a, a, N * sizeof(int),
                              cudaMemcpyHostToDevice );
     cudaMemcpy( dev_b, b, N * sizeof(int),
                              cudaMemcpyHostToDevice );

    // 3. Execute the vector addition 'kernel function' on th GPU device,
    // declaring how many blocks and how many threads per block to use.
    VectorAdd<<<BLOCKS,THREADS>>>( dev_a, dev_b, dev_c ,N);

    // 4. copy the array 'c' back from the GPU to the CPU
    cudaMemcpy( c, dev_c, N * sizeof(int),
                              cudaMemcpyDeviceToHost );

    // verify that the GPU did the work we requested
    bool success = true;
    int total=0;
    printf("Checking %d values in the array.\n", N);
    for (int i=0; i<N; i++) {
        if ((a[i] + b[i]) != c[i]) {
            printf( "Error:  %d + %d != %d\n", a[i], b[i], c[i] );
            success = false;
        }
        total += 1;
    }
    if (success)  printf( "We did it, %d values correct!\n", total );

    clock_t end = clock();
    int diff = end-start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("Time taken %d seconds %d milliseconds", msec/1000, msec%1000);

    // free the memory we allocated on the CPU
    free( a );
    free( b );
    free( c );

    // free the memory we allocated on the GPU
     cudaFree( dev_a );
     cudaFree( dev_b );
     cudaFree( dev_c );

    return 0;
    }