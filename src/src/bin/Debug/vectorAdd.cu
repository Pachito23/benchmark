#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define N 1024*1024*4 //4GB

/*
    Optimal for testing for memory is the VectorAdd_best
    My time: 11 ms
*/

//use the version with the blocks and threads pre-computed
__global__ void VectorAdd_best(int *a, int *b, int *c)
{
	int i=blockDim.x * blockIdx.x + threadIdx.x;

	if(i<N)
	{
		c[i] = a[i] + b[i];
	}
}

//can use a variable number of blocks and threads
/*
    To isolate threads set the thread number variable and block nr to 1
    To isolate blocks set the block number variable and block nr to 1
    To isolate everything set block number and thread number to both 1

    Remark: to see the real difference in speed use high number of blocks (1000) with 1 thread
            and do the same but inverted for threads
*/
__global__ void VectorAdd_isolation(int *a, int *b, int *c)
{
	int i=blockDim.x * blockIdx.x + threadIdx.x;

	while(i<N)
	{
		c[i] = a[i] + b[i];
        i+=blockDim.x;
	}

    /*    More explicit version
    int index=blockDim.x * blockIdx.x +threadIdx.x;
    int stride=blockDim.x;
    
    for(int i=index; i<N; i+=stride)
    {
        c[i] = (a[i] + b[i]);
    }*/
}


int main( void ) {

    //reading from file the type of test
    int multi_thread_test;
    FILE* fin = fopen("input.txt","r+");
    fscanf(fin,"%d",&multi_thread_test);
    fclose(fin)

    //multi_thread_test=0;

    int blocks_single=1;
    int threads_single=1;

    //pre-computed number of blocks and thread for best performance
    int Threads = 1024;
    int Blocks = (N+Threads-1)/Threads;

    int *a, *b, *c;               // The arrays on the host CPU machine
    int *dev_a, *dev_b, *dev_c;   // The arrays for the GPU device

    // Allocate the memory on the CPU
    a = (int*)malloc( N * sizeof(int) );
    b = (int*)malloc( N * sizeof(int) );
    c = (int*)malloc( N * sizeof(int) );

    // Fill the arrays 'a' and 'b' on the CPU with dummy values
    for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = i;
    }

    //START WARMUP

    // Allocate the memory on the GPU
     cudaMalloc( (void**)&dev_a, N * sizeof(int) );
     cudaMalloc( (void**)&dev_b, N * sizeof(int) );
     cudaMalloc( (void**)&dev_c, N * sizeof(int) );

    // Copy the arrays 'a' and 'b' to the GPU
     cudaMemcpy( dev_a, a, N * sizeof(int),
                              cudaMemcpyHostToDevice );
     cudaMemcpy( dev_b, b, N * sizeof(int),
                              cudaMemcpyHostToDevice );

    // Execute the vector addition 'kernel function' on th GPU device
    VectorAdd_best<<<Blocks,Threads>>>( dev_a, dev_b, dev_c);

    // free the memory we allocated on the GPU
     cudaFree( dev_a );
     cudaFree( dev_b );
     cudaFree( dev_c );

    //END WARMUP

    int total_time=0;
    int msec;

    for(int i=1;i<=10;i++)
    {

    clock_t start = clock();
    // Allocate again the memory on the GPU
     cudaMalloc( (void**)&dev_a, N * sizeof(int) );
     cudaMalloc( (void**)&dev_b, N * sizeof(int) );
     cudaMalloc( (void**)&dev_c, N * sizeof(int) );

    // Copy the arrays 'a' and 'b' to the GPU again
     cudaMemcpy( dev_a, a, N * sizeof(int),
                              cudaMemcpyHostToDevice );

     cudaMemcpy( dev_b, b, N * sizeof(int),
                              cudaMemcpyHostToDevice );

    // Execute the vector addition 'kernel function' on th GPU device again
    if(multi_thread_test)
        VectorAdd_best<<<Blocks,Threads>>>( dev_a, dev_b, dev_c);
    else
        VectorAdd_isolation<<<blocks_single,threads_single>>>( dev_a, dev_b, dev_c);

    cudaDeviceSynchronize();

    // Copy the array 'c' back from the GPU to the CPU again
    cudaMemcpy( c, dev_c, N * sizeof(int),
                              cudaMemcpyDeviceToHost );

    // free the memory we allocated on the GPU again
     cudaFree( dev_a );
     cudaFree( dev_b );
     cudaFree( dev_c );

     clock_t end = clock();

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

    int diff = end-start;
    msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
    printf("Time taken %d \n",msec);

    total_time+=msec;

    }

    printf("Total time taken %d milliseconds\n", total_time);


    // free the memory we allocated on the CPU
    free( a );
    free( b );
    free( c );

    //write in file
    FILE* f=fopen("results_vec_add.txt","w+");
    fprintf(f,"%d",total_time);
    fclose(f);

    return 0;
    }