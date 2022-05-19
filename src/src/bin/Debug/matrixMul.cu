#include <stdio.h>
#include <time.h>
#define SIZE 512*4 //4GB

//maximum nr of SIZE recommended is: SIZE<512*4 = 4GB 

//parallelized matrix multiplication algorithm
__global__ void matrix_mult (int *m1, int *m2, int *res)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;	
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row<SIZE && col<SIZE)
	{
		int result=0;
		for(int k=0;k<SIZE;k++)
		{
			result+=m1[row*SIZE+k]*m2[k*SIZE+col];
		}
		res[row*SIZE+col]=result;
	}
}

//CPU testing that all the values of the matrix are correct
void matrix_mult_test (int *m1, int *m2, int *res)
{
	printf("Checking all the %d values in the matrix.\n", SIZE*SIZE);
	bool success = true;
	for(int i=0;i<SIZE;i++)
	{
		for(int j=0;j<SIZE;j++)
		{
			int correct_value=0;
			for(int k=0;k<SIZE;k++)
			{
				correct_value+=m1[i*SIZE+k]*m2[k*SIZE+j];
			}
			if(correct_value!=res[i*SIZE+j])
			{
				printf( "Error at element (%d,%d)\n",i,j);
				success = false;
			}
		}
	}
	if (success)  
		printf( "We did it, all the values are correct!\n");
}

//initializion of elements of the matrix
void matrix_init(int *m)
{
	for(int i=0;i<SIZE;i++)
	{
		for(int j=0;j<SIZE;j++)
		{
			m[i*SIZE+j]=i+j;
		}
	}
}

int main()
{
	//reading from file the type of test
    int multi_thread_test;
    FILE* fin = fopen("input.txt","r+");
    fscanf(fin,"%d",&multi_thread_test);

	//multi_thread_test=1;

	//pre-computed number of blocks and thread for best performance
	/*
		Blocks are not to be modified but threads can be
		Threads range from 1 to 32
		Threads and blocks are inversely proportional => 
			more threads -> less blocks
			less threads -> more blocks
	*/

	int Threads;
	if(multi_thread_test)
		Threads = 32;
	else
		Threads = 1;

    int Blocks = (SIZE + Threads - 1)/Threads;

	printf("Number of blocks : %d\nNumber of threads per block : %d\n",Blocks,Threads);

	//we create a 2D array of blocks on the grid with respective threads
	dim3 THREADS(Threads,Threads);
	dim3 BLOCKS(Blocks,Blocks);

	int *m1,*m2,*res; // The arrays on the host CPU machine
	int *a,*b,*c; // The arrays for the GPU device

	// Allocate the memory on the CPU
    m1 = (int*)malloc( SIZE * SIZE * sizeof(int));
    m2 = (int*)malloc( SIZE * SIZE * sizeof(int));
    res = (int*)malloc( SIZE * SIZE * sizeof(int));

	//initilize CPU matrix
	matrix_init(m1);
	matrix_init(m2);


	//START WARMUP

	// Allocate the memory on the GPU
    cudaMalloc( (void***)&a, SIZE * SIZE * sizeof(int));
    cudaMalloc( (void***)&b, SIZE * SIZE * sizeof(int));
    cudaMalloc( (void***)&c, SIZE * SIZE * sizeof(int));

	// Copy the matrices 'm1' and 'm2' to the GPU
     cudaMemcpy( a, m1, SIZE * SIZE * sizeof(int),
                              cudaMemcpyHostToDevice );
     cudaMemcpy( b, m2, SIZE * SIZE * sizeof(int),
                              cudaMemcpyHostToDevice );

	//we compute the matrix with the parallel algorithm
	matrix_mult<<<BLOCKS,THREADS>>>(a,b,c);

	//we synchronize the GPU to have stability in the result
	cudaDeviceSynchronize();

	// Copy the matrix 'c' back from the GPU to the CPU
    cudaMemcpy( res, c, SIZE * SIZE * sizeof(int),
                              cudaMemcpyDeviceToHost );

	// free the memory we allocated on the GPU
    cudaFree( a );
    cudaFree( b );
    cudaFree( c );

	//END WARMUP


	clock_t start = clock();

	// Allocate the memory on the GPU
    cudaMalloc( (void***)&a, SIZE * SIZE * sizeof(int));
    cudaMalloc( (void***)&b, SIZE * SIZE * sizeof(int));
    cudaMalloc( (void***)&c, SIZE * SIZE * sizeof(int));

	// Copy the matrices 'm1' and 'm2' to the GPU
     cudaMemcpy( a, m1, SIZE * SIZE * sizeof(int),
                              cudaMemcpyHostToDevice );
     cudaMemcpy( b, m2, SIZE * SIZE * sizeof(int),
                              cudaMemcpyHostToDevice );

	//we compute the matrix with the parallel algorithm
	matrix_mult<<<BLOCKS,THREADS>>>(a,b,c);

	//we synchronize the GPU to have stability in the result
	cudaDeviceSynchronize();

	// Copy the matrix 'c' back from the GPU to the CPU
    cudaMemcpy( res, c, SIZE * SIZE * sizeof(int),
                              cudaMemcpyDeviceToHost );

	// free the memory we allocated on the GPU
    cudaFree( a );
    cudaFree( b );
    cudaFree( c );

	clock_t end = clock();

	//we verify the GPU computed everything corrected as we requested
	matrix_mult_test(m1,m2,res);

	//caluculate the time elapsed
	int diff = end-start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

	// free the memory we allocated on the CPU
    free( m1 );
    free( m2 );
    free( res );

	//write in file
    FILE* f=fopen("results_matrix_mult.txt","w+");
    fprintf(f,"%d",msec);
    fclose(f);

	return 0;
}