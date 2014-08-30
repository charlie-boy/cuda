#include <stdio.h>

__global__
void matrixScaling(float *A, float *B, int n, int m)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i<n && j<m)
		B[i*m+j] = 2*A[i*m+j];
}

__host__
void checkError(cudaError_t error)
{
	if(error != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
}

__host__
float* callKernel(float *h_a, int n, int m)
{
	cudaError_t err;
	float* d_a, d_b, h_b;
	int size = n*m*sizeof(float);
	
	err = cudaMalloc((void**)&d_a, size);
	checkError(cudaError_t err);
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	err = cudaMalloc((void**)&d_b, size);
	checkError(cudaError_t err);

	dim3 dimGrid((n-1)/16.0 + 1, (m-1)/16.0 + 1, 1);
	dim3 dimBlock(16, 16, 1);

	matrixScaling<<<dimGrid, dimBlock>>>(d_a, d_b, n, m);

	cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
	return h_b;
}

__host__
int main()
{
	float h_a[100][100], *h_b;
	int n=60, m=90, i, j;
	for(i=0;i<n;i++)
		for(j=0;j<m;j++)
	{
		h_a[i][j] = i;
	}
	h_b = callKernel(h_a, n, m);
	for(i=0;i<n;i++)
		for(j=0;j<m;j++)
			printf("%d\t",h_c[i][j]);

	return 0;
}