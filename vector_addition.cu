#include <stdio.h>

__global__
void vecAddition(float *A, float *B, float *C, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n)
		C[i] = A[i] + B[i];
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
float* callKernel(float *h_a, float *h_b, int n)
{
	cudaError_t err;
	float* d_a, d_b, d_c, h_c;
	int size = n*sizeof(float);
	
	err = cudaMalloc((void**)&d_a, size);
	checkError(cudaError_t err);
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	err = cudaMalloc((void**)&d_b, size);
	checkError(cudaError_t err);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
	err = cudaMalloc((void**)&d_c, size);
	checkError(cudaError_t err);

	dim3 dimGrid((n-1)/256.0 + 1, 1, 1);
	dim3 dimBlock(256, 1, 1);

	vecAddition<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);

	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
	return h_c;
}

__host__
int main()
{
	float h_a[100], h_b[100], *h_c;
	int n=60, i;
	for(i=0;i<n;i++)
	{
		h_a[i] = i;
		h_b[i] = n-i;
	}
	h_c = callKernel(h_a, h_b, n);
	for(i=0;i<n;i++)
		printf("%d\t",h_c[i]);

	return 0;
}