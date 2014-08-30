#include <stdio.h>

__global__
void matrixMultiplication(float *A, float *B, float *C, int m, int n, int k)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int i, sum=0;
	if(row < m && col < k)
	{
		for(i=0;i<n;i++)
		{
			sum = sum + A[row*m + i]*B[col + i*k]
		}
		C[row*k + col] = sum;
	}
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
float* callKernel(float *h_a, float *h_b, int m, int n, int k)
{
	cudaError_t err;
	float* d_a, d_b, d_c, h_c;
	int size_a = m*n*sizeof(float), size_b = n*k*sizeof(float), size_c = m*k*sizeof(float);
	
	err = cudaMalloc((void**)&d_a, size_a);
	checkError(cudaError_t err);
	cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
	err = cudaMalloc((void**)&d_b, size_b);
	checkError(cudaError_t err);
	cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
	err = cudaMalloc((void**)&d_c, size_c);
	checkError(cudaError_t err);

	dim3 dimGrid((k-1)/16.0 + 1, (m-1)/16.0 + 1, 1);
	dim3 dimBlock(16, 16, 1);

	matrixMultiplication<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);

	cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
	return h_c;
}

__host__
int main()
{
	float h_a[100][100], h_b[100][100], *h_c;
	int m=60, n=80, k= 50, i, j;

	for(i=0;i<m;i++)
		for(j=0;j<n;j++)
	{
		h_a[i][j] = i;
	}
	for(i=0;i<n;i++)
		for(j=0;j<k;j++)
	{
		h_a[i][j] = i+1;
	}

	h_c = callKernel(h_a, h_b, m, n, k);

	for(i=0;i<m;i++)
		for(j=0;j<k;j++)
			printf("%d\t",h_c[i][j]);

	return 0;
}