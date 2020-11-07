
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include <stdio.h>

#define CHECK(_t, _e) if (_e != cudaSuccess) { fprintf(stderr, "%s failed: %s", _t, cudaGetErrorString(_e)); goto Error; }
#define HERR(_t, _e) if (_e != cudaSuccess) { fprintf(stderr, "%s failed: %s", _t, cudaGetErrorString(_e)); }

__global__ void getBlockDimAndGridDim(int * blockDimX, int * blockDimY, int * gridDimX, int * gridDimY) {
	*blockDimX = blockDim.x;
	*blockDimY = blockDim.y;
	*gridDimX = gridDim.x;
	*gridDimY = gridDim.y;
}

__global__ void init_device_a(int * device_a, int * max_index)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int offset = row * gridDim.x * blockDim.x + col;
	if(offset < *max_index)
		device_a[offset] = offset;
}

/* A helper function to print a matrix a with height and width*/
void print_a(const int * a, int height, int width) {
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) printf("%d ", a[row * width + col]);
		printf("\n");
	}
}

void cudaGenerateMatrixFunction() {

	int HEIGHT = 8;
	int WIDTH = 16;

	int * a = (int *)malloc(HEIGHT * WIDTH * sizeof(int));

	int * device_a;
	CHECK("cudaMalloc device_a", cudaMalloc((void **)&device_a, HEIGHT * WIDTH * sizeof(int)));

	dim3 blockDimCust(8, 2);
	dim3 grid(8);
	int max_index = HEIGHT * WIDTH;
	int *device_max_index;

	CHECK("cudaMalloc device_max_index", cudaMalloc((void **)&device_max_index, sizeof(int)));
	CHECK("cudaMemcpy device_max_index max_index", cudaMemcpy(device_max_index, &max_index, sizeof(int), cudaMemcpyHostToDevice));

	init_device_a <<<grid, blockDimCust>>> (device_a, device_max_index);

	CHECK("cudaMemcpy a device_a", cudaMemcpy(a, device_a, HEIGHT * WIDTH * sizeof(int), cudaMemcpyDeviceToHost));

	print_a(a, HEIGHT, WIDTH);

Error:
	HERR("cudaFree device_a", cudaFree(device_a));
	HERR("cudaFree device_max_index", cudaFree(device_max_index));

}

void getDimensions() {
	int HEIGHT = 8;
	int WIDTH = 16;

	dim3 blockDimCust(8, 2);
	dim3 grid(8);

	int * deviceBlockDimX;
	int * deviceBlockDimY;
	int * deviceGridDimX;
	int * deviceGridDimY;

	CHECK("cudaMalloc deviceBlockDimX", cudaMalloc((void **)&deviceBlockDimX, sizeof(int)));
	CHECK("cudaMalloc deviceBlockDimY", cudaMalloc((void **)&deviceBlockDimY, sizeof(int)));
	CHECK("cudaMalloc deviceGridDimX", cudaMalloc((void **)&deviceGridDimX, sizeof(int)));
	CHECK("cudaMalloc deviceGridDimY", cudaMalloc((void **)&deviceGridDimY, sizeof(int)));

	getBlockDimAndGridDim<<<grid, blockDimCust>>>(deviceBlockDimX, deviceBlockDimY, deviceGridDimX, deviceGridDimY);

	int blockDimX, blockDimY, gridDimX, gridDimY;

	CHECK("cudaMemcpy blockDimX deviceBlockDimX", cudaMemcpy(&blockDimX, deviceBlockDimX, sizeof(int), cudaMemcpyDeviceToHost));
	CHECK("cudaMemcpy blockDimY deviceBlockDimY", cudaMemcpy(&blockDimY, deviceBlockDimY, sizeof(int), cudaMemcpyDeviceToHost));
	CHECK("cudaMemcpy ", cudaMemcpy(&gridDimX, deviceGridDimX, sizeof(int), cudaMemcpyDeviceToHost));
	CHECK("cudaMemcpy blockDimX deviceBlockDimX", cudaMemcpy(&gridDimY, deviceGridDimY, sizeof(int), cudaMemcpyDeviceToHost));

	printf("blockDimX = %d, blockDimY = %d, gridDimX = %d, gridDimY = %d\n", blockDimX, blockDimY, gridDimX, gridDimY);

Error:
	HERR("cudaFree deviceBlockDimX", cudaFree(deviceBlockDimX));
	HERR("cudaFree deviceBlockDimY", cudaFree(deviceBlockDimY));
	HERR("cudaFree deviceGridDimX", cudaFree(deviceGridDimX));
	HERR("cudaFree deviceGridDimY", cudaFree(deviceGridDimY));

}

int main()
{

	getDimensions();
	cudaGenerateMatrixFunction();

	return 0;
}