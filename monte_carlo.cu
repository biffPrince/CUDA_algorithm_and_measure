#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include <cstdio>
#include <cstdlib>
#include "curand_mtgp32.h"
#include "curand_mtgp32_host.h"
#include "curand_mtgp32_kernel.h"
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define NBLOCKS 200
#define BLOCKSIZE 256
#define TEST_RUNS 100

__constant__ unsigned int LOOP_COUNT;

struct curandSharedStateMtgp32 {
	unsigned int s[MTGP32_STATE_SIZE];
	int offset;
	int pIdx;
	int pos;
	unsigned int mask;
	int sh1;			/*< shift value 1. 0 < sh1 < 32. */
	int sh2;			/*< shift value 2. 0 < sh2 < 32. */
	unsigned int tbl[16];		/*< a small matrix. */
	unsigned int tmp_tbl[16];	/*< a small matrix for tempering. */
};

__forceinline__ __device__ unsigned int para_rec_shared(curandSharedStateMtgp32 * k, unsigned int X1, unsigned int X2, unsigned int Y) {
	unsigned int X = (X1 & k->mask) ^ X2;
	unsigned int MAT;

	X ^= X << k->sh1;
	Y = X ^ (Y >> k->sh2);
	MAT = k->tbl[Y & 0x0f];
	return Y ^ MAT;
}

__forceinline__ __device__ unsigned int temper_shared(curandSharedStateMtgp32 * k, unsigned int V, unsigned int T) {
	unsigned int MAT;

	T ^= T >> 16;
	T ^= T >> 8;
	MAT = k->tmp_tbl[T & 0x0f];
	return V ^ MAT;
}

__forceinline__ __device__ unsigned int shared_curand(curandSharedStateMtgp32 *state)
{
	unsigned int t;
	unsigned int d;
	unsigned int r;
	unsigned int o;

	d = blockDim.z * blockDim.y * blockDim.x;
	t = (blockDim.z * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;
	r = para_rec_shared(state, state->s[(t + state->offset) & MTGP32_STATE_MASK],
		state->s[(t + state->offset + 1) & MTGP32_STATE_MASK],
		state->s[(t + state->offset + state->pos) & MTGP32_STATE_MASK]);

	state->s[(t + state->offset + MTGPDC_N) & MTGP32_STATE_MASK] = r;

	o = temper_shared(state, r, state->s[(t + state->offset + state->pos - 1) & MTGP32_STATE_MASK]);

#if __CUDA_ARCH__ != 0
	__syncthreads();
#endif
	if (t == 0)
	{
		state->offset = (state->offset + d) & MTGP32_STATE_MASK;
	}
#if __CUDA_ARCH__ != 0
	__syncthreads();
#endif
	return o;
}


__global__ void do_steps(curandStateMtgp32_t *state, unsigned int *data )
{
	int thread_data = 0;
	for (int i = 0; i < LOOP_COUNT; ++i) {
		thread_data += curand(&state[blockIdx.x]);
	}
	data[blockIdx.x * blockDim.x + threadIdx.x] += thread_data;
}

__global__ void do_steps_shared(curandStateMtgp32_t *state, unsigned int *data)
{
	__shared__ curandSharedStateMtgp32 state_;
	state += blockIdx.x;
	// MTGP32_STATE_SIZE = 1024 bytes and we have 256 threads (ommiting range check)
#pragma unroll
	for (unsigned short p = 0; p < MTGP32_STATE_SIZE; p += blockDim.x) {
		state_.s[threadIdx.x + p] = state->s[threadIdx.x + p];
	}

	//copy to shared memory data used by the mtgp
	if (threadIdx.x == 0) {
		state_.pIdx = state->pIdx;
		state_.offset = state->offset;
		state_.pos = state->k->pos_tbl[state->pIdx];
		state_.sh1 = state->k->sh1_tbl[state->pIdx];
		state_.sh2 = state->k->sh2_tbl[state->pIdx];
		state_.mask = state->k->mask[0];
	}

	//Copy to shared memory temper tables and param tables TBL_SIZE = 16
	if (threadIdx.x < TBL_SIZE) {
		state_.tbl[threadIdx.x] = state->k->param_tbl[state_.pIdx][threadIdx.x];
		state_.tmp_tbl[threadIdx.x] = state->k->temper_tbl[state_.pIdx][threadIdx.x];
	}
	__syncthreads();
	int thread_data = 0;
	for (int i = 0; i < LOOP_COUNT; ++i) {
		thread_data += shared_curand(&state_);
	}
	data[blockIdx.x * blockDim.x + threadIdx.x] += thread_data;

	//copy back to global memory state and offset
	for (unsigned short p = 0; p < MTGP32_STATE_SIZE; p += blockDim.x) {
			state->s[threadIdx.x + p] = state_.s[threadIdx.x + p];
	}
	if (threadIdx.x == 0) {
		state->offset = state_.offset;
	}
}

void init_mtgp(curandStateMtgp32_t * state, mtgp32_kernel_params_t * params, unsigned int seed = 123)
{
	//Copy cuda kernel params from the pre-generated fast parmas 11213
	curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, params);

	//Initialize the kernel states
	curandMakeMTGP32KernelState(state, mtgp32dc_params_fast_11213, params, NBLOCKS, seed);
}

int main()
{
	unsigned int * ddata1, *ddata2;
	unsigned int * hdata1, *hdata2;
	unsigned int loop_count = 1000;
	const unsigned int size = NBLOCKS * BLOCKSIZE * sizeof(unsigned int);
	curandStateMtgp32_t * d_random_state_;
	mtgp32_kernel_params_t * d_random_params_;


	//allocate data for 200 blocks each with 256 threads for each method tested
	gpuErrchk(cudaMalloc(&ddata1, size));
	gpuErrchk(cudaMemset(ddata1, 0, size));
	gpuErrchk(cudaMalloc(&ddata2, size));
	gpuErrchk(cudaMemset(ddata2, 0, size));
	gpuErrchk(cudaMallocHost(&hdata1, size));
	gpuErrchk(cudaMallocHost(&hdata2, size));

	//Malloc memory for curand kernel params
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_random_params_), sizeof(mtgp32_kernel_params)));

	//Malloc memory for curand kernel states
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_random_state_), NBLOCKS * sizeof(curandStateMtgp32)));
	
	//Transfer loop bounds to const symbol
	gpuErrchk(cudaMemcpyToSymbol(LOOP_COUNT, &loop_count, sizeof(int)));

	init_mtgp(d_random_state_, d_random_params_, 123);
	for (int i = 0; i < TEST_RUNS; ++i)
		do_steps<<<NBLOCKS,BLOCKSIZE >>>(d_random_state_, ddata1);

	init_mtgp(d_random_state_, d_random_params_, 123);
	for (int i = 0; i < TEST_RUNS; ++i)
		do_steps_shared << <NBLOCKS, BLOCKSIZE >> >(d_random_state_, ddata2);

	cudaMemcpy(hdata1, ddata1, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(hdata2, ddata2, size, cudaMemcpyDeviceToHost);

	unsigned int comp = 0;
	for (int i = 0; i < NBLOCKS * BLOCKSIZE; ++i) {
		comp += hdata1[i] ^ hdata2[i];
	}
	if (comp == 0) {
		std::cout << "Both kernels returned same random numbers" << std::endl;
	} else {
		std::cerr << "My hacked code is wrong" << std::endl;
	}
	

	cudaFree(ddata1);
	cudaFree(ddata2);
	cudaFreeHost(hdata1);
	cudaFreeHost(hdata2);
	cudaFree(d_random_state_);
	cudaFree(d_random_state_);
}