#ifndef CHECK_CUDA_ERRORS_H
#define CHECK_CUDA_ERRORS_H

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, char const* const file, int const line);

#endif