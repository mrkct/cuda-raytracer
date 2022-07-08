#ifndef CHECK_CUDA_ERRORS_H
#define CHECK_CUDA_ERRORS_H

/**
 * @brief Checks wheter the expression has returned a cudaError other
 * than success; in that case halts the program and prints some info
 * about the source code and the cuda error message string
 */
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, char const* const file, int const line);

#endif