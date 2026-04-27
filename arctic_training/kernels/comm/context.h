#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <iostream>
#include <vector>
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"

#include <cuda.h>
#include <nccl.h>
#include <stdlib.h>
#include <sys/time.h>
#include <map>
#include <memory>
#include <stack>
#include <string>
#include <cuda_runtime_api.h>
#define WARP_SIZE 32

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                                                          \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x) \
        for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); j += blockDim.y * gridDim.y)

#define DS_CUDA_NUM_THREADS 512
#define DS_MAXIMUM_NUM_BLOCKS 262144

inline int DS_GET_BLOCKS(const int N)
{
    return std::max(
        std::min((N + DS_CUDA_NUM_THREADS - 1) / DS_CUDA_NUM_THREADS, DS_MAXIMUM_NUM_BLOCKS),
        // Use at least 1 block, since CUDA does not allow empty block
        1);
}

class CoMMContext {
public:
    CoMMContext()
        : _workspace(nullptr),
          _seed(42),
          _curr_offset(0),
          _comm_stream(0),
          _comp_stream(0),
          _comm_created(false)
    {
        curandCreateGenerator(&_gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(_gen, 123);
        if (cublasCreate(&_cublasHandle) != CUBLAS_STATUS_SUCCESS) {
            auto message = std::string("Fail to create cublas handle.");
            std::cerr << message << std::endl;
            throw std::runtime_error(message);
        }
        cublasSetMathMode(_cublasHandle, CUBLAS_TENSOR_OP_MATH);
        cudaEventCreate(&_comp_event, (cudaEventDisableTiming | cudaEventBlockingSync));
        cudaEventCreate(&_comm_event, (cudaEventDisableTiming | cudaEventBlockingSync));
    }

    virtual ~CoMMContext()
    {
        cublasDestroy(_cublasHandle);
        cudaFree(_workspace);
        ncclCommDestroy(_nccl_comm);
        cudaEventDestroy(_comp_event);
        cudaEventDestroy(_comm_event);
    }

    static CoMMContext& Instance()
    {
        static CoMMContext _ctx;
        return _ctx;
    }

    void create_comm_group(std::vector<int> comm_ranks, int rank)
    {
        //
    }

    inline ncclUniqueId get_nccl_uid()
    {
        ncclUniqueId _nccl_uid;
        ncclGetUniqueId(&_nccl_uid);
        return _nccl_uid;
    }

    void create_nccl_comm(std::vector<int> comm_ranks, int rank, void* nccl_uid_ptr){

        unsigned num_ranks = comm_ranks.size();
        ncclUniqueId _nccl_uid = *((ncclUniqueId*)nccl_uid_ptr);
        _nranks = num_ranks;
        ncclCommInitRank(&_nccl_comm, num_ranks, _nccl_uid, rank);
        printf("********** nccl comm: %p \n", _nccl_comm);

    }
    inline ncclComm_t GetNCCLComm() { return _nccl_comm; }

    inline unsigned GetNumRanks() const { return _nranks; }

     inline void barrier() {
        //
    }

    inline void SynchComp()
    {
        cudaEventRecord(_comp_event, _comp_stream);
        cudaStreamWaitEvent(_comm_stream, _comp_event, 0);
    }
    inline void SynchComm()
    {
        cudaEventRecord(_comm_event, _comm_stream);
        cudaStreamWaitEvent(_comp_stream, _comm_event, 0);
    }
    void GenWorkSpace(size_t size)
    {
        if (!_workspace) {
            assert(_workspace == nullptr);
            cudaMalloc(&_workspace, size);
        } else if (_workSpaceSize < size) {
            cudaFree(_workspace);
            cudaMalloc(&_workspace, size);
        }

        _workSpaceSize = size;
    }

    void* GetWorkSpace() { return _workspace; }

    curandGenerator_t& GetRandGenerator() { return _gen; }

    cudaStream_t GetCurrentStream()
    {
        // get current pytorch stream.
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        return stream;
    }

    cudaStream_t GetCommStream(bool async_op = false)
    {
        // if (!_comm_stream)
        //     _comm_stream = async_op ? at::cuda::getStreamFromPool(true)
        //                             : at::cuda::getCurrentCUDAStream();
        return at::cuda::getCurrentCUDAStream(); //_comm_stream;
    }

    cublasHandle_t GetCublasHandle() { return _cublasHandle; }

    std::pair<uint64_t, uint64_t> IncrementOffset(uint64_t offset_inc)
    {
        uint64_t offset = _curr_offset;
        _curr_offset += offset_inc;
        return std::pair<uint64_t, uint64_t>(_seed, offset);
    }

    void SetSeed(uint64_t new_seed) { _seed = new_seed; }

private:
    curandGenerator_t _gen;
    cublasHandle_t _cublasHandle;
    cudaEvent_t _comp_event;
    cudaEvent_t _comm_event;

    void* _workspace;
    uint64_t _seed;
    uint64_t _curr_offset;
    size_t _workSpaceSize;
    cudaStream_t _comp_stream;
    cudaStream_t _comm_stream;
    ncclComm_t _nccl_comm;
    unsigned _nranks;
    bool _comm_created;
};
