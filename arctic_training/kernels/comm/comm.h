
#include <torch/extension.h>
#include <stdint.h>

#include "stdio.h"

void ds_barrier();
void wait_comm();
void ds_create_comm(std::vector<int>& comm_ranks, int rank);
void ds_allreduce(torch::Tensor& send_buf, torch::Tensor& rcv_buf, int size, bool async_op);
void ds_allgather(torch::Tensor& send_buf, torch::Tensor& rcv_buf, int size, bool async_op);
void ds_broadcast(torch::Tensor& send_buf, torch::Tensor& rcv_buf, int size, bool async_op);
void ds_alltoall(torch::Tensor& send_buf, torch::Tensor& rcv_buf, torch::Tensor& counts, size_t max_count, bool async_op);
torch::Tensor ds_get_nccl_uid();
void ds_create_nccl_comm(std::vector<int>& comm_ranks, int rank, torch::Tensor& nccl_uid);
