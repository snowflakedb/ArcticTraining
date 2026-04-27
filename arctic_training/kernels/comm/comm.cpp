#include "comm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("init_comm_group", &ds_create_comm, "create comm group");
    m.def("barrier", &ds_barrier, "barrier");
    m.def("broadcast", &ds_broadcast, "broadcast");
    m.def("wait_comm", &wait_comm, "wait on communication event");
    m.def("allReduce", &ds_allreduce, "AllReduce");
    m.def("alltoall", &ds_alltoall, "AllToAll");
    m.def("allGather", &ds_allgather, "AllGather");
    m.def("get_nccl_uid", &ds_get_nccl_uid, "Get NCCL UID");
    m.def("init_nccl_comm", &ds_create_nccl_comm, "Create NCCL Comm");
}
//////
