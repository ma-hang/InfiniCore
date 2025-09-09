#ifndef __INFINIOP_OPENCL_COMMON_H__
#define __INFINIOP_OPENCL_COMMON_H__

#include "../../../utils.h"
#include "../pool.h"
#include "opencl_handle.h"
#include <functional>
#include <vector>

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#include <CL/cl.h>

namespace device::opencl {

class Handle::Internal {

    int _warp_size,
        _max_threads_per_block,
        _block_size[3];

    template <typename T>
    using Fn = std::function<infiniStatus_t(T)>;

public:
    Internal(int);

    int warpSize() const;
    int maxThreadsPerBlock() const;
    int blockSizeX() const;
    int blockSizeY() const;
    int blockSizeZ() const;
};

} // namespace device::opencl

#endif // __INFINIOP_OPENCL_COMMON_H__
