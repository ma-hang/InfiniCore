#ifndef __INFINIRT_OPENCL_H__
#define __INFINIRT_OPENCL_H__
#include "../infinirt_impl.h"

namespace infinirt::opencl {
infiniStatus_t init();
#ifdef ENABLE_OPENCL_API
INFINIRT_DEVICE_API_IMPL
#else
INFINIRT_DEVICE_API_NOOP
#endif
} // namespace infinirt::opencl

#endif // __INFINIRT_OPENCL_H__
