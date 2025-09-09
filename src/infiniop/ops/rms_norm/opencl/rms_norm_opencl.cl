#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef Ta
#define Ta float
#endif

#ifndef Tw
#define Tw float
#endif

#ifndef Tcompute
#define Tcompute float
#endif

#ifndef ITEMS_THREAD
#define ITEMS_THREAD 1
#endif

typedef unsigned int Tidx;

kernel void rms_norm(
    global Ta *y_,
    int const s_y_batch,
    int const s_y_nhead,
    global Ta const *x_,
    int const s_x_batch,
    int const s_x_nhead,
    global Tw const *w,
    float const epsilon,
    Tidx const nhead,
    Tidx const d) {

    Tidx g_idx = get_group_id(0),
         l_idx = get_local_id(0),
         l_len = get_local_size(0);
    Tidx batch_id = g_idx / nhead,
         nhead_id = g_idx % nhead;
    global Ta
        *y
        = y_ + batch_id * s_y_batch + nhead_id * s_y_nhead;
    global Ta const
        *x
        = x_ + batch_id * s_x_batch + nhead_id * s_x_nhead;

    Tcompute val_x[ITEMS_THREAD];
    Tcompute val_w[ITEMS_THREAD];
    Tcompute squared = 0;
    for (Tidx i = 0, idx = l_idx; idx < d; ++i, idx += l_len) {
        val_x[i] = (Tcompute)x[idx];
        val_w[i] = (Tcompute)w[idx];
        squared += val_x[i] * val_x[i];
    }
    // TODO:测试加载相邻元素处理；
    Tcompute mean_sq = work_group_reduce_add(squared) / (Tcompute)d;
    Tcompute rms = native_rsqrt(mean_sq + (Tcompute)epsilon);

    for (Tidx i = 0, idx = l_idx; idx < d; ++i, idx += l_len) {
        y[idx] = (Ta)(rms * val_x[i] * val_w[i]);
    }
}
