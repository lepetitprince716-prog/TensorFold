// DeepSeek-V4.1's mHC decode boundary as one CUDA launch (dsv41_ops.cu includes it; same flags, same bits).
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

// _hc_post_only, then _hc_finish_s's collapse + RMSNorm of what it wrote (SINK 0, NROT 0): one launch, the same bits.
extern "C" __global__ void __cluster_dims__(8, 1, 1) __launch_bounds__(32) tf_ds_hc_post_finish_kernel(
    const __nv_bfloat16* x, __nv_bfloat16* xo, const float* g, long long rs, int world, const float* post,
    const float* comb, const float* pre_in, const __nv_bfloat16* nw, __nv_bfloat16* out, float eps) {
    constexpr int NC = 5, D = NC * 1024; // a row a cluster of 8 one-warp blocks, block k the finish's warp k
    const long long r = blockIdx.x >> 3;
    unsigned k;
    asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(k));
    __shared__ float red[8];
    if (threadIdx.x < 8) red[threadIdx.x] = __int_as_float(0x7fffffff); // a NaN every block's warp sum replaces
    // the cluster's blocks must all have started before one writes another's shared memory (the wait before the sum)
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    const int col = 4 * (32 * k + threadIdx.x);
    auto f = [](unsigned b) { return __uint_as_float(b << 16); };
    auto rb = [](float v) { return (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(v)); };
    auto lane = [](uint2 u, int e) { return (e & 2 ? u.y : u.x) >> (e & 1 ? 16 : 0) & 0xffffu; };
    uint2 w[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) w[c] = *reinterpret_cast<const uint2*>(nw + c * 1024 + col);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    uint2 xs[4][NC];
    float4 ga[NC], gb[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int s = 0; s < 4; ++s) xs[s][c] = *reinterpret_cast<const uint2*>(x + (r * 4 + s) * D + c * 1024 + col);
        ga[c] = *reinterpret_cast<const float4*>(g + r * D + c * 1024 + col);
        if (world > 1) gb[c] = *reinterpret_cast<const float4*>(g + rs + r * D + c * 1024 + col);
    }
    float cm[16], ps[4], p[4];
#pragma unroll
    for (int i = 0; i < 16; ++i) cm[i] = comb[r * 16 + i];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        ps[s] = post[r * 4 + s];
        p[s] = pre_in[r * 4 + s];
    }
    float cc[NC][4], acc[4];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float y[4] = {ga[c].x, ga[c].y, ga[c].z, ga[c].w};
        if (world > 1) {
            const float b[4] = {gb[c].x, gb[c].y, gb[c].z, gb[c].w};
#pragma unroll
            for (int e = 0; e < 4; ++e) y[e] = __fadd_rn(y[e], b[e]);
        }
        for (int q = 2; q < world; ++q) {
            const float4 v = *reinterpret_cast<const float4*>(g + q * rs + r * D + c * 1024 + col);
            const float b[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) y[e] = __fadd_rn(y[e], b[e]);
        }
        float h[4][4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float ye = f(rb(y[e]));
            const float x0 = f(lane(xs[0][c], e)), x1 = f(lane(xs[1][c], e));
            const float x2 = f(lane(xs[2][c], e)), x3 = f(lane(xs[3][c], e));
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const float v = __fmaf_rn(cm[8 + s], x2, __fmaf_rn(cm[s], x0, __fmul_rn(cm[4 + s], x1)));
                h[s][e] = f(rb(__fmaf_rn(ps[s], ye, __fmaf_rn(cm[12 + s], x3, v))));
            }
            // as the finish's SASS: even columns take fma(p1, x1, p0 * x0), odd ones fma(p0, x0, p1 * x1)
            const float s01 = e & 1 ? __fmaf_rn(p[0], h[0][e], __fmul_rn(p[1], h[1][e]))
                                    : __fmaf_rn(p[1], h[1][e], __fmul_rn(p[0], h[0][e]));
            cc[c][e] = f(rb(__fmaf_rn(p[3], h[3][e], __fmaf_rn(p[2], h[2][e], s01))));
            acc[e] = __fmaf_rn(cc[c][e], cc[c][e], c == 0 ? 0.0f : acc[e]);
        }
#pragma unroll
        for (int s = 0; s < 4; ++s)
            *reinterpret_cast<uint2*>(xo + (r * 4 + s) * D + c * 1024 + col) =
                make_uint2(rb(h[s][0]) | rb(h[s][1]) << 16, rb(h[s][2]) | rb(h[s][3]) << 16);
    }
    // tl.sum's tree: a thread's columns in order, butterflies over the lanes (16 .. 1), then over the 8 warps (4 .. 1)
    float sum = __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]), acc[3]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, off));
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    if (threadIdx.x < 8) {
        unsigned a = (unsigned)__cvta_generic_to_shared(&red[k]), remote;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(a), "r"(threadIdx.x));
        asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(remote), "f"(sum) : "memory");
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    sum = __fadd_rn(__fadd_rn(__fadd_rn(red[0], red[4]), __fadd_rn(red[2], red[6])),
                    __fadd_rn(__fadd_rn(red[1], red[5]), __fadd_rn(red[3], red[7])));
    if (sum != sum) __trap(); // a warp sum that never arrived (or a NaN in the row)
    float sq, rinv;
    asm("sqrt.approx.f32 %0, %1;" : "=f"(sq) : "f"(__fmaf_rn(sum, 1.0f / D, eps)));
    asm("div.full.f32 %0, %1, %2;" : "=f"(rinv) : "f"(1.0f), "f"(sq));
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        unsigned o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) o[e] = rb(__fmul_rn(f(lane(w[c], e)), __fmul_rn(cc[c][e], rinv)));
        *reinterpret_cast<uint2*>(out + r * D + c * 1024 + col) = make_uint2(o[0] | o[1] << 16, o[2] | o[3] << 16);
    }
}
