# Hand-written Metal kernels for the hot ops, using Apple's simdgroup 8x8 matrix units, which
# Mojo cannot target yet. Device compiles this source once at startup (DeviceContext.load_function)
# and launches it like a Mojo kernel: argument i is bound to buffer(i), pointers as GPU addresses.
# Each kernel has a Mojo stub (same signature, empty body) that only gives load_function its type.

comptime FP = Pointer[Float32, MutAnyOrigin]


def gemm_stub(c: FP, a: FP, b: FP, M: Int32, N: Int32, K: Int32, H: Int32,
              sAi: Int32, sAk: Int32, sBk: Int32, sBj: Int32, sCi: Int32,
              zbA: Int32, zhA: Int32, zbB: Int32, zhB: Int32, zbC: Int32, zhC: Int32,
              alpha: Float32, acc: Int32, tiles_n: Int32, bias: UInt64):
    pass


def attn_fwd_stub(att: FP, lse: FP, qkv: FP, D: Int32, H: Int32, scale: Float32, T: Int32, FT: Int32,
                  band: Int32):
    pass


def attn_bwd_stub(dqkv: FP, qkv: FP, datt: FP, lse: FP, dvec: FP, D: Int32, H: Int32, scale: Float32,
                  T: Int32, FT: Int32, band: Int32):
    pass


comptime MSL = """
#include <metal_stdlib>
using namespace metal;
typedef simdgroup_float8x8 frag;
#define DPTR(name) ((device float *)(name))

// Strided, batched C = alpha * A @ B (+ bias[col] when biasp != 0) (+ C), the contract of
// b_gemm in gpu.mojo plus an optional bias, with C unit
// stride along columns. One 256-thread threadgroup computes a 64x64 tile of C, each of its 8
// simdgroups a 16x32 block as 2x4 fragments. Needs M, N % 64 == 0 and K % 16 == 0.
// TA: A is stored k-major (sAi == 1) instead of i-major (sAk == 1); TB likewise for B (sBk == 1).
#define GEMM_PARAMS \\
    constant ulong &cp [[buffer(0)]], constant ulong &ap [[buffer(1)]], constant ulong &bp [[buffer(2)]], \\
    constant int &M [[buffer(3)]], constant int &N [[buffer(4)]], constant int &K [[buffer(5)]], \\
    constant int &H [[buffer(6)]], constant int &sAi [[buffer(7)]], constant int &sAk [[buffer(8)]], \\
    constant int &sBk [[buffer(9)]], constant int &sBj [[buffer(10)]], constant int &sCi [[buffer(11)]], \\
    constant int &zbA [[buffer(12)]], constant int &zhA [[buffer(13)]], constant int &zbB [[buffer(14)]], \\
    constant int &zhB [[buffer(15)]], constant int &zbC [[buffer(16)]], constant int &zhC [[buffer(17)]], \\
    constant float &alpha [[buffer(18)]], constant int &acc [[buffer(19)]], constant int &tiles_n [[buffer(20)]], \\
    constant ulong &biasp [[buffer(21)]], \\
    uint blk [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], \\
    ushort tid [[thread_index_in_threadgroup]]
#define GEMM_ARGS cp, ap, bp, M, N, K, H, sAi, sAk, sBk, sBj, sCi, zbA, zhA, zbB, zhB, zbC, zhC, alpha, acc, tiles_n, biasp, blk, sg, tid, As, Bs

// The A (64 x BK) and B (BK x 64) slices are staged through threadgroup memory in the
// orientation that keeps global reads contiguous (float4s, so offsets and strides must be
// multiples of 4); simdgroup_load transposes as needed. The next slice is fetched into
// registers while the current one is multiplied.
// A slice is [k][i] with row stride LDT when k-major in memory, else [i][k] with stride LDN.
#define BK 16
#define LDT (64 + 4)
#define LDN (BK + 4)
#define TILE (64 * LDN)

// Global -> registers: 1 float4 per thread per operand. k-major (T) slices are BK rows of 64,
// i-major slices 64 rows of BK.
template <bool T>
inline void fetch(thread float4 *r, const device float *P, int s, int o0, int k0, ushort tid) {
    int idx = tid;
    if (T) { int kk = idx / 16, i = (idx % 16) * 4; *r = *(const device float4 *)(P + (o0 + i) + (k0 + kk) * s); }
    else { int i = idx / 4, kk = (idx % 4) * 4; *r = *(const device float4 *)(P + (o0 + i) * s + k0 + kk); }
}

template <bool T>
inline void stash(threadgroup float *S, thread float4 *r, ushort tid) {
    int idx = tid;
    if (T) { int kk = idx / 16, i = (idx % 16) * 4; *(threadgroup float4 *)(S + kk * LDT + i) = *r; }
    else { int i = idx / 4, kk = (idx % 4) * 4; *(threadgroup float4 *)(S + i * LDN + kk) = *r; }
}

template <bool TA, bool TB>
inline void gemm_body(ulong cp, ulong ap, ulong bp, int M, int N, int K, int H, int sAi, int sAk,
                      int sBk, int sBj, int sCi, int zbA, int zhA, int zbB, int zhB, int zbC, int zhC,
                      float alpha, int acc, int tiles_n, ulong biasp, uint blk, ushort sg, ushort tid,
                      threadgroup float *As, threadgroup float *Bs) {
    int per_z = (M / 64) * tiles_n;
    int z = blk / per_z;
    int rem = blk - z * per_z;
    int tm = rem / tiles_n;
    int tn = rem - tm * tiles_n;
    int zb = z / H;
    int zh = z - zb * H;
    int R0 = tm * 64;
    int C0 = tn * 64;
    int sr = (sg / 2) * 16;
    int sc = (sg % 2) * 32;
    const device float *A = DPTR(ap) + zb * zbA + zh * zhA;
    const device float *B = DPTR(bp) + zb * zbB + zh * zhB;
    device float *C = DPTR(cp) + zb * zbC + zh * zhC;
    // A's k stride / i stride, and B's k stride / j stride, as the fetch orientation needs.
    int sa = TA ? sAk : sAi;
    int sb = TB ? sBj : sBk;
    frag cf[8];  // 2 x 4 fragments: more than 8 accumulators spill out of registers
    #pragma unroll
    for (int i = 0; i < 8; i++)
        cf[i] = frag(0.0f);
    float4 ra, rb;
    fetch<TA>(&ra, A, sa, R0, 0, tid);
    fetch<!TB>(&rb, B, sb, C0, 0, tid);
    for (int k0 = 0; k0 < K; k0 += BK) {
        stash<TA>(As, &ra, tid);
        stash<!TB>(Bs, &rb, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + BK < K) {
            fetch<TA>(&ra, A, sa, R0, k0 + BK, tid);
            fetch<!TB>(&rb, B, sb, C0, k0 + BK, tid);
        }
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            frag af[2], bf[4];
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                if (TA) simdgroup_load(af[i], As + kk * LDT + sr + 8 * i, LDT, ulong2(0, 0), true);
                else simdgroup_load(af[i], As + (sr + 8 * i) * LDN + kk, LDN);
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (TB) simdgroup_load(bf[j], Bs + (sc + 8 * j) * LDN + kk, LDN, ulong2(0, 0), true);
                else simdgroup_load(bf[j], Bs + kk * LDT + sc + 8 * j, LDT);
            }
            #pragma unroll
            for (int i = 0; i < 2; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(cf[i * 4 + j], af[i], bf[j], cf[i * 4 + j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Each simdgroup writes its 16x32 block 8 rows at a time through its own 8x32 scratch
    // (inside As, free after the last barrier), so device writes are row-contiguous. Every
    // simdgroup runs the same loop, so threadgroup barriers (see attn_fwd32) are safe.
    ushort l = tid % 32;
    threadgroup float *Cs = As + sg * 256;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++)
            simdgroup_store(cf[i * 4 + j], Cs + 8 * j, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            int idx = l + 32 * e;  // row idx / 32 of the 8, column idx % 32
            device float *p = C + (R0 + sr + 8 * i + e) * sCi + C0 + sc + l;
            float v = alpha * Cs[idx];
            if (biasp != 0) v += DPTR(biasp)[C0 + sc + l];
            *p = acc != 0 ? *p + v : v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void gemm_nn(GEMM_PARAMS) {
    threadgroup float sm[2 * TILE];
    threadgroup float *As = sm, *Bs = sm + TILE;
    gemm_body<false, false>(GEMM_ARGS);
}
kernel void gemm_nt(GEMM_PARAMS) {
    threadgroup float sm[2 * TILE];
    threadgroup float *As = sm, *Bs = sm + TILE;
    gemm_body<false, true>(GEMM_ARGS);
}
kernel void gemm_tn(GEMM_PARAMS) {
    threadgroup float sm[2 * TILE];
    threadgroup float *As = sm, *Bs = sm + TILE;
    gemm_body<true, false>(GEMM_ARGS);
}
kernel void gemm_tt(GEMM_PARAMS) {
    threadgroup float sm[2 * TILE];
    threadgroup float *As = sm, *Bs = sm + TILE;
    gemm_body<true, true>(GEMM_ARGS);
}

// ---------------------------------------------------------------------------- attention
// Fused attention for head size 32, with the layouts of models/*/arch/attention.mojo:
//   qkv [B*T, 3D] (head h's Q, K, V at columns h*32, D + h*32, 2D + h*32), att [B*T, D],
//   lse [B*H*T] (log-sum-exp of each query row's scaled scores).
// One 256-thread threadgroup = 64 rows of one (sprite, head), 8 rows per simdgroup; the
// streamed 32-row tiles are staged in threadgroup memory, the next one prefetched into
// registers. Scores go through a per-simdgroup scratch for the softmax arithmetic, and
// accumulators are rescaled by multiplying with a diagonal fragment: reading or writing
// fragment elements directly (thread_elements) is very slow here. Even per-simdgroup
// scratch is fenced with threadgroup barriers: simdgroup_barrier does not order these
// fragment stores and the scalar loads that follow (results came out sporadically wrong).
#define AHD 32
#define ABK 32
#define ALD (AHD + 4)
#define SLD (ABK + 4)
#define NEG_BIG (-3.0e38f)

// Band-sparse attention (the sliding-window part of DeepSeek's NSA): rows t and u interact
// only when their 64-row blocks, counted within a period of FT rows (one frame), are at most
// band apart, so a block sees the same few rows of every frame. FT = T with band >= T / 64 is
// full attention. The allowed rows of a block are [lo, hi) of each period; next_row is the
// first allowed row at or after k (T or more when none is left). The rule is symmetric, so
// the key-side backward walks the same rows.
inline int band_lo(int t, int FT, int band) { return max(0, (t % FT) / 64 - band) * 64; }
inline int band_hi(int t, int FT, int band) { return min(FT, ((t % FT) / 64 + band + 1) * 64); }
inline int next_row(int k, int lo, int hi, int FT) {
    int f = k / FT, r = k - f * FT;
    return r < lo ? f * FT + lo : (r >= hi ? (f + 1) * FT + lo : k);
}

kernel void attn_fwd32(constant ulong &attp [[buffer(0)]], constant ulong &lsep [[buffer(1)]],
                       constant ulong &qkvp [[buffer(2)]], constant int &D [[buffer(3)]],
                       constant int &H [[buffer(4)]], constant float &scale [[buffer(5)]],
                       constant int &T [[buffer(6)]], constant int &FT [[buffer(7)]],
                       constant int &band [[buffer(8)]],
                       uint blk [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float Ks[ABK * ALD], Vs[ABK * ALD], Ss[8 * 8 * SLD], Dg[8 * 64];
    int tiles = T / 64;
    int bh = blk / tiles;
    int t0 = (blk - bh * tiles) * 64;
    int b = bh / H, h = bh - b * H;
    int s3 = 3 * D;
    const device float *base = DPTR(qkvp) + (long)b * T * s3 + h * AHD;
    threadgroup float *S = Ss + sg * 8 * SLD;
    threadgroup float *G = Dg + sg * 64;
    int q0 = t0 + sg * 8;
    // This simdgroup's 8 query rows, scaled, via the scratch; lane: row lane / 4, 8 columns.
    int rr = lane / 4, cc = (lane % 4) * 8;
    #pragma unroll
    for (int e = 0; e < 2; e++)
        *(threadgroup float4 *)(S + rr * SLD + cc + 4 * e) =
            scale * *(const device float4 *)(base + (long)(q0 + rr) * s3 + cc + 4 * e);
    for (int i = lane; i < 64; i += 32) G[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    frag q[4], o[4];
    #pragma unroll
    for (int d = 0; d < 4; d++) { simdgroup_load(q[d], S + d * 8, SLD); o[d] = frag(0.0f); }
    float m = NEG_BIG, l = 0.0f;

    int kr = tid / 8, kc = (tid % 8) * 4;  // this thread's float4 of the K and V tiles
    const device float *kb = base + D + (long)kr * s3 + kc;
    int lo = band_lo(t0, FT, band), hi = band_hi(t0, FT, band);
    int k0 = next_row(0, lo, hi, FT);
    float4 rk = *(const device float4 *)(kb + (long)k0 * s3), rv = *(const device float4 *)(kb + (long)k0 * s3 + D);
    while (k0 < T) {
        *(threadgroup float4 *)(Ks + kr * ALD + kc) = rk;
        *(threadgroup float4 *)(Vs + kr * ALD + kc) = rv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int kn = next_row(k0 + ABK, lo, hi, FT);
        if (kn < T) {
            rk = *(const device float4 *)(kb + (long)kn * s3);
            rv = *(const device float4 *)(kb + (long)kn * s3 + D);
        }
        // S = Q K^T in two 8x16 halves (fewer live fragments), into the scratch.
        #pragma unroll
        for (int hf = 0; hf < 2; hf++) {
            frag s0 = frag(0.0f), s1 = frag(0.0f);
            #pragma unroll
            for (int d = 0; d < 4; d++) {
                frag k;
                simdgroup_load(k, Ks + (hf * 16) * ALD + d * 8, ALD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s0, q[d], k, s0);
                simdgroup_load(k, Ks + (hf * 16 + 8) * ALD + d * 8, ALD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s1, q[d], k, s1);
            }
            simdgroup_store(s0, S + hf * 16, SLD);
            simdgroup_store(s1, S + hf * 16 + 8, SLD);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Online softmax over the tile; the 4 lanes of a row reduce with shuffles.
        threadgroup float *row = S + rr * SLD + cc;
        float4 x0 = *(threadgroup float4 *)row, x1 = *(threadgroup float4 *)(row + 4);
        float mx = max(max(max(x0.x, x0.y), max(x0.z, x0.w)), max(max(x1.x, x1.y), max(x1.z, x1.w)));
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        float mn = max(m, mx);
        x0 = exp(x0 - mn);
        x1 = exp(x1 - mn);
        float ts = (x0.x + x0.y) + (x0.z + x0.w) + (x1.x + x1.y) + (x1.z + x1.w);
        ts += simd_shuffle_xor(ts, 1);
        ts += simd_shuffle_xor(ts, 2);
        float alpha = exp(m - mn);
        l = l * alpha + ts;
        m = mn;
        *(threadgroup float4 *)row = x0;
        *(threadgroup float4 *)(row + 4) = x1;
        if ((lane & 3) == 0) G[rr * 9] = alpha;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        frag g;
        simdgroup_load(g, G, 8);
        #pragma unroll
        for (int d = 0; d < 4; d++) { frag t; simdgroup_multiply(t, g, o[d]); o[d] = t; }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            frag p;
            simdgroup_load(p, S + j * 8, SLD);
            #pragma unroll
            for (int d = 0; d < 4; d++) {
                frag v;
                simdgroup_load(v, Vs + j * 8 * ALD + d * 8, ALD);
                simdgroup_multiply_accumulate(o[d], p, v, o[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // Ks, Vs and the scratch are reused
        k0 = kn;
    }
    if ((lane & 3) == 0) G[rr * 9] = 1.0f / l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    frag g;
    simdgroup_load(g, G, 8);
    #pragma unroll
    for (int d = 0; d < 4; d++) { frag t; simdgroup_multiply(t, g, o[d]); simdgroup_store(t, S + d * 8, SLD); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *out = DPTR(attp) + ((long)b * T + q0 + rr) * D + h * AHD + cc;
    *(device float4 *)out = *(threadgroup float4 *)(S + rr * SLD + cc);
    *(device float4 *)(out + 4) = *(threadgroup float4 *)(S + rr * SLD + cc + 4);
    if ((lane & 3) == 0) DPTR(lsep)[(long)bh * T + q0 + rr] = m + log(l);
}

// Backward, in two kernels like models/*/arch/attention.mojo (dvec = sum_d dO * O per query
// row comes from k_attn_dvec). Both overwrite their columns of dqkv.
#define BLD (16 + 4)
#define ATTN_BWD_PARAMS \\
    constant ulong &dqkvp [[buffer(0)]], constant ulong &qkvp [[buffer(1)]], constant ulong &dattp [[buffer(2)]], \\
    constant ulong &lsep [[buffer(3)]], constant ulong &dvecp [[buffer(4)]], constant int &D [[buffer(5)]], \\
    constant int &H [[buffer(6)]], constant float &scale [[buffer(7)]], constant int &T [[buffer(8)]], \\
    constant int &FT [[buffer(9)]], constant int &band [[buffer(10)]], \\
    uint blk [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], \\
    ushort tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]

// dK_j = scale * sum_t dS_tj Q_t and dV_j = sum_t P_tj dO_t, for 64 key rows j per threadgroup
// (8 per simdgroup), streaming 32-row query tiles. Transposed scores S^T = (scale K) Q^T and
// dP^T = V dO^T are formed 16 queries at a time.
kernel void attn_bwd_kv32(ATTN_BWD_PARAMS) {
    threadgroup float Qs[ABK * ALD], Os[ABK * ALD], Ls[ABK], Dv[ABK], Ss[8 * 8 * BLD], Ps[8 * 8 * BLD];
    int tiles = T / 64;
    int bh = blk / tiles;
    int j0 = (blk - bh * tiles) * 64 + sg * 8;
    int b = bh / H, h = bh - b * H;
    int s3 = 3 * D;
    const device float *base = DPTR(qkvp) + (long)b * T * s3 + h * AHD;
    const device float *dbase = DPTR(dattp) + (long)b * T * D + h * AHD;
    threadgroup float *S = Ss + sg * 8 * BLD;   // S^T, then P^T
    threadgroup float *P = Ps + sg * 8 * BLD;   // dP^T, then dS^T
    int rr = lane / 4, cc = (lane % 4) * 4;     // scalar work: key row rr, 4 query columns
    // This simdgroup's K (scaled) and V rows, through the scratch (8 x 16 at a time).
    frag k[4], v[4], dk[4], dv[4];
    #pragma unroll
    for (int hf = 0; hf < 2; hf++) {
        const device float *kr = base + (long)(j0 + rr) * s3 + D + hf * 16 + cc;
        *(threadgroup float4 *)(S + rr * BLD + cc) = scale * *(const device float4 *)kr;
        *(threadgroup float4 *)(P + rr * BLD + cc) = *(const device float4 *)(kr + D);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int d = 0; d < 2; d++) {
            simdgroup_load(k[hf * 2 + d], S + d * 8, BLD);
            simdgroup_load(v[hf * 2 + d], P + d * 8, BLD);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for (int d = 0; d < 4; d++) { dk[d] = frag(0.0f); dv[d] = frag(0.0f); }

    int qr = tid / 8, qc = (tid % 8) * 4;  // this thread's float4 of the Q and dO tiles
    const device float *qb = base + (long)qr * s3 + qc;
    const device float *ob = dbase + (long)qr * D + qc;
    int lo = band_lo(j0, FT, band), hi = band_hi(j0, FT, band);
    int t0 = next_row(0, lo, hi, FT);
    float4 rq = *(const device float4 *)(qb + (long)t0 * s3), ro = *(const device float4 *)(ob + (long)t0 * D);
    while (t0 < T) {
        *(threadgroup float4 *)(Qs + qr * ALD + qc) = rq;
        *(threadgroup float4 *)(Os + qr * ALD + qc) = ro;
        if (tid < ABK) Ls[tid] = DPTR(lsep)[(long)bh * T + t0 + tid];
        else if (tid < 2 * ABK) Dv[tid - ABK] = DPTR(dvecp)[(long)bh * T + t0 + tid - ABK];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int tn = next_row(t0 + ABK, lo, hi, FT);
        if (tn < T) {
            rq = *(const device float4 *)(qb + (long)tn * s3);
            ro = *(const device float4 *)(ob + (long)tn * D);
        }
        #pragma unroll
        for (int hf = 0; hf < 2; hf++) {
            frag s0 = frag(0.0f), s1 = frag(0.0f), p0 = frag(0.0f), p1 = frag(0.0f);
            #pragma unroll
            for (int d = 0; d < 4; d++) {
                frag x;
                simdgroup_load(x, Qs + (hf * 16) * ALD + d * 8, ALD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s0, k[d], x, s0);
                simdgroup_load(x, Qs + (hf * 16 + 8) * ALD + d * 8, ALD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s1, k[d], x, s1);
                simdgroup_load(x, Os + (hf * 16) * ALD + d * 8, ALD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(p0, v[d], x, p0);
                simdgroup_load(x, Os + (hf * 16 + 8) * ALD + d * 8, ALD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(p1, v[d], x, p1);
            }
            simdgroup_store(s0, S, BLD);
            simdgroup_store(s1, S + 8, BLD);
            simdgroup_store(p0, P, BLD);
            simdgroup_store(p1, P + 8, BLD);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            int c0 = hf * 16 + cc;  // query column within the tile
            float4 st = *(threadgroup float4 *)(S + rr * BLD + cc);
            float4 dp = *(threadgroup float4 *)(P + rr * BLD + cc);
            float4 lq = *(threadgroup float4 *)(Ls + c0);
            float4 dq = *(threadgroup float4 *)(Dv + c0);
            float4 pr = exp(st - lq);
            *(threadgroup float4 *)(S + rr * BLD + cc) = pr;
            *(threadgroup float4 *)(P + rr * BLD + cc) = pr * (dp - dq);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (int c = 0; c < 2; c++) {
                frag pt, dst;
                simdgroup_load(pt, S + c * 8, BLD);
                simdgroup_load(dst, P + c * 8, BLD);
                #pragma unroll
                for (int d = 0; d < 4; d++) {
                    frag x;
                    simdgroup_load(x, Os + (hf * 16 + c * 8) * ALD + d * 8, ALD);
                    simdgroup_multiply_accumulate(dv[d], pt, x, dv[d]);
                    simdgroup_load(x, Qs + (hf * 16 + c * 8) * ALD + d * 8, ALD);
                    simdgroup_multiply_accumulate(dk[d], dst, x, dk[d]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        t0 = tn;
    }
    // dK * scale and dV, 16 columns at a time through the scratch.
    device float *row = DPTR(dqkvp) + ((long)b * T + j0 + rr) * s3 + h * AHD + D;
    #pragma unroll
    for (int hf = 0; hf < 2; hf++) {
        simdgroup_store(dk[hf * 2], S, BLD);
        simdgroup_store(dk[hf * 2 + 1], S + 8, BLD);
        simdgroup_store(dv[hf * 2], P, BLD);
        simdgroup_store(dv[hf * 2 + 1], P + 8, BLD);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        *(device float4 *)(row + hf * 16 + cc) = scale * *(threadgroup float4 *)(S + rr * BLD + cc);
        *(device float4 *)(row + D + hf * 16 + cc) = *(threadgroup float4 *)(P + rr * BLD + cc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// dQ_t = scale * sum_j dS_tj K_j for 64 query rows t per threadgroup (8 per simdgroup),
// streaming 32-row key tiles, 16 keys at a time.
kernel void attn_bwd_q32(ATTN_BWD_PARAMS) {
    threadgroup float Ks[ABK * ALD], Vs[ABK * ALD], Ss[8 * 8 * BLD], Ps[8 * 8 * BLD];
    int tiles = T / 64;
    int bh = blk / tiles;
    int q0 = (blk - bh * tiles) * 64 + sg * 8;
    int b = bh / H, h = bh - b * H;
    int s3 = 3 * D;
    const device float *base = DPTR(qkvp) + (long)b * T * s3 + h * AHD;
    const device float *dbase = DPTR(dattp) + (long)b * T * D + h * AHD;
    threadgroup float *S = Ss + sg * 8 * BLD;
    threadgroup float *P = Ps + sg * 8 * BLD;
    int rr = lane / 4, cc = (lane % 4) * 4;
    frag q[4], o[4], dq[4];
    #pragma unroll
    for (int hf = 0; hf < 2; hf++) {
        *(threadgroup float4 *)(S + rr * BLD + cc) = scale * *(const device float4 *)(base + (long)(q0 + rr) * s3 + hf * 16 + cc);
        *(threadgroup float4 *)(P + rr * BLD + cc) = *(const device float4 *)(dbase + (long)(q0 + rr) * D + hf * 16 + cc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int d = 0; d < 2; d++) {
            simdgroup_load(q[hf * 2 + d], S + d * 8, BLD);
            simdgroup_load(o[hf * 2 + d], P + d * 8, BLD);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for (int d = 0; d < 4; d++) dq[d] = frag(0.0f);
    float lr = DPTR(lsep)[(long)bh * T + q0 + rr];
    float dr = DPTR(dvecp)[(long)bh * T + q0 + rr];

    int kr = tid / 8, kc = (tid % 8) * 4;
    const device float *kb = base + D + (long)kr * s3 + kc;
    int lo = band_lo(q0, FT, band), hi = band_hi(q0, FT, band);
    int k0 = next_row(0, lo, hi, FT);
    float4 rk = *(const device float4 *)(kb + (long)k0 * s3), rv = *(const device float4 *)(kb + (long)k0 * s3 + D);
    while (k0 < T) {
        *(threadgroup float4 *)(Ks + kr * ALD + kc) = rk;
        *(threadgroup float4 *)(Vs + kr * ALD + kc) = rv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int kn = next_row(k0 + ABK, lo, hi, FT);
        if (kn < T) {
            rk = *(const device float4 *)(kb + (long)kn * s3);
            rv = *(const device float4 *)(kb + (long)kn * s3 + D);
        }
        #pragma unroll
        for (int hf = 0; hf < 2; hf++) {
            frag s0 = frag(0.0f), s1 = frag(0.0f), p0 = frag(0.0f), p1 = frag(0.0f);
            #pragma unroll
            for (int d = 0; d < 4; d++) {
                frag x;
                simdgroup_load(x, Ks + (hf * 16) * ALD + d * 8, ALD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s0, q[d], x, s0);
                simdgroup_load(x, Ks + (hf * 16 + 8) * ALD + d * 8, ALD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s1, q[d], x, s1);
                simdgroup_load(x, Vs + (hf * 16) * ALD + d * 8, ALD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(p0, o[d], x, p0);
                simdgroup_load(x, Vs + (hf * 16 + 8) * ALD + d * 8, ALD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(p1, o[d], x, p1);
            }
            simdgroup_store(s0, S, BLD);
            simdgroup_store(s1, S + 8, BLD);
            simdgroup_store(p0, P, BLD);
            simdgroup_store(p1, P + 8, BLD);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float4 st = *(threadgroup float4 *)(S + rr * BLD + cc);
            float4 dp = *(threadgroup float4 *)(P + rr * BLD + cc);
            *(threadgroup float4 *)(P + rr * BLD + cc) = exp(st - lr) * (dp - dr);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (int c = 0; c < 2; c++) {
                frag ds;
                simdgroup_load(ds, P + c * 8, BLD);
                #pragma unroll
                for (int d = 0; d < 4; d++) {
                    frag x;
                    simdgroup_load(x, Ks + (hf * 16 + c * 8) * ALD + d * 8, ALD);
                    simdgroup_multiply_accumulate(dq[d], ds, x, dq[d]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        k0 = kn;
    }
    device float *row = DPTR(dqkvp) + ((long)b * T + q0 + rr) * s3 + h * AHD;
    #pragma unroll
    for (int hf = 0; hf < 2; hf++) {
        simdgroup_store(dq[hf * 2], S, BLD);
        simdgroup_store(dq[hf * 2 + 1], S + 8, BLD);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        *(device float4 *)(row + hf * 16 + cc) = scale * *(threadgroup float4 *)(S + rr * BLD + cc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"""
