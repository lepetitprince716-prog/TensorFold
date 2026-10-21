// A sampled round's candidates on the device (TF_DS_GPU_CANDIDATES): topCandidates' lists without the logits.

// One candidate: a logit as stored (-0.0 stays -0.0) and its column; id 0xffffffff pads, 0xfffffffe flags a NaN row.
struct TfDsCandidate {
    float value;
    unsigned int id;
};

// The k largest logits of each row of logits [rows, n] fp32 (row stride ld), in two launches; k <= 64, 256 threads.
extern "C" __global__ void __launch_bounds__(256) tf_ds_candidates_kernel(const float* logits, const TfDsCandidate* partial,
                                                                           TfDsCandidate* out, unsigned long long ld, int n,
                                                                           int k, int tiles, int merge) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned long long prefix, mask, keys[64];
    __shared__ TfDsCandidate kept[64];
    __shared__ int want, count, bad;
    const int tile = blockIdx.x, row = blockIdx.y, t = threadIdx.x;
    // merge 0: block (tile, row) reads columns [4096 tile, +4096) of its row; merge 1: block (0, row) the tiles' lists
    const int begin = merge ? 0 : tile * 4096;
    const int end = merge ? tiles * k : min(begin + 4096, n);
    if (t == 0) { prefix = mask = 0; want = min(k, end - begin); count = bad = 0; }
    if (t < 64) { keys[t] = 0; kept[t] = {0.0f, 0xffffffffu}; }
    __syncthreads();
    // a NaN has no place in the host's order (its list follows the scan): the row is flagged and the host draws it
    for (int i = begin + t; i < end; i += 256) {
        const TfDsCandidate c = merge ? partial[row * tiles * k + i] : TfDsCandidate{logits[row * ld + i], (unsigned int)i};
        if (c.id == 0xfffffffeu || (c.id < 0xfffffffeu && c.value != c.value)) atomicExch(&bad, 1);
    }
    __syncthreads();
    if (bad) {
        if (t < k) out[(row * (merge ? 1 : tiles) + tile) * k + t] = {0.0f, 0xfffffffeu};
        return;
    }
    // a radix select of the want-th largest key: value descending, a lower column first among equals, -0.0 as +0.0
    for (int digit = 7; digit >= 0; --digit) {
        hist[t] = 0;
        __syncthreads();
        const unsigned long long p = prefix, m = mask;
        for (int i = begin + t; i < end; i += 256) {
            const TfDsCandidate c = merge ? partial[row * tiles * k + i] : TfDsCandidate{logits[row * ld + i], (unsigned int)i};
            const unsigned long long key = c.id == 0xffffffffu ? 0 : tf_ds_score_key(c.value, c.id);
            if ((key & m) == p) atomicAdd(hist + ((key >> (digit * 8)) & 255), 1u);
        }
        __syncthreads();
        if (t == 0) {
            int b = 255;
            while (b > 0 && (int)hist[b] < want) want -= hist[b--];
            prefix |= (unsigned long long)b << (digit * 8);
            mask |= 255ull << (digit * 8);
        }
        __syncthreads();
    }
    // the keys are unique (the column is in them): exactly want of them are at or above the selected one
    for (int i = begin + t; i < end; i += 256) {
        const TfDsCandidate c = merge ? partial[row * tiles * k + i] : TfDsCandidate{logits[row * ld + i], (unsigned int)i};
        if (c.id != 0xffffffffu) {
            const unsigned long long key = tf_ds_score_key(c.value, c.id);
            if (key >= prefix) {
                const int at = atomicAdd(&count, 1);
                if (at < k) { keys[at] = key; kept[at] = c; }
            }
        }
    }
    __syncthreads();
    // a bitonic sort of the 64 slots, largest key first; the padding (key 0) stays below every real key
    for (int size = 2; size <= 64; size *= 2) {
        for (int stride = size / 2; stride; stride /= 2) {
            if (t < 64) {
                const int other = t ^ stride;
                if (other > t && ((keys[t] < keys[other]) == ((t & size) == 0))) {
                    const auto key = keys[t]; keys[t] = keys[other]; keys[other] = key;
                    const auto c = kept[t]; kept[t] = kept[other]; kept[other] = c;
                }
            }
            __syncthreads();
        }
    }
    if (t < k) out[(row * (merge ? 1 : tiles) + tile) * k + t] = kept[t];
}
