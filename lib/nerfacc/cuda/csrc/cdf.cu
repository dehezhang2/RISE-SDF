/*
 * Copyright (c) 2023 Shaofei Wang, ETH Zurich.
 */

#include "include/helpers_cuda.h"
#include <cstdint>

template <typename scalar_t>
__global__ void cdf_resampling_kernel(
    const uint32_t n_rays,
    const int *packed_info,  // input ray & point indices.
    const scalar_t *starts,  // input start t
    const scalar_t *ends,    // input end t
    const scalar_t *weights, // transmittance weights
    const int *resample_packed_info,
    scalar_t *resample_ts,
    scalar_t *resample_offsets,
    int64_t *resample_indices,
    int32_t *resample_fg_counts,
    int32_t *resample_bg_counts)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];  // point idx start.
    const int steps = packed_info[i * 2 + 1]; // point idx shift.
    const int resample_base = resample_packed_info[i * 2 + 0];
    const int resample_steps = resample_packed_info[i * 2 + 1];
    if (steps == 0)
        return;

    starts += base;
    ends += base;
    weights += base;
    resample_fg_counts += base;
    // resample_starts += resample_base;
    // resample_ends += resample_base;
    resample_ts += resample_base;
    resample_offsets += resample_base;
    resample_indices += resample_base;

    // Do not normalize weights, instead, add a new interval at the end
    // with weight = 1.0 - weights_sum
    scalar_t weights_sum = 0.0f;
    for (int j = 0; j < steps; j++)
        weights_sum += weights[j];
    weights_sum += fmaxf(1.0f - weights_sum, 0.0f);

    int num_bins = resample_steps;
    scalar_t cdf_step_size = (1.0f - 1.0 / num_bins) / (resample_steps - 1);

    int idx = 0, j = 0;
    scalar_t cdf_prev = 0.0f, cdf_next = weights[idx] / weights_sum;
    scalar_t cdf_u = 1.0 / (2 * num_bins);
    while (j < num_bins && idx < steps)
    {
        if (cdf_u < cdf_next)
        {
            // printf("cdf_u: %f, cdf_next: %f\n", cdf_u, cdf_next);
            // resample in this interval
            scalar_t scaling = (ends[idx] - starts[idx]) / (cdf_next - cdf_prev);
            scalar_t offset = (cdf_u - cdf_prev) * scaling; 
            scalar_t t = offset + starts[idx];
            // if (j < num_bins - 1)
            //     resample_starts[j] = t;
            // if (j > 0)
            //     resample_ends[j - 1] = t;
            resample_ts[j] = t;
            resample_offsets[j] = offset;
            resample_indices[j] = idx + base;
            // Increasing the count of the interval via atomicAdd
            atomicAdd(&resample_fg_counts[idx], 1);
            // going further to next resample
            cdf_u += cdf_step_size;
            j += 1;
        }
        else if (idx < steps - 1)
        {
            // going to next interval
            idx += 1;
            cdf_prev = cdf_next;
            cdf_next += weights[idx] / weights_sum;
        } else {
            break;
        }
    }
    // If we are out of the loop with j < num_bins, it means we have not sampled
    // enough points. In this case, the remaining points are sampled on the last
    // interval, i.e. the background.
    while (j < num_bins) {
        // no need to resample, just record fixed positions
        scalar_t offset = 10000.f; 
        scalar_t t = offset + ends[steps-1];
        resample_ts[j] = t;
        resample_offsets[j] = offset;
        resample_indices[j] = steps - 1 + base;
        // going further to next resample
        cdf_u += cdf_step_size;
        j += 1;
        // Increasing the count of the interval
        // Note that we do not need to use atomicAdd here, since we parallelize
        // over rays.
        resample_bg_counts[i] += 1;
    }
    if (j != num_bins)
    {
        printf("Error: %d %d %f\n", j, num_bins, weights_sum);
    }
    return;
}

std::vector<torch::Tensor> ray_resampling(
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor weights,
    const int steps)
{
    DEVICE_GUARD(packed_info);

    CHECK_INPUT(packed_info);
    CHECK_INPUT(starts);
    CHECK_INPUT(ends);
    CHECK_INPUT(weights);

    TORCH_CHECK(packed_info.ndimension() == 2 & packed_info.size(1) == 2);
    TORCH_CHECK(starts.ndimension() == 2 & starts.size(1) == 1);
    TORCH_CHECK(ends.ndimension() == 2 & ends.size(1) == 1);
    TORCH_CHECK(weights.ndimension() == 1);

    const uint32_t n_rays = packed_info.size(0);
    const uint32_t n_samples = weights.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    torch::Tensor num_steps = torch::split(packed_info, 1, 1)[1];
    torch::Tensor resample_num_steps = (num_steps > 0).to(num_steps.options()) * steps;
    torch::Tensor resample_cum_steps = resample_num_steps.cumsum(0, torch::kInt32);
    torch::Tensor resample_packed_info = torch::cat(
        {resample_cum_steps - resample_num_steps, resample_num_steps}, 1);

    int total_steps = resample_cum_steps[resample_cum_steps.size(0) - 1].item<int>();
    torch::Tensor resample_ts = torch::empty({total_steps, 1}, starts.options());
    torch::Tensor resample_offsets = torch::empty({total_steps, 1}, starts.options());
    torch::Tensor resample_indices = torch::empty({total_steps}, starts.options().dtype(torch::kInt64));

    int total_samples = num_steps.sum().item<int>();
    torch::Tensor resample_fg_counts = torch::zeros({total_samples}, starts.options().dtype(torch::kInt32));
    torch::Tensor resample_bg_counts = torch::zeros({n_rays}, starts.options().dtype(torch::kInt32));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.scalar_type(),
        "ray_resampling",
        ([&]
         { cdf_resampling_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
               n_rays,
               // inputs
               packed_info.data_ptr<int>(),
               starts.data_ptr<scalar_t>(),
               ends.data_ptr<scalar_t>(),
               weights.data_ptr<scalar_t>(),
               resample_packed_info.data_ptr<int>(),
               // outputs
               resample_ts.data_ptr<scalar_t>(),
               resample_offsets.data_ptr<scalar_t>(),
               resample_indices.data_ptr<int64_t>(),
               resample_fg_counts.data_ptr<int32_t>(),
               resample_bg_counts.data_ptr<int32_t>()); }));

    return {resample_packed_info, resample_ts, resample_offsets, resample_indices, resample_fg_counts, resample_bg_counts};
}
