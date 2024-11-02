#include <curand_kernel.h>
#include <cmath>

// Normal Distribution
__device__ float sample_normal(curandState* state, float mean, float stddev) {
    return mean + stddev * curand_normal(state);
}

__device__ float log_prob_normal(float x, float mean, float stddev) {
    float variance = stddev * stddev;
    return -0.5 * logf(2 * M_PI * variance) - (x - mean) * (x - mean) / (2 * variance);
}

// Laplace Distribution
__device__ float sample_laplace(curandState* state, float mean, float scale) {
    float u = curand_uniform(state) - 0.5;
    return mean - scale * copysignf(logf(1 - 2 * fabsf(u)), u);
}

__device__ float log_prob_laplace(float x, float mean, float scale) {
    return -logf(2 * scale) - fabsf(x - mean) / scale;
}

// Beta Distribution
__device__ float sample_beta(curandState* state, float alpha, float beta) {
    float a = alpha - 1.0f, b = beta - 1.0f;
    float A = a + b;
    float B = 1.0f / (1.0f + sqrtf(2.0f * A - 1.0f));
    float C = a + (1.0f / B);
    float L = C * logf(C) - C + lgammaf(a + 1.0f) + lgammaf(b + 1.0f) - lgammaf(A + 1.0f);
    float p, u, x, y;
    do {
        u = curand_uniform(state);
        x = B * u;
        y = C * x;
        p = y < 1.0f ? expf(a * logf(x) + b * logf(y) - L) : 0.0f;
    } while (curand_uniform(state) >= p);
    return x / (x + y);
}

__device__ float log_prob_beta(float x, float alpha, float beta) {
    if (x < 0 || x > 1) return -INFINITY;
    return (alpha - 1) * logf(x) + (beta - 1) * logf(1 - x) - lgammaf(alpha) - lgammaf(beta) + lgammaf(alpha + beta);
}

// Uniform Distribution
__device__ float sample_uniform(curandState* state, float lower, float upper) {
    return lower + (upper - lower) * curand_uniform(state);
}

__device__ float log_prob_uniform(float x, float lower, float upper) {
    if (x < lower || x > upper) return -INFINITY;
    return -logf(upper - lower);
}

// Bernoulli Distribution (Discrete)
__device__ int sample_bernoulli(curandState* state, float p) {
    return curand_uniform(state) < p ? 1 : 0;
}

__device__ float log_prob_bernoulli(int x, float p) {
    return x == 1 ? logf(p) : logf(1 - p);
}

// Log-normal Distribution
__device__ float sample_log_normal(curandState* state, float mean, float stddev) {
    return expf(sample_normal(state, mean, stddev));
}

__device__ float log_prob_log_normal(float x, float mean, float stddev) {
    if (x <= 0) return -INFINITY;
    float log_x = logf(x);
    return log_prob_normal(log_x, mean, stddev) - log_x;
}

// Poisson Distribution (Discrete)
__device__ int sample_poisson(curandState* state, float lambda) {
    int k = 0;
    float L = expf(-lambda), p = 1.0;
    do {
        ++k;
        p *= curand_uniform(state);
    } while (p > L);
    return k - 1;
}

__device__ float log_prob_poisson(int k, float lambda) {
    if (k < 0) return -INFINITY;
    return k * logf(lambda) - lambda - lgammaf(k + 1);
}

// Exponential Distribution
__device__ float sample_exponential(curandState* state, float lambda) {
    return -logf(1.0f - curand_uniform(state)) / lambda;
}

__device__ float log_prob_exponential(float x, float lambda) {
    if (x < 0) return -INFINITY;
    return logf(lambda) - lambda * x;
}

// Custom Sampling for Gamma (Marsaglia and Tsang’s method)
__device__ float sample_gamma(curandState* state, float shape, float scale) {
    if (shape < 1.0) {
        float u = curand_uniform(state);
        return sample_gamma(state, shape + 1.0f, scale) * powf(u, 1.0f / shape);
    }
    float d = shape - 1.0f / 3.0f;
    float c = 1.0f / sqrtf(9.0f * d);
    while (true) {
        float x = curand_normal(state);
        float v = 1.0f + c * x;
        if (v > 0) {
            v = v * v * v;
            float u = curand_uniform(state);
            if (u < 1.0f - 0.0331f * (x * x) * (x * x)) return d * v * scale;
            if (logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v))) return d * v * scale;
        }
    }
}

__device__ float log_prob_gamma(float x, float shape, float scale) {
    if (x <= 0) return -INFINITY;
    return (shape - 1) * logf(x) - x / scale - shape * logf(scale) - lgammaf(shape);
}
