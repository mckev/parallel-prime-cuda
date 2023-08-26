// Generate prime numbers using GPU
// Algorithm: Sieve of Eratosthenes with parallel sieve
//
// To compile and run:
//   $ nvcc --optimize 3 main.cu -o main
//   $ ./main


#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>


// ------- Generate prime numbers using CPU -------

class SieveCpu {
private:
    uint64_t max;
    uint8_t* sieve_buffer;
    uint64_t sieve_buffer_size;

    bool is_prime(uint64_t num) const {
        uint64_t byte_index = num / sizeof(char);
        uint8_t bit_index = num % sizeof(char);
        return sieve_buffer[byte_index] & (1 << bit_index);
    }

    void mark_as_composite(uint64_t num) {
        uint64_t byte_index = num / sizeof(char);
        uint8_t bit_index = num % sizeof(char);
        sieve_buffer[byte_index] &= ~(1 << bit_index);
    }

public:
    SieveCpu(uint64_t _max) {
        max = _max;
        sieve_buffer_size = max / sizeof(char) + 1;
        sieve_buffer = (uint8_t*) malloc(sieve_buffer_size * sizeof(char));
    }

    ~SieveCpu() {
        free(sieve_buffer);
    }

    void sieve() {
        std::memset(sieve_buffer, (uint8_t) ~0, sieve_buffer_size * sizeof(char));
        mark_as_composite(0);
        mark_as_composite(1);
        uint64_t sqrt_max = std::sqrt(max);
        for (uint64_t i = 0; i <= sqrt_max; i++) {
            if (is_prime(i)) {
                for (uint64_t j = 2 * i; j <= max; j += i) {
                    mark_as_composite(j);
                }
            }
        }
    }

    uint64_t count_primes() const {
        uint64_t result = 0;
        for (uint64_t i = 0; i <= max; i++) {
            if (is_prime(i)) {
                result++;
            }
        }
        return result;
    }

    std::vector<uint64_t> get_primes() const {
        std::vector<uint64_t> result;
        for (uint64_t i = 0; i <= max; i++) {
            if (is_prime(i)) {
                result.push_back(i);
            }
        }
        return result;
    }
};


// ------- Generate prime numbers using GPU (NVIDIA CUDA) -------

__global__ void sieve_kernel(uint64_t max, long long* sieve_buffer, uint64_t sieve_buffer_size, uint64_t* seed_primes, uint64_t seed_primes_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (uint64_t i = index; i <= max; i += stride) {
        if (i < 2) {
            continue;
        }
        // Mark all primes * i as composite, in which i >= 2
        for (uint64_t p = 0; p < seed_primes_size; p++) {
            uint64_t prime = seed_primes[p];
            if (prime * i > max) break;
            uint64_t byte_index = (prime * i) / sizeof(long long);
            uint8_t bit_index = (prime * i) % sizeof(long long);
            atomicAnd(&sieve_buffer[byte_index], ~((long long) 1 << bit_index));
        }
    }
}

class SieveGpu {
private:
    uint64_t max;
    long long* sieve_buffer_host;
    long long* sieve_buffer_device;
    uint64_t sieve_buffer_size;
    uint64_t* seed_primes_host;
    uint64_t* seed_primes_device;
    uint64_t seed_primes_size;

    bool is_prime(uint64_t num) const {
        uint64_t byte_index = num / sizeof(long long);
        uint8_t bit_index = num % sizeof(long long);
        return sieve_buffer_host[byte_index] & (1 << bit_index);
    }

    void mark_as_composite(uint64_t num) {
        uint64_t byte_index = num / sizeof(long long);
        uint8_t bit_index = num % sizeof(long long);
        sieve_buffer_host[byte_index] &= ~(1 << bit_index);
    }

public:
    SieveGpu(uint64_t _max, const std::vector<uint64_t>& _seed_primes) {
        max = _max;
        sieve_buffer_size = max / sizeof(long long) + 1;
        sieve_buffer_host = (long long*) malloc(sieve_buffer_size * sizeof(long long));
        cudaMalloc(&sieve_buffer_device, sieve_buffer_size * sizeof(long long));
        seed_primes_size = _seed_primes.size();
        seed_primes_host = (uint64_t*) malloc(seed_primes_size * sizeof(uint64_t));
        memcpy(seed_primes_host, _seed_primes.data(), seed_primes_size * sizeof(uint64_t));
        cudaMalloc(&seed_primes_device, seed_primes_size * sizeof(uint64_t));
        cudaMemcpy(seed_primes_device, seed_primes_host, seed_primes_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    ~SieveGpu() {
        free(sieve_buffer_host);
        free(seed_primes_host);
        cudaFree(sieve_buffer_device);
        cudaFree(seed_primes_device);
    }

    void sieve() {
        std::memset(sieve_buffer_host, (uint8_t) ~0, sieve_buffer_size * sizeof(long long));
        mark_as_composite(0);
        mark_as_composite(1);
        // From my test, using dedicated device memory is much faster than unified memory, as we are performing intensive memory operations in GPU
        cudaMemcpy(sieve_buffer_device, sieve_buffer_host, sieve_buffer_size * sizeof(long long), cudaMemcpyHostToDevice);
        int block_size = 256;
        int num_blocks = (max + block_size - 1) / block_size;
        sieve_kernel<<<num_blocks, block_size>>>(max, sieve_buffer_device, sieve_buffer_size, seed_primes_device, seed_primes_size);
        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();
        cudaMemcpy(sieve_buffer_host, sieve_buffer_device, sieve_buffer_size * sizeof(long long), cudaMemcpyDeviceToHost);
    }

    uint64_t count_primes() const {
        uint64_t result = 0;
        for (uint64_t i = 0; i <= max; i++) {
            if (is_prime(i)) {
                result++;
            }
        }
        return result;
    }

    std::vector<uint64_t> get_primes() const {
        std::vector<uint64_t> result;
        for (uint64_t i = 0; i <= max; i++) {
            if (is_prime(i)) {
                result.push_back(i);
            }
        }
        return result;
    }
};


int main() {
    uint64_t max = 5000000000;
    uint64_t sqrt_max = std::sqrt(max);

    // Use CPU to calculate all prime numbers up to sqrt(max).
    // This becomes seed primes for generating prime numbers up to max.
    SieveCpu sieve_cpu = SieveCpu(sqrt_max);
    std::cout << "Sieving seed primes in CPU" << std::endl;
    sieve_cpu.sieve();
    std::cout << "Number of seed primes: " << sieve_cpu.count_primes() << std::endl;
    std::vector<uint64_t> seed_primes = sieve_cpu.get_primes();

    // Use GPU to sieve composites
    SieveGpu sieve_gpu = SieveGpu(max, seed_primes);
    std::cout << "Sieving composites in GPU" << std::endl;
    sieve_gpu.sieve();
    std::cout << "Number of primes: " << sieve_gpu.count_primes() << std::endl;

    return 0;
}
