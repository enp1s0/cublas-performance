#include <iostream>
#include <chrono>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include <cutf/curand.hpp>
#include <mateval/comparison_cuda.hpp>

constexpr unsigned min_log_N = 5;
constexpr unsigned max_log_N = 14;
constexpr unsigned log_N_interval = 1;

constexpr unsigned num_tests = 1u << 5;

int main(int argc, char** argv) {

	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();

	if (argc >= 2 && std::string(argv[1]) == "tf32") {
		CUTF_HANDLE_ERROR(cublasSetMathMode(*cublas_handle.get(), CUBLAS_TF32_TENSOR_OP_MATH));
	}
	if (argc >= 2 && std::string(argv[1]) == "fp16") {
		CUTF_HANDLE_ERROR(cublasSetMathMode(*cublas_handle.get(), CUBLAS_TENSOR_OP_MATH));
	}

	auto mat_a = cutf::memory::get_device_unique_ptr<float>(1u << (2 * max_log_N));
	auto mat_b = cutf::memory::get_device_unique_ptr<float>(1u << (2 * max_log_N));
	auto mat_c = cutf::memory::get_device_unique_ptr<float>(1u << (2 * max_log_N));

	auto cugen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_PHILOX4_32_10);
	CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*cugen.get(), 10));
	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*cugen.get(), mat_a.get(), 1u << (2 * max_log_N)));
	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*cugen.get(), mat_b.get(), 1u << (2 * max_log_N)));

	std::printf("m,n,k,residual,troughput_in_tflops\n");
	for (unsigned log_m = min_log_N; log_m <= max_log_N; log_m += log_N_interval) {
		for (unsigned log_n = min_log_N; log_n <= max_log_N; log_n += log_N_interval) {
			for (unsigned log_k = min_log_N; log_k <= max_log_N; log_k += log_N_interval) {
				const auto m = 1lu << log_m;
				const auto n = 1lu << log_n;
				const auto k = 1lu << log_k;

				float alpha = 1.0f, beta = 0.0f;

				// accuracy
				CUTF_CHECK_ERROR(cutf::cublas::gemm(
							*cublas_handle.get(),
							CUBLAS_OP_N, CUBLAS_OP_N,
							m, n, k,
							&alpha,
							mat_a.get(), m,
							mat_b.get(), k,
							&beta,
							mat_c.get(), m
							));

				const auto error = mtk::mateval::cuda::get_error_AxB(
						mtk::mateval::relative_residual,
						m, n, k,
						mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major,
						mat_a.get(), m,
						mat_b.get(), k,
						mat_c.get(), m
						);

				// throughput
				CUTF_CHECK_ERROR(cudaDeviceSynchronize());
				const auto start_clock = std::chrono::system_clock::now();
				for (unsigned t = 0; t < num_tests; t++) {
					CUTF_CHECK_ERROR(cutf::cublas::gemm(
								*cublas_handle.get(),
								CUBLAS_OP_N, CUBLAS_OP_N,
								m, n, k,
								&alpha,
								mat_a.get(), m,
								mat_b.get(), k,
								&beta,
								mat_c.get(), m
								));
				}
				CUTF_CHECK_ERROR(cudaDeviceSynchronize());
				const auto end_clock = std::chrono::system_clock::now();

				const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6;
				const auto tflops = 2 * m * n * k / 1e12 / duration / num_tests;

				std::printf("%lu,%lu,%lu,%e,%e\n",
						m, n, k,
						error.at(mtk::mateval::relative_residual),
						tflops);
			}
		}
	}
}
