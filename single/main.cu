#include <iostream>
#include <chrono>
#include <cutf/cublas.hpp>
#include <cutf/memory.hpp>

std::string get_mode_str(const cublasComputeType_t compute_mode) {
	switch(compute_mode) {
	case CUBLAS_COMPUTE_32F_FAST_16F:
		return "FP16";
	case CUBLAS_COMPUTE_32F_FAST_TF32:
		return "TF32";
	default:
		break;
	}
	return "FP32";
}

int main(int argc, char** argv) {
	if (argc < 5) {
		std::fprintf(stderr, "Usage: %s [m] [n] [k] [num_compute] [mode: fp16/tf32/fp32(default)]\n", argv[0]);
		return 1;
	}

	const auto m = std::stoul(argv[1]);
	const auto n = std::stoul(argv[2]);
	const auto k = std::stoul(argv[3]);
	const auto num_compute = std::stoul(argv[4]);
	const std::string mode = argv[5];

	auto mat_a_uptr = cutf::memory::get_device_unique_ptr<float>(m * k);
	auto mat_b_uptr = cutf::memory::get_device_unique_ptr<float>(k * n);
	auto mat_c_uptr = cutf::memory::get_device_unique_ptr<float>(m * n);

	auto cublas_handle_uptr = cutf::cublas::get_cublas_unique_ptr();

	auto compute_mode = CUBLAS_COMPUTE_32F;
	if (mode == "fp16") {
		compute_mode = CUBLAS_COMPUTE_32F_FAST_16F;
	} else if (mode == "tf32") {
		compute_mode = CUBLAS_COMPUTE_32F_FAST_TF32;
	}

	const float alpha = 1.0f, beta = 0.0f;
	CUTF_CHECK_ERROR(cublasGemmEx(
				*cublas_handle_uptr.get(),
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				m, n, k,
				&alpha,
				mat_a_uptr.get(), CUDA_R_32F, m,
				mat_b_uptr.get(), CUDA_R_32F, k,
				&beta,
				mat_c_uptr.get(), CUDA_R_32F, m,
				compute_mode,
				CUBLAS_GEMM_ALGO0
				));
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());

	if (num_compute != 0) {
		const auto start_clock = std::chrono::system_clock::now();
		for (std::uint64_t i = 0; i < num_compute; i++) {
			CUTF_CHECK_ERROR(cublasGemmEx(
						*cublas_handle_uptr.get(),
						CUBLAS_OP_N,
						CUBLAS_OP_N,
						m, n, k,
						&alpha,
						mat_a_uptr.get(), CUDA_R_32F, m,
						mat_b_uptr.get(), CUDA_R_32F, k,
						&beta,
						mat_c_uptr.get(), CUDA_R_32F, m,
						compute_mode,
						CUBLAS_GEMM_ALGO0
						));
		}
		CUTF_CHECK_ERROR(cudaDeviceSynchronize());
		const auto end_clock = std::chrono::system_clock::now();

		const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6;

		std::printf("mode=%s, shape=(%lu,%lu,%lu), throughput=%.3f TFlop/s\n",
				get_mode_str(compute_mode).c_str(),
				m, n, k,
				(2lu * m * n * k * num_compute) / elapsed_time * 1e-12
				);
	}
}
