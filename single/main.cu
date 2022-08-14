#include <iostream>
#include <cutf/cublas.hpp>
#include <cutf/memory.hpp>

int main(int argc, char** argv) {
	if (argc < 5) {
		std::fprintf(stderr, "Usage: %s [m] [n] [k] [mode: fp16/tf32/fp32(default)]\n", argv[0]);
		return 1;
	}

	const auto m = std::stoul(argv[1]);
	const auto n = std::stoul(argv[2]);
	const auto k = std::stoul(argv[3]);
	const std::string mode = argv[4];

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
}
