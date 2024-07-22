NVCC=nvcc
NVCCFLAGS=-std=c++14
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
NVCCFLAGS+=-lcublas -lcurand
NVCCFLAGS+=-I./mateval/include -L./mateval/build -lmateval_cuda
NVCCFLAGS+=-I./cutf/include

TARGET=cublas.test

$(TARGET):main.cu ./mateval/build/libmateval_cuda.a
	$(NVCC) $< -o $@ $(NVCCFLAGS)

clone_submodule_cutf:cutf/README.md
	git submodule update --init --recursive

clone_submodule_mateval:mateval/README.md
	git submodule update --init --recursive

clone:
	make clone_submodule_mateval
	make clone_submodule_cutf

./mateval/build/libmateval_cuda.a: clone
	mkdir -p ./mateval/build/ && \
	cd ./mateval/build/ && \
	cmake .. && \
	make -j
  
clean:
	rm -f $(TARGET)
