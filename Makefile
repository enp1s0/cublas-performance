NVCC=nvcc
NVCCFLAGS=-std=c++14
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
NVCCFLAGS+=-lcublas -lcurand
NVCCFLAGS+=-I./mateval/include -L./mateval/build -lmateval_cuda
NVCCFLAGS+=-I./cutf/include

TARGET=cublas.test

$(TARGET):main.cu ./mateval/build/libmateval_cuda.a
	$(NVCC) $< -o $@ $(NVCCFLAGS)

./mateval/build/libmateval_cuda.a:
	mkdir -p ./mateval/build/ && \
	cd ./mateval/build/ && \
	cmake .. && \
	make -j
  
clean:
	rm -f $(TARGET)
