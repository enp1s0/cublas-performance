NVCC=nvcc
NVCCFLAGS=-std=c++14
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
NVCCFLAGS+=-lcublas
NVCCFLAGS+=-I./mateval/include -L./mateval/build -lmateval_cuda
NVCCFLAGS+=-I./cutf/include

TARGET=cublas.test

$(TARGET):main.cu
	$(NVCC) $< -o $@ $(NVCCFLAGS)
  
clean:
	rm -f $(TARGET)
