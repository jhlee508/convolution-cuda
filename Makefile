TARGET=main
OBJECTS=src/main.o src/util.o src/convolution_cpu.o src/convolution_gpu.o src/convolution_cudnn.o
INCLUDES=-I/usr/local/cuda/include -I./include

CPPFLAGS=-std=c++14 -O3 -Wall -march=native -mavx2 -mfma -fopenmp -mno-avx512f $(INCLUDES)
CUDA_CFLAGS:=$(foreach option, $(CPPFLAGS),-Xcompiler=$(option))

LDFLAGS=-L/usr/local/cuda/lib64
LDLIBS=-lstdc++ -lcudart -lm -lcudnn

CXX=g++
CUX=/usr/local/cuda/bin/nvcc

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CPPFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c -o $@ $^

%.o: %.cu
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)