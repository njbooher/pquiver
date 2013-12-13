LIB = libbqcuda.so

all: default

default:
	nvcc -arch=sm_21 --compiler-options -fno-strict-aliasing -I. -I/opt/cuda/include -c -Xcompiler -fPIC -shared bqcuda.cu

registerinfo:
	nvcc -G -Xptxas="-v" -arch=sm_21 --compiler-options -fno-strict-aliasing -I. -I/opt/cuda/include -c -Xcompiler -fPIC -shared bqcuda.cu

clean:
	rm -f *.o *~ $(LIB)
