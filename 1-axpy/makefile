# compile for Volta GPU
CUDADIR ?= /usr/local/cuda

# gpu architecture
# Ampere by default, with forward compatibility
sm=80
GENCODE_SM   = -gencode arch=compute_$(sm),code=sm_$(sm)
GENCODE_COMP = -gencode arch=compute_$(sm),code=compute_$(sm)

LIB=-L${CUDADIR}/lib64 -lcublas

all:
	nvcc ${GENCODE_SM} ${GENCODE_COMP} axpy.cu -o axpy ${LIB}

clean:
	rm -rf axpy

