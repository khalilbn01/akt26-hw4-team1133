# gpu architecture
# Ampere by default, with forward compatibility
sm=80
GENCODE_SM   = -gencode arch=compute_$(sm),code=sm_$(sm)
GENCODE_COMP = -gencode arch=compute_$(sm),code=compute_$(sm)

all:
	nvcc ${GENCODE_SM} ${GENCODE_COMP} hello-world.cu -o hello-world

clean:
	rm -rf hello-world

