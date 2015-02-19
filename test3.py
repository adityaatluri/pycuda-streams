# Test 3 for NO - Streams in PyCUDA

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import time

print "===Creating kernel==="
mod = []
mod.append(SourceModule("""
__global__ void mul_gpu(double *a, double *b, double *c){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	c[tid] = a[tid] * b[tid];
}
"""))

mod.append(SourceModule("""
__global__ void div_gpu(double *a, double *b, double *c){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	c[tid] = a[tid] / b[tid];
}
"""))

kernel_name = ["mul_gpu", "div_gpu"]
my_kernel = []

for i in range(len(kernel_name)):
	my_kernel.append(mod[i].get_function(kernel_name[i]))

print "===Created Kernels==="

# Number of elements = 1024 * 1024

N = 1024 * 1024
num_data = 3
num_streams = 100

print "===Creating Data==="
cpu_data, gpu_data = [], []
for i in range(num_data):
	cpu_data.append(np.random.randn(N).astype(np.float64))
	gpu_data.append(drv.mem_alloc(cpu_data[i].nbytes))

print "===Created Data==="

for k in range(2):
	drv.memcpy_htod(gpu_data[k], cpu_data[k])


print "===Running Kernels without using streams==="

ran = 0

for j in range(100):
	start = time.time()
	for k in range(num_streams):
		my_kernel[0](gpu_data[0], gpu_data[1], gpu_data[2],block=(N/1024, 1,1), grid=(1024,1,1))
		my_kernel[1](gpu_data[0], gpu_data[1], gpu_data[2],block=(N/1024, 1,1), grid=(1024,1,1))
	end = time.time()
	ran += (end - start)
print "===Ran Kernels without using streams==="

drv.memcpy_dtoh(cpu_data[2], gpu_data[2])

print "== Done!! =="

print "Total run time = ", ran
