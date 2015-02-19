# Test 2 for Streams in PyCUDA
# Here we compare kernels with streams and with out streams

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import time

def run(N):
#	print "===Creating kernel==="
	mod = []
	mod.append(SourceModule("""
	__global__ void mul_gpu(double *a, double *b, double *c, double *d, double *e, double *f){
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		for(int i=0;i<100;i++){
		e[tid] = a[tid] + b[tid];
		e[tid] = a[tid] + e[tid];
		e[tid] = a[tid] + e[tid];
		e[tid] = a[tid] + e[tid];
		e[tid] = a[tid] + e[tid];
		f[tid] = c[tid] - d[tid];
		f[tid] = c[tid] - f[tid];
		f[tid] = c[tid] - f[tid];
		f[tid] = c[tid] - f[tid];
		f[tid] = c[tid] - f[tid];
		e[tid] = b[tid] * e[tid];
		e[tid] = a[tid] / e[tid];
		e[tid] = b[tid] * e[tid];
		e[tid] = a[tid] / e[tid];
		e[tid] = a[tid] + e[tid];
		f[tid] = c[tid] * d[tid];
		f[tid] = d[tid] / f[tid];
		f[tid] = c[tid] * f[tid];
		f[tid] = d[tid] / f[tid];
		f[tid] = c[tid] - f[tid];
		}
	}
	"""))

	kernel_name = ["mul_gpu"]
	my_kernel = []
	
	for i in range(len(kernel_name)):
		my_kernel.append(mod[i].get_function(kernel_name[i]))
	
#	print "===Created Kernels==="
	
	# Number of elements = 1024 * 1024

	num_data = 6
	num_streams = 25

#	print "===Creating Data==="
	cpu_data, gpu_data = [], []
	for i in range(num_data):
		cpu_data.append(np.random.randn(N).astype(np.float64))
		gpu_data.append(drv.mem_alloc(cpu_data[i].nbytes))

#	print "===Created Data==="

#	print "===Creating Streams==="
	stream = []
	for i in range(num_streams):
		stream.append(drv.Stream())

#	print "===Created Streams==="

	for k in range(4):
		drv.memcpy_htod(gpu_data[k], cpu_data[k])

#	print "===Running Kernels using streams==="
	
	ran = 0
	start = time.time()
	for j in range(1000):
		for k in range(num_streams):
			my_kernel[0](gpu_data[0],gpu_data[1],gpu_data[2],gpu_data[3],gpu_data[4],gpu_data[5],block=(N, 1,1), grid=(1,1,1))
	end = time.time()
	ran += (end - start)
#	print "===Ran Kernels using streams==="

	drv.memcpy_dtoh(cpu_data[4], gpu_data[4])
	drv.memcpy_dtoh(cpu_data[5], gpu_data[5])

#	print "== Done!! =="

	print "Total run time = ", ran

N = 1000
run(N)
