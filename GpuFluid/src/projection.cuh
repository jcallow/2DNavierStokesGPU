/*
 * projection.cuh
 *
 *  Created on: Dec 7, 2015
 *      Author: john
 */

#ifndef PROJECTION_CUH_
#define PROJECTION_CUH_

#include "GridData.h"
#include "Launch_Parameters.h"

using namespace std;

__global__
void buildRHS(GridData *u, GridData *v, float *r, float delx, int width, int height) {
	float scale = 1.0/delx;
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	int x = id % width;
	int y = id/width;

	if (x < width && y < height) {
		r[id] = -scale*(u->get(x+1,y) - u->get(x,y) + v->get(x, y+1) - v->get(x,y));
	}

}

// Used for testing
__host__
void buildPressureMatrixHost(float *dev_aDiag, float *dev_aPlusX, float *dev_aPlusY, int width, int height, float density, float delx, float timestep) {
	float *aDiag, *aPlusX, *aPlusY;
	aDiag = new float[width*height];
	aPlusX = new float[width*height];
	aPlusY = new float[width*height];

	cudaMemcpy(aDiag, dev_aDiag, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(aPlusX, dev_aPlusX, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(aPlusY, dev_aPlusY, width*height*sizeof(float), cudaMemcpyDeviceToHost);

	float scale = timestep/(density*delx*delx);

	for (int y = 0, idx = 0; y < height; y++) {
		for (int x = 0; x < width; x++, idx++) {
			if (x < width - 1) {
				aDiag [idx    ] +=  scale;
				aDiag [idx + 1] +=  scale;
				aPlusX[idx    ]  = -scale;
			} else
				aPlusX[idx] = 0.0;

			if (y < height - 1) {
				aDiag [idx     ] +=  scale;
				aDiag [idx + width] +=  scale;
				aPlusY[idx     ]  = -scale;
			} else
				aPlusY[idx] = 0.0;
		}
	}

	cudaMemcpy(dev_aDiag, aDiag, width*height*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_aPlusX, aPlusX, width*height*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_aPlusY, aPlusY, width*height*sizeof(float), cudaMemcpyHostToDevice);
}

// no hardware atomic adds for floats, would have to rethink my plans here.
__global__
void buildPressureMatrix(float *aDiag, float *aPlusX, float *aPlusY, int width, int height, float density, float delx, float timestep) {
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	int x = id % width;
	int y = id/width;

	float scale = timestep/(density*delx*delx);

	if(x < width && y < height) {
		if (x < width - 1) {
			aDiag[id] += scale;
			aDiag[id + 1] += scale;
			aPlusX[id] = -scale;
		} else {
			aPlusX[id] = 0;
		}

		if (y < height - 1) {
			aDiag[id] += scale;
			aDiag[id + width] += scale;
			aPlusY[id] = -scale;
		} else {
			aPlusY[id] = 0;
		}
	}
}


__host__
void buildPreconditionerHost(float *dev_aDiag, float *dev_aPlusX, float *dev_aPlusY, float *dev_preconditioner, int width, int height) {
	float *aDiag, *aPlusX, *aPlusY, *preconditioner;
	aDiag = new float[width*height];
	aPlusX = new float[width*height];
	aPlusY = new float[width*height];
	preconditioner = new float[width*height];

	cudaMemcpy(aDiag, dev_aDiag, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(aPlusX, dev_aPlusX, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(aPlusY, dev_aPlusY, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(preconditioner, dev_preconditioner, width*height*sizeof(float), cudaMemcpyDeviceToHost);

	float tau = 0.97;
	float sigma = 0.25;

	for (int y = 0, idx = 0; y < height; y++) {
		for (int x = 0; x < width; x++, idx++) {
			float e = aDiag[idx];

			if (x > 0) {
				float px = aPlusX[idx - 1]*preconditioner[idx - 1];
				float py = aPlusY[idx - 1]*preconditioner[idx - 1];
				e = e - (px*px + tau*px*py);
			}
			if (y > 0) {
				float px = aPlusX[idx - width]*preconditioner[idx - width];
				float py = aPlusY[idx - width]*preconditioner[idx - width];
				e = e - (py*py + tau*px*py);
			}

			if (e < sigma*aDiag[idx])
				e = aDiag[idx];

			preconditioner[idx] = 1.0/sqrt(e);
		}
	}

	cudaMemcpy(dev_aDiag, aDiag, width*height*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_aPlusX, aPlusX, width*height*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_aPlusY, aPlusY, width*height*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_preconditioner, preconditioner, width*height*sizeof(float), cudaMemcpyHostToDevice);
}


__global__
void buildPreconditioner(float *aDiag, float *aPlusX, float *aPlusY, float *preconditioner, int width, int height) {
	float tau = 0.97;
	float sigma = 0.25;
	int id = threadIdx.x;
	int x, y, matrixId;
	int limit = max(width, height);
	for (int i = 0; i < 2*limit; i++) {
		x = i - id;
		y = id;
		matrixId = x + y*width;
		if (x >= 0 && x < width && y >= 0 && y < height) {
			float e = aDiag[matrixId];
			if (x > 0) {
				float px = aPlusX[matrixId - 1]*preconditioner[matrixId-1];
				float py = aPlusY[matrixId-1]*preconditioner[matrixId-1];
				e = e - (px*px + tau*px*py);
			}
			if (y > 0) {
				float px = aPlusX[matrixId - width]*preconditioner[matrixId-width];
				float py = aPlusY[matrixId-width]*preconditioner[matrixId-width];
				e = e - (py*py + tau*px*py);
			}
			if (e < sigma*aDiag[matrixId]) {
				e = aDiag[matrixId];
			}
			preconditioner[matrixId] = 1.0/sqrt(e);
		}
		__syncthreads();
	}
}


__host__
void applyPreconditioner_Host(float *dev_aDiag, float *dev_aPlusX, float *dev_aPlusY, float *dev_preconditioner, float *dev_dst, float *dev_a, int width, int height) {
	float *aDiag, *aPlusX, *aPlusY, *preconditioner, *dst, *a;
	aDiag = new float[width*height];
	aPlusX = new float[width*height];
	aPlusY = new float[width*height];
	preconditioner = new float[width*height];
	dst = new float[width*height];
	a = new float[width*height];

	cudaMemcpy(aDiag, dev_aDiag, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(aPlusX, dev_aPlusX, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(aPlusY, dev_aPlusY, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(preconditioner, dev_preconditioner, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(dst, dev_dst, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(a, dev_a, width*height*sizeof(float), cudaMemcpyDeviceToHost);

    for (int y = 0, idx = 0; y < height; y++) {
         for (int x = 0; x < width; x++, idx++) {
             float t = a[idx];

             if (x > 0)
                 t -= aPlusX[idx -  1]*preconditioner[idx -  1]*dst[idx -  1];
             if (y > 0)
                 t -= aPlusY[idx - width]*preconditioner[idx - width]*dst[idx - width];

             dst[idx] = t*preconditioner[idx];
         }
     }

     for (int y = height - 1, idx = width*height - 1; y >= 0; y--) {
         for (int x = width - 1; x >= 0; x--, idx--) {
             idx = x + y*width;


             float t = dst[idx];

             if (x < width - 1)
                 t -= aPlusX[idx]*preconditioner[idx]*dst[idx +  1];
             if (y < height - 1)
                 t -= aPlusY[idx]*preconditioner[idx]*dst[idx + width];

             dst[idx] = t*preconditioner[idx];
         }
     }

	cudaMemcpy(dev_aDiag, aDiag, width*height*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_aPlusX, aPlusX, width*height*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_aPlusY, aPlusY, width*height*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_preconditioner, preconditioner, width*height*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dst, dst, width*height*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_a, a, width*height*sizeof(float), cudaMemcpyHostToDevice);
}


__global__
void applyPreconditioner_forward(float *aDiag, float *aPlusX, float *aPlusY, float *preconditioner, float *dst, float *a, int width, int height, int row) {
	int id = threadIdx.x;

	int start_x = (row-blockIdx.x)*blockDim.x;
	int start_y = blockIdx.x*blockDim.x;

	if (start_x >=0 && start_x < width && start_y >=0 && start_y < height) {

		int loc_x, loc_y, x,y,matrixId;

		for (int i = 0; i < 2*blockDim.x; i++) {
			loc_x = i -id;
			loc_y = id;
			x = start_x + loc_x;
			y = start_y + loc_y;
			matrixId = x + y*width;

			if (loc_x >= 0 && loc_x < blockDim.x && loc_y >= 0 && loc_y < blockDim.x) {

				float t = a[matrixId];

				if (x > 0) {
					t -= aPlusX[matrixId -1]*preconditioner[matrixId-1]*dst[matrixId-1];
				}
				if (y > 0) {
					t -= aPlusY[matrixId - width]*preconditioner[matrixId-width]*dst[matrixId-width];
				}
				dst[matrixId] = t*preconditioner[matrixId];
			}

			__syncthreads();
		}
	}
}

__global__
void applyPreconditioner_backward(float *aDiag, float *aPlusX, float *aPlusY, float *preconditioner, float *dst, float *a, int width, int height, int row) {
	int id = threadIdx.x;

	int start_x = (row-blockIdx.x)*blockDim.x;
	int start_y = blockIdx.x*blockDim.x;

	if (start_x >=0 && start_x < width && start_y >=0 && start_y < height) {
		int loc_x, loc_y, x,y,matrixId;

		for (int i = 2*blockDim.x-1; i >= 0; i--) {
			loc_x = i - blockDim.x + id;
			loc_y = blockDim.x-1-id;
			x = start_x + loc_x;
			y = start_y + loc_y;
			matrixId = x + y*width;

			if (loc_x >= 0 && loc_x < blockDim.x && loc_y >= 0 && loc_y < blockDim.x) {

				float t = dst[matrixId];

				if (x < width-1) {
					t -= aPlusX[matrixId]*preconditioner[matrixId]*dst[matrixId+1];
				}
				if (y < height-1) {
					t -= aPlusY[matrixId]*preconditioner[matrixId]*dst[matrixId+width];
				}
				dst[matrixId] = t*preconditioner[matrixId];
			}

			__syncthreads();
		}
	}
}


__global__
void applyPreconditioner(float *aDiag, float *aPlusX, float *aPlusY, float *preconditioner, float *dst, float *a, int width, int height) {
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	int x = 0;
	int y = 0;
	int matrixId = 0;

	int limit = max(width, height);


	for (int i = 0; i < 2*limit; i++) {
		x = i - id;
		y = id;
		matrixId = x + y*width;

		if (x >= 0 && x < width && y >= 0 && y < height) {

			float t = a[matrixId];

			if (x > 0) {
				t -= aPlusX[matrixId -1]*preconditioner[matrixId-1]*dst[matrixId-1];
			}
			if (y > 0) {
				t -= aPlusY[matrixId - width]*preconditioner[matrixId-width]*dst[matrixId-width];
			}
			dst[matrixId] = t*preconditioner[matrixId];
		}

		__syncthreads();
	}

	__syncthreads();


	for (int i = 2*limit-1; i >= 0; i--) {
		x = i - limit + id;
		y = limit-1-id;
		matrixId = x + y*width;

		if (x >= 0 && x < width && y >= 0 && y < height) {

			float t = dst[matrixId];

			if (x < width-1) {
				t -= aPlusX[matrixId]*preconditioner[matrixId]*dst[matrixId+1];
			}
			if (y < height-1) {
				t -= aPlusY[matrixId]*preconditioner[matrixId]*dst[matrixId+width];
			}
			dst[matrixId] = t*preconditioner[matrixId];
		}

		__syncthreads();
	}

	__syncthreads();


}

__global__
void applyPreconditioner_f(float *aDiag, float *aPlusX, float *aPlusY, float *preconditioner, float *dst, float *a, int width, int height) {
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	int x = 0;
	int y = 0;
	int matrixId = 0;

	int limit = max(width, height);


	for (int i = 0; i < 2*limit; i++) {
		x = i - id;
		y = id;
		matrixId = x + y*width;

		if (x >= 0 && x < width && y >= 0 && y < height) {

			float t = a[matrixId];

			if (x > 0) {
				t -= aPlusX[matrixId -1]*preconditioner[matrixId-1]*dst[matrixId-1];
			}
			if (y > 0) {
				t -= aPlusY[matrixId - width]*preconditioner[matrixId-width]*dst[matrixId-width];
			}
			dst[matrixId] = t*preconditioner[matrixId];
		}

		__syncthreads();
	}

}

__global__
void applyPreconditioner_b(float *aDiag, float *aPlusX, float *aPlusY, float *preconditioner, float *dst, float *a, int width, int height) {
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	int x = 0;
	int y = 0;
	int matrixId = 0;

	int limit = max(width, height);

	for (int i = 2*limit-1; i >= 0; i--) {
		x = i - limit + id;
		y = limit-1-id;
		matrixId = x + y*width;

		if (x >= 0 && x < width && y >= 0 && y < height) {

			float t = dst[matrixId];

			if (x < width-1) {
				t -= aPlusX[matrixId]*preconditioner[matrixId]*dst[matrixId+1];
			}
			if (y < height-1) {
				t -= aPlusY[matrixId]*preconditioner[matrixId]*dst[matrixId+width];
			}
			dst[matrixId] = t*preconditioner[matrixId];
		}

		__syncthreads();
	}

	__syncthreads();


}

__global__
void dotProduct_Device(float *a, float *b, float *result, int length, int nearest2pow) {
	extern __shared__ float tmp[];

	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int locId = threadIdx.x;

	if (id < length) {
		tmp[locId] = a[id]*b[id];
	} else {
		tmp[locId] = 0;
	}

	__syncthreads();
	// Reduction

	for (int i = nearest2pow/2; i > 0; i /= 2) {
		if (locId < i && locId + i < blockDim.x) {
			tmp[locId] += tmp[locId + i];
		}
		__syncthreads();
	}

	if (locId == 0) {
		result[blockIdx.x] = tmp[0];
	}

}

__host__
int nearest_power_2(int x) {
   if (x < 0)
		return 0;
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return x+1;
}

__host__
float dotProduct(float *a, float *b, float* dev_partial, int length, int blocks, int threads, int nearest2pow) {

	dotProduct_Device<<<blocks, threads, blocks*sizeof(float)>>>(a, b, dev_partial, length, nearest2pow);

	float partial[blocks];

	CUDA_CHECK_RETURN(cudaMemcpy(partial, dev_partial, blocks*sizeof(float), cudaMemcpyDeviceToHost));

	float dotproduct = 0;
	for (int i = 0; i < blocks; i++) {
		dotproduct += partial[i];
	}

	return dotproduct;
}



__global__
void matrixVectorProduct(float *aDiag, float *aPlusX, float *aPlusY, float *dst, float *b, int width, int height) {
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	int x = id % width;
	int y = id/width;

	if (x < width && y < height) {
		float t = aDiag[id]*b[id];

		if (x > 0) {
			t += aPlusX[id-1]*b[id-1];
		}
		if (y > 0) {
			t += aPlusY[id - width]*b[id - width];
		}
		if (x < width - 1) {
			t += aPlusX[id]*b[id + 1];
		}
		if (y < height - 1) {
			t += aPlusY[id]*b[id + width];
		}
		dst[id] = t;
	}
}

__global__
void scaledAdd(float *dst, float *a, float *b, float s, int width, int height) {
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	if (id < width*height) {
		dst[id] = a[id] + b[id]*s;
	}
}



__global__
void infinityNorm_Device(float *a, float *result, int length, int nearest2pow) {
	extern __shared__ float tmp[];

	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int locId = threadIdx.x;

	if (id < length)
		tmp[locId] = abs(a[id]);
	else
		tmp[locId] = 0;

	for (int i = nearest2pow/2; i > 0; i /= 2) {
		if (locId < i && locId + i < blockDim.x) {
			tmp[locId] = max(tmp[locId + i], tmp[locId]);
		}
		__syncthreads();
	}

	if (locId == 0) {
		result[blockIdx.x] = tmp[0];
	}

}

__host__
float infinityNorm(float* a, float *dev_partial, int length, int blocks, int threads, int nearest2pow) {
	infinityNorm_Device<<<blocks, threads, blocks*sizeof(float)>>>(a, dev_partial, length, nearest2pow);

	float partial[blocks];

	CUDA_CHECK_RETURN(cudaMemcpy(partial, dev_partial, blocks*sizeof(float), cudaMemcpyDeviceToHost));

	float infinity_norm = 0;
	for (int i = 0; i < blocks; i++) {
		infinity_norm = max(infinity_norm, partial[i]);
	}

	return infinity_norm;
}

__host__
void applyPressureHost(GridData* dev_u, GridData* dev_v, GridData* u, GridData* v, float *dev_p, float timestep, float density, float delx, int width, int height) {
	float *p, *dev_u_src, *dev_v_src;
	p = new float[width*height];

	CUDA_CHECK_RETURN(cudaMemcpy(p, dev_p, width*height*sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	cudaMalloc((void**)&dev_u_src, sizeof(float*));
	CUDA_CHECK_RETURN(cudaMemcpy(&dev_u_src, &(dev_u->src), sizeof(float*), cudaMemcpyDeviceToHost));

	cudaMalloc((void**)&dev_v_src, sizeof(float*));
	CUDA_CHECK_RETURN(cudaMemcpy(&dev_v_src, &(dev_v->src), sizeof(float*), cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaMemcpy(u->src, dev_u_src, u->width*u->height*sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(v->src, dev_v_src, v->width*v->height*sizeof(float), cudaMemcpyDeviceToHost));

	float scale = timestep/(density*delx);

	for (int y = 0, idx = 0; y < height; y++) {
		for (int x = 0; x < width; x++, idx++) {
			u->get(x,     y    ) -= scale*p[idx];
			u->get(x + 1, y    ) += scale*p[idx];
			v->get(x,     y    ) -= scale*p[idx];
			v->get(x,     y + 1) += scale*p[idx];
		}
	}

	for (int y = 0; y < height; y++)
		u->get(0, y) = u->get(width, y) = 0.0;
	for (int x = 0; x < width; x++)
		v->get(x, 0) = v->get(x, height) = 0.0;



	CUDA_CHECK_RETURN(cudaMemcpy(dev_u_src, u->src, u->width*u->height*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_v_src, v->src, v->width*v->height*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(dev_p, p, width*height*sizeof(float), cudaMemcpyHostToDevice));

}

// doesn't work, unfortunately no atomic add for floats.
__global__
void applyPressure(GridData* u, GridData* v, float *p, float timestep, float density, float delx, int width, int height) {
	float scale = timestep/(density*delx);

	int id = threadIdx.x + blockIdx.x*blockDim.x;

	int x = id % width;
	int y = id/width;

	if (x < width - 1 && x > 0 && y < height - 1 && y > 0) {
		u->get(x+1, y) -= scale*p[id];
		u->get(x,y) += scale*p[id];
		v->get(x,y) -= scale*p[id];
		v->get(x,y+1) += scale*p[id];
	}

	// boundary set to 0
	if (x == 0 && y < height) {
		u->get(x,y) = u->get(width, y) = 0.0;
	}
	if (y == 0 && x < width) {
		v->get(x,y) = u->get(x, height) = 0.0;
	}
}

#endif /* PROJECTION_CUH_ */
