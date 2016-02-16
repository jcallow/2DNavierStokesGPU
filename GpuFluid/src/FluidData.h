/*
 * FluidData.h
 *
 *  Created on: Nov 29, 2015
 *      Author: john
 */

#ifndef FLUIDDATA_H_
#define FLUIDDATA_H_

#include "GridData.h"
#include "CudaMacroes.h"
#include "advection.cuh"
#include "projection.cuh"
#include "Launch_Parameters.h"
#include "utils.h"

using namespace std;



class FluidData {
public:
	GridData* dev_density;
	GridData* dev_velocity_u;
	GridData* dev_velocity_v;

	GridData* h_u;
	GridData* h_v;

	Launch_Parameters* dlp;
	Launch_Parameters* ulp;
	Launch_Parameters* vlp;

	int width;
	int height;

	// dotproduct stuff
	float *dev_partialresults;
	int length;
	int threads;
	int blocks;
	int nearest2power;

	// Fluid constants
	float delx;
	float fluid_density;
	float timestep;

	// PCG
	float *dev_r;
	float *dev_p;
	float *dev_z;
	float *dev_s;
	float *dev_preconditioner;


	float *dev_A_Diagonal;
	float *dev_A_PlusX;
	float *dev_A_PlusY;

	int projectLimit;
	float tolerance;

	FluidData(int width, int height, float fluid_density, float timestep, int projectLimit): width(width), height(height), fluid_density(fluid_density), timestep(timestep), projectLimit(projectLimit) {
		delx = 1.0/min(width, height);
		tolerance = 0.01;

		dev_density = setupGridData(width, height, 0.5, 0.5);
		dlp = setLaunchParameter(width, height);

		h_u = new GridData(width+1, height, 0.0, 0.5);
		dev_velocity_u = setupGridData(width+1, height, 0.0, 0.5);
	    ulp = setLaunchParameter(width+1, height);

	    h_v = new GridData(width, height+1, 0.0, 0.5);
	    dev_velocity_v = setupGridData(width, height+1, 0.5, 0.0);
	    vlp = setLaunchParameter(width, height + 1);

	    CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_r, width*height*sizeof(float)));
	    CUDA_CHECK_RETURN(cudaMemset(dev_r, 0, width*height*sizeof(float)));

	    CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_z, width*height*sizeof(float)));
	    CUDA_CHECK_RETURN(cudaMemset(dev_z, 0, width*height*sizeof(float)));

	    CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_p, width*height*sizeof(float)));
	    CUDA_CHECK_RETURN(cudaMemset(dev_p, 0, width*height*sizeof(float)));

	    CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_s, width*height*sizeof(float)));
	    CUDA_CHECK_RETURN(cudaMemset(dev_s, 0, width*height*sizeof(float)));


	    CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_preconditioner, width*height*sizeof(float)));
	    CUDA_CHECK_RETURN(cudaMemset(dev_preconditioner, 0, width*height*sizeof(float)));

	    CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_A_Diagonal, width*height*sizeof(float)));
	    CUDA_CHECK_RETURN(cudaMemset(dev_A_Diagonal, 0, width*height*sizeof(float)));

	    CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_A_PlusX, width*height*sizeof(float)));
	    CUDA_CHECK_RETURN(cudaMemset(dev_A_PlusX, 0, width*height*sizeof(float)));

	    CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_A_PlusY, width*height*sizeof(float)));
	    CUDA_CHECK_RETURN(cudaMemset(dev_A_PlusY, 0, width*height*sizeof(float)));

	    // setup for dotproduct
	    length = width*height;
	    threads = 256;
	    blocks = ceil(length/(float)threads);

	    nearest2power = nearest_power_2(threads);

	    CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_partialresults, blocks*sizeof(float)));
	    CUDA_CHECK_RETURN(cudaMemset(dev_partialresults, 0, blocks*sizeof(float)));


	    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	}

	~FluidData() {
		cudaFree(dev_r);
		cudaFree(dev_z);
		cudaFree(dev_s);
		cudaFree(dev_p);

		cudaFree(dev_preconditioner);
		cudaFree(dev_A_Diagonal);
		cudaFree(dev_A_PlusX);
		cudaFree(dev_A_PlusY);
		cudaFree(dev_partialresults);
	}

	void addInFlows(float x, float y, float w, float h, float d, float u, float v) {
		addFlow<<<ulp->blocks, ulp->threads>>>(dev_velocity_u, delx, x, y, x+w, y+h, u);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		addFlow<<<vlp->blocks, vlp->threads>>>(dev_velocity_v, delx, x, y, x+w, y+h, v);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		addFlow<<<dlp->blocks, dlp->threads>>>(dev_density, delx, x, y, x+w, y+h, d);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	}

	void project() {
		CUDA_CHECK_RETURN(cudaMemset(dev_p, 0, width*height*sizeof(float)));
		int threads_pre = 256;

		//applyPreconditioner_Host(dev_A_Diagonal, dev_A_PlusX, dev_A_PlusY, dev_preconditioner, dev_z, dev_r, width, height);
		//applyPreconditioner<<<1, max(width, height)>>>(dev_A_Diagonal, dev_A_PlusX, dev_A_PlusY, dev_preconditioner, dev_z, dev_r, width, height);

		for (int row = 0; row < 2*(width/threads) - 1; row++) {
			applyPreconditioner_forward<<<width/threads, threads>>>(dev_A_Diagonal, dev_A_PlusX, dev_A_PlusY, dev_preconditioner, dev_z, dev_r, width, height, row);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}

		for (int row = 2*(width/threads) -1; row >= 0; row--) {
			applyPreconditioner_backward<<<width/threads, threads>>>(dev_A_Diagonal, dev_A_PlusX, dev_A_PlusY, dev_preconditioner, dev_z, dev_r, width, height, row);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}

		CUDA_CHECK_RETURN(cudaMemcpy(dev_s, dev_z, width*height*sizeof(float), cudaMemcpyDeviceToDevice));


		float maxerror = infinityNorm(dev_r, dev_partialresults, width*height, blocks, threads, nearest2power);

		if (maxerror < tolerance) {
			return;
		}

		float sigma = dotProduct(dev_z, dev_r, dev_partialresults, width*height, blocks, threads,  nearest2power);

		for (int i = 0; i < projectLimit; i++) {
			matrixVectorProduct<<<dlp->blocks, dlp->threads>>>(dev_A_Diagonal, dev_A_PlusX, dev_A_PlusY, dev_z, dev_s, width, height);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			float alpha = sigma/dotProduct(dev_z, dev_s, dev_partialresults, width*height, blocks, threads, nearest2power);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			scaledAdd<<<dlp->blocks, dlp->threads>>>(dev_p, dev_p, dev_s, alpha, width, height);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			scaledAdd<<<dlp->blocks, dlp->threads>>>(dev_r, dev_r, dev_z, -alpha, width, height);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());

			maxerror = infinityNorm(dev_r, dev_partialresults, width*height, blocks, threads, nearest2power);
		//	cout << maxerror << endl;
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			if (maxerror < tolerance) {
				printf("Exiting solver after %d iterations, maximum error is %f\n", i, maxerror);
				return;
			}


			timestamp_t t0 = get_timestamp();

			//applyPreconditioner_Host(dev_A_Diagonal, dev_A_PlusX, dev_A_PlusY, dev_preconditioner, dev_z, dev_r, width, height);

		//	applyPreconditioner<<<1, max(width, height)>>>(dev_A_Diagonal, dev_A_PlusX, dev_A_PlusY, dev_preconditioner, dev_z, dev_r, width, height);


			for (int row = 0; row < 2*(width/threads_pre) - 1; row++) {
				applyPreconditioner_forward<<<width/threads_pre, threads_pre>>>(dev_A_Diagonal, dev_A_PlusX, dev_A_PlusY, dev_preconditioner, dev_z, dev_r, width, height, row);
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			}

			for (int row = 2*(width/threads_pre) -1; row >= 0; row--) {
				applyPreconditioner_backward<<<width/threads_pre, threads_pre>>>(dev_A_Diagonal, dev_A_PlusX, dev_A_PlusY, dev_preconditioner, dev_z, dev_r, width, height, row);
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			}


			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		//	timestamp_t t1 = get_timestamp();
		//	double secs = (t1 - t0) / 1000000.0L;

		//	cout << "Apply Preconditioner took: " << secs << " seconds" << endl;

			float sigmaNext = dotProduct(dev_z, dev_r, dev_partialresults, width*height, blocks, threads, nearest2power);


			scaledAdd<<<dlp->blocks, dlp->threads>>>(dev_s, dev_z, dev_s, sigmaNext/sigma, width, height);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			sigma = sigmaNext;

		}
		printf("Exceeded budget of %d iterations, maximum error was %f\n", projectLimit, maxerror);
	}

	void buildPressureMatrix() {
		CUDA_CHECK_RETURN(cudaMemset(dev_A_Diagonal, 0, width*height*sizeof(float)));
		//buildPressureMatrix<<<dlp->blocks, dlp->threads>>>(dev_A_Diagonal, dev_A_PlusX, dev_A_PlusY, width, height, fluid_density, delx, timestep);
		buildPressureMatrixHost(dev_A_Diagonal, dev_A_PlusX, dev_A_PlusY, width, height, fluid_density, delx, timestep);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	}


	void update() {

		// Enforce incompressibility
		buildRHS<<<dlp->blocks, dlp->threads>>>(dev_velocity_u, dev_velocity_v, dev_r, delx, width, height);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		buildPreconditioner<<<1, max(width, height)>>>(dev_A_Diagonal, dev_A_PlusX, dev_A_PlusY, dev_preconditioner, width, height);
		//buildPreconditionerHost(dev_A_Diagonal, dev_A_PlusX, dev_A_PlusY, dev_preconditioner, width, height);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		project();

		applyPressureHost(dev_velocity_u, dev_velocity_v, h_u, h_v, dev_p, timestep, fluid_density, delx, width, height);
		//applyPressure<<<dlp->blocks, dlp->threads>>>(dev_velocity_u, dev_velocity_v, dev_p, timestep, fluid_density, delx, width, height);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());




		// Advect
		advect<<<ulp->blocks, ulp->threads>>>(dev_velocity_u, timestep, delx, dev_velocity_u, dev_velocity_v);
		advect<<<vlp->blocks, vlp->threads>>>(dev_velocity_v, timestep, delx, dev_velocity_u, dev_velocity_v);
		advect<<<dlp->blocks, dlp->threads>>>(dev_density, timestep, delx, dev_velocity_u, dev_velocity_v);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		swapAll<<<1,1>>>(dev_density, dev_velocity_u, dev_velocity_v);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	}

private:
	GridData* setupGridData(int width, int height, float offset_x, float offset_y) {

		GridData* data;
		CUDA_CHECK_RETURN(cudaMalloc((void**)&data, sizeof(GridData)));

		float *dev_src;
		CUDA_CHECK_RETURN(cudaMalloc((void**)&(dev_src), width*height*sizeof(float)));
		CUDA_CHECK_RETURN(cudaMemset(dev_src, 0, width*height*sizeof(float)));
		CUDA_CHECK_RETURN(cudaMemcpy(&(data->src), &dev_src, sizeof(float*), cudaMemcpyHostToDevice));

	    float *dev_dst;
	    CUDA_CHECK_RETURN(cudaMalloc((void**)&(dev_dst), width*height*sizeof(float)));
	    CUDA_CHECK_RETURN(cudaMemset(dev_dst, 0, width*height*sizeof(float)));
	    CUDA_CHECK_RETURN(cudaMemcpy(&(data->dst), &dev_dst, sizeof(float*), cudaMemcpyHostToDevice));

	    // constants
	    CUDA_CHECK_RETURN(cudaMemcpy(&(data->width), &width, sizeof(int), cudaMemcpyHostToDevice));
	    CUDA_CHECK_RETURN(cudaMemcpy(&(data->height), &height, sizeof(int), cudaMemcpyHostToDevice));
	    CUDA_CHECK_RETURN(cudaMemcpy(&(data->offset_x), &offset_x, sizeof(float), cudaMemcpyHostToDevice));
	    CUDA_CHECK_RETURN(cudaMemcpy(&(data->offset_y), &offset_y, sizeof(float), cudaMemcpyHostToDevice));

	    return data;
	}

	Launch_Parameters* setLaunchParameter(int width, int height) {
		float total_threads = width*height;
		dim3 blocks(ceil(total_threads/256.0));
		dim3 threads(256.0);
		Launch_Parameters* lp = new Launch_Parameters(blocks, threads);
		return lp;
	}
};



#endif /* FLUIDDATA_H_ */
