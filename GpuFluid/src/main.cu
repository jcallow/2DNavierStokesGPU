/*
 * main.cu
 *
 *  Created on: Nov 29, 2015
 *      Author: john
 */
#include <iostream>
#include "FluidData.h"
#include "projection.cuh"
#include "gpu_anim.cuh"
#include <iostream>

#include "utils.h"

using namespace std;


void generate_frame(uchar4* bitmap, FluidData &fluidData) {
	timestamp_t t0 = get_timestamp();
	fluidData.addInFlows(0.45, 0.2, 0.15, 0.03, 1.0, 0.0, 3.0);

	fluidData.update();

	render<<<fluidData.dlp->blocks, fluidData.dlp->threads>>>(fluidData.dev_density, bitmap);
	timestamp_t t1 = get_timestamp();

	double secs = (t1 - t0) / 1000000.0L;

	cout << "Frame took: " << secs << " seconds" << endl;
}

int main() {

	int width = 256;
	int height = 256;
	float density = 0.1;
	float timestep = 0.005;
	int projectLimit = 500;

	FluidData fluidData(width, height, density, timestep, projectLimit);
	fluidData.addInFlows(0.45, 0.2, 0.15, 0.03, 1.0, 0.0, 3.0);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	fluidData.buildPressureMatrix();
	GPUAnimBitmap bitmap(fluidData);
	cout << "Done with setup" << endl;

	bitmap.anim_and_exit( (void (*)(uchar4*, FluidData&))generate_frame, NULL );


	return 0;
}

