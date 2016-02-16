/*
 * GridData.h
 *
 *  Created on: Nov 29, 2015
 *      Author: john
 */

#ifndef GRIDDATA_H_
#define GRIDDATA_H_

#include <algorithm>

struct GridData {

	float* src;
	float* dst;

	int width;
	int height;
	float offset_x;
	float offset_y;

	__device__ __host__
	void swap(void) {
		float *temp = src;
		src = dst;
		dst = temp;
	}

	__device__ __host__
	float get(int x, int y) const {
		return src[x + y*width];
	}

	__device__ __host__
	float &get(int x, int y) {
		return src[x + y*width];
	}

	__host__
	GridData(int width, int height, float ox, float oy) : width(width), height(height), offset_x(ox), offset_y(oy) {
		src = new float[width*height];
		dst = new float[width*height];
	}
};

#endif /* GRIDDATA_H_ */
