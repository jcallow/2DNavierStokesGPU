/*
 * advection.cuh
 *
 *  Created on: Nov 29, 2015
 *      Author: john
 */

#ifndef ADVECTION_CUH_
#define ADVECTION_CUH_

#include "GridData.h"

__device__
float linear_interpolation(float a, float b, float x) {
	return a*(1.0 - x) + b*x;
}

__device__
float linear_interpolation(GridData* data, float x, float y) {
	x = min(max(x - data->offset_x, 0.0), data->width - 1.001);
	y = min(max(y - data->offset_y, 0.0), data->height - 1.001);

	int ix = (int)x;
	int iy = (int)y;

	x -= ix;
	y -= iy;

	float x_00 = data->get(ix, iy), x_10 = data->get(ix+1, iy);
	float x_01 = data->get(ix, iy + 1), x_11 = data->get(ix+1, iy+1);

	return linear_interpolation(linear_interpolation(x_00, x_10, x), linear_interpolation(x_01, x_11, x), y);
}

__device__
void rungeKutta3(GridData* data, float &x, float &y, float timestep, float delx, GridData *u, GridData *v) {
	float u_0 = linear_interpolation(u, x, y)/delx;
	float v_0 = linear_interpolation(v, x, y)/delx;

	float x_mid = x - 0.5*timestep*u_0;
	float y_mid = y - 0.5*timestep*v_0;

	float u_mid = linear_interpolation(u, x_mid, y_mid)/delx;
	float v_mid = linear_interpolation(v, x_mid, y_mid)/delx;

	float x_final = x - 0.75*timestep*u_mid;
	float y_final = y - 0.75*timestep*v_mid;

	float u_final = linear_interpolation(u, x_final, y_final);
	float v_final = linear_interpolation(v, x_final, y_final);

	x -= timestep*((2.0/9.0)*u_0 + (3.0/9.0)*u_mid + (4.0/9.0)*u_final);
	y -= timestep*((2.0/9.0)*v_0 + (3.0/9.0)*v_mid + (4.0/9.0)*v_final);
}

__device__
float cubic_interpolation(float a, float b, float c, float d, float x) {
	float xsq = x*x;
	float xcu = xsq*x;

	float minV = min(a, min(b, min(c, d)));
	float maxV = max(a, max(b, max(c, d)));

	float t =
		a*(0.0 - 0.5*x + 1.0*xsq - 0.5*xcu) +
		b*(1.0 + 0.0*x - 2.5*xsq + 1.5*xcu) +
		c*(0.0 + 0.5*x + 2.0*xsq - 1.5*xcu) +
		d*(0.0 + 0.0*x - 0.5*xsq + 0.5*xcu);

	return min(max(t, minV), maxV);
}

__device__
float cubic_interpolation(GridData* data, float x, float y) {
    x = min(max(x - data->offset_x, 0.0), data->width- 1.001);
    y = min(max(y - data->offset_y, 0.0), data->height - 1.001);
    int ix = (int)x;
    int iy = (int)y;
    x -= ix;
    y -= iy;

    int x0 = max(ix - 1, 0), x1 = ix, x2 = ix + 1, x3 = min(ix + 2, data->width - 1);
    int y0 = max(iy - 1, 0), y1 = iy, y2 = iy + 1, y3 = min(iy + 2, data->height - 1);

    float q0 = cubic_interpolation(data->get(x0, y0), data->get(x1, y0), data->get(x2, y0), data->get(x3, y0), x);
    float q1 = cubic_interpolation(data->get(x0, y1), data->get(x1, y1), data->get(x2, y1), data->get(x3, y1), x);
    float q2 = cubic_interpolation(data->get(x0, y2), data->get(x1, y2), data->get(x2, y2), data->get(x3, y2), x);
    float q3 = cubic_interpolation(data->get(x0, y3), data->get(x1, y3), data->get(x2, y3), data->get(x3, y3), x);

    return cubic_interpolation(q0, q1, q2, q3, y);
}


__global__
void advect(GridData* data, float timestep, float delx, GridData *u, GridData *v) {
	int index = threadIdx.x + blockIdx.x*blockDim.x;

	int x = index % data->width;
	int y = index/data->width;

	if (x < data->width && y < data->height) {

		float ox = x + data->offset_x;
		float oy = y + data->offset_y;

		rungeKutta3(data, ox, oy, timestep, delx, u, v);

		data->dst[index] = cubic_interpolation(data, ox, oy);
	}
}

__global__
void swapAll(GridData* d, GridData* u, GridData* v) {
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	if (id == 0) {
		d->swap();
		u->swap();
		v->swap();
	}
}

__device__
float length(float x, float y) {
	return sqrt(x*x + y*y);
}

__device__
float cubicPulse(float x) {
	x = min(fabs(x), 1.0);
	return 1.0 - x*x*(3.0 - 2.0*x);
}

__global__
void addFlow(GridData *data, float delx, float x0, float y0, float x1, float y1, float v) {

	int ix0 = (int)(x0/delx - data->offset_x);
	int iy0 = (int)(y0/delx - data->offset_y);
	int ix1 = (int)(x1/delx - data->offset_x);
	int iy1 = (int)(y1/delx - data->offset_y);

	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int x = id % data->width;
	int y = id/data->width;

	if (x >= max(ix0, 0) && x < min(ix1, data->height) && y >= max(iy0, 0) && y < min(iy1, data->height)) {
		float l = length(
			(2.0*(x + 0.5)*delx - (x0 + x1))/(x1 - x0),
			(2.0*(y + 0.5)*delx - (y0 + y1))/(y1 - y0)
		);
		float vi = cubicPulse(l)*v;
		if (fabs(data->get(x,y)) < fabs(vi)) {
			data->src[id] = vi;
		}
	}

}

__global__
void test(float fluidData) {
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	printf("%f\n", fluidData);

}



#endif /* ADVECTION_CUH_ */
