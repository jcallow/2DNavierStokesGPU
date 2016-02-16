/*
 * Launch_Parameters.h
 *
 *  Created on: Nov 29, 2015
 *      Author: john
 */

#ifndef LAUNCH_PARAMETERS_H_
#define LAUNCH_PARAMETERS_H_

struct Launch_Parameters {
  dim3 blocks;
  dim3 threads;
  Launch_Parameters(dim3 b, dim3 t) : blocks(b), threads(t) {}
};

#endif /* LAUNCH_PARAMETERS_H_ */
