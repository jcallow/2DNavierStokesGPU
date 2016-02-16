/*
 * CudaMacroes.h
 *
 *  Created on: Nov 29, 2015
 *      Author: john
 */

#ifndef CUDAMACROES_H_
#define CUDAMACROES_H_

#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }



#endif /* CUDAMACROES_H_ */
