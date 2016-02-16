/*
 * utils.h
 *
 *  Created on: Dec 9, 2015
 *      Author: john
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <sys/time.h>
using namespace std;

typedef unsigned long long timestamp_t;

static timestamp_t get_timestamp () {
    struct timeval now;
    gettimeofday (&now, NULL);
    return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}



#endif /* UTILS_H_ */
