#ifndef STOPWATCH_H_
#define STOPWATCH_H_

#include <cstdlib>
#include <sys/time.h>

void stopwatchStart();
double stopwatchReadSeconds();

struct timeval StartTime;

void stopwatchStart()
{
	gettimeofday(&StartTime, NULL);
}

double stopwatchReadSeconds()
{
	struct timeval endTime;
	gettimeofday(&endTime, 0);
    
	long ds = endTime.tv_sec - StartTime.tv_sec;
	long dus = endTime.tv_usec - StartTime.tv_usec;
	return ds + 0.000001*dus;
}

#endif
