#ifndef STOPWATCH_H_
#define STOPWATCH_H_

#include <cstdlib>
#include <sys/time.h>

struct Stopwatch {
	Stopwatch() : startTime((struct timeval){0,0}), stopTime((struct timeval){0,0}), elapsedTime(0.0) {}

	struct timeval startTime;
	struct timeval stopTime;
	double elapsedTime;
};

void stopwatchStart(struct Stopwatch *sw);
void stopwatchStop(struct Stopwatch *sw);
void stopwatchReset(struct Stopwatch *sw);

void stopwatchStart(struct Stopwatch *sw)
{
	gettimeofday(&sw->startTime, NULL);
}

void stopwatchStop(struct Stopwatch *sw)
{
	assert (sw->startTime.tv_sec != 0 && sw->startTime.tv_usec != 0);
	gettimeofday(&sw->stopTime, NULL);
	long ds = sw->stopTime.tv_sec - sw->startTime.tv_sec;
	long dus = sw->stopTime.tv_usec - sw->startTime.tv_usec;
	sw->elapsedTime = ds + 0.000001 * dus;
}

void stopwatchReset(struct Stopwatch *sw)
{
	sw->startTime = (struct timeval){0,0};
	sw->stopTime = (struct timeval){0,0};
	sw->elapsedTime = 0.0;
}

#endif
