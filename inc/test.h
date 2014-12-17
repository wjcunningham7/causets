#ifndef TEST_H_
#define TEST_H_

#include "Causet.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

void test()
{
	float *x = (float*)malloc(sizeof(float)*2);
	float *y = (float*)malloc(sizeof(float)*2);

	printf("Initial Addresses:\n");
	printf("\t%p\n", x);
	printf("\t%p\n", y);

	Coordinates *c = new Coordinates2D();
	//c->x(x);
	//c->y(y);
	c->x() = x;
	c->y() = NULL;
	
	printf("Referenced Addresses:\n");
	printf("\t%p\n", c->x());
	printf("\t%p\n", c->y());

	printf("Reached End of Test.\n");
	exit(EXIT_SUCCESS);
}

#endif
