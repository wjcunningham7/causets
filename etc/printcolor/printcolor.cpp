#include "printcolor.h"

void printf_cyan()
{
	printf("\x1b[36m");
}

void printf_red()
{
	printf("\x1b[31m");
}

void printf_std()
{
	printf("\x1b[0m");
}
