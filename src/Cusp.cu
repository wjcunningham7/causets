#include <cusp/copy.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>

#include "Cusp.h"

/////////////////////////////
//(C) Will Cunningham 2015 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

void func1()
{
	cusp::csr_matrix<int, float, cusp::host_memory> csr_host(5, 8, 12);
}
