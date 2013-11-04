#include "CausetSubroutines.cuh"

int main(int argc, char **argv)
{
	Network network = Network(parseArgs(argc, argv));
	bool success = false;

	shrQAStart(argc, argv);

	if (network.network_properties.flags.use_gpu)
		connectToGPU(argc, argv);
	
	if (!initializeNetwork(&network)) goto CausetExit;
	//Measure Network Properties Here
	if (network.network_properties.flags.disp_network && !displayNetwork(network.nodes, network.links, argc, argv)) goto CausetExit;
	if (network.network_properties.flags.print_network && !printNetwork(network)) goto CausetExit;
	if (!destroyNetwork(&network)) goto CausetExit;

	if (network.network_properties.flags.use_gpu)
		cuCtxDetach(cuContext);

	success = true;

	CausetExit:
	shrQAFinish(argc, (const char**)argv, success ? QA_PASSED : QA_FAILED);
	printf("PROGRAM COMPLETED\n");
}