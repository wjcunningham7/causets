#include "causetGL.hpp"

int main(int argc, char** argv)
{
	parseArgs(argc, argv);

	loc = new float*[2];
	loc[0] = new float[N];	//eta   values
	loc[1] = new float[N];	//theta values

	con = new unsigned int*[2];
	con[0] = new unsigned int[K];
	con[1] = new unsigned int[K];

	glutInit(&argc, argv);
	glutCreateWindow("Causets");
	glutInitWindowSize(720, 720);
	glutInitWindowPosition(50, 50);
	glutDisplayFunc(display);
	glutMainLoop();
}

void display()
{
	//
}
