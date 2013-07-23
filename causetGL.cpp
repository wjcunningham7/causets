#include "causetGL.hpp"

int main(int argc, char** argv)
{
	parseArgs(argc, argv);
	initData();

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE);
	glutCreateWindow("Causets");
	glutInitWindowSize(720, 720);
	glutInitWindowPosition(50, 50);
	glutDisplayFunc(display);
	glutMainLoop();

	return 0;
}

void initData()
{
	string line;
	ifstream locFile ("locations.txt");

	loc = new float*[2];
	loc[0] = new float[N];	//Eta   values
	loc[1] = new float[N];	//Theta values

	if (locFile.is_open()) {
		unsigned int i = 0;
		while (locFile.good()) {
			string eta, theta;
			getline(locFile, line);
			istringstream liness(line);
			getline(liness, eta, ' ');
			getline(liness, theta, ' ');
			loc[0][i] = atof(eta.c_str());
			loc[1][i] = atof(theta.c_str());
			//printf("eta: %f\ttheta: %f\n", loc[0][i], loc[1][i]);
			i++;
		}
		locFile.close();
	} else
		printf("Error opening file: locations.txt\n");

	ifstream conFile ("connections.txt");

	con = new unsigned int*[2];
	con[0] = new unsigned int[K];
	con[1] = new unsigned int[K];

	if (conFile.is_open()) {
		unsigned int i = 0;
		while (conFile.good()) {
			string node0, node1;
			getline(conFile, line);
			istringstream liness(line);
			getline(liness, node0, ' ');
			getline(liness, node1, ' ');
			con[0][i] = atoi(node0.c_str());
			con[1][i] = atoi(node1.c_str());
			//printf("node0: %d\tnode1: %d\n", con[0][i], con[1][i]);
			i++;
		}
		conFile.close();
	} else
		printf("Error opening file: connections.txt\n");
}

void display()
{
	glClear(GL_COLOR_BUFFER_BIT);

	glBegin(GL_LINES);
		for (unsigned int i = 0; i < K; i++)
			glVertex2f(loc[0]
	glEnd();

	//glFlush();
	glutSwapBuffers();
}
