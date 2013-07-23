#include "causetGL.hpp"
using namespace std;

int main(int argc, char** argv)
{
	parseArgs(argc, argv);
	initData();

	glutInit(&argc, argv);
	//glutInitDisplayMode(GLUT_DOUBLE);
	glutInitWindowSize(900, 900);
	glutInitWindowPosition(50, 50);
	glutCreateWindow("Causets");
	glutDisplayFunc(display);
	//glutReshapeFunc(resize);
	//initGL();
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
			//printf("i: %d\tnode0: %d\tnode1: %d\n", i, con[0][i], con[1][i]);
			i++;
		}
		conFile.close();
	} else
		printf("Error opening file: connections.txt\n");
}

void initGL()
{
	glOrtho(0.0f, 0.01f, 0.0f, 6.3f, -1.0f, 1.0f);
}

void display()
{
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glBegin(GL_LINES);
		glColor3f(0.0f, 0.0f, 0.0f);
		for (unsigned int i = 0; i < K; i++) {
			glVertex2f(sizeFactor * loc[0][con[0][i]] * cos(loc[1][con[0][i]]), sizeFactor * loc[0][con[0][i]] * sin(loc[1][con[0][i]]));
			glVertex2f(sizeFactor * loc[0][con[1][i]] * cos(loc[1][con[1][i]]), sizeFactor * loc[0][con[1][i]] * sin(loc[1][con[1][i]]));
		}
	glEnd();

	glFlush();
	//glutSwapBuffers();
}

void resize(int w, int h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, w, h, 0);
}
