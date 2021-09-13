#include <iostream>
#include <fstream>
#include <cuda_profiler_api.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

using namespace std;

const int image_width = 1024;
const int image_height = 1024;
const int frameSize = image_width * image_height;
char *char_buf = new char[frameSize]; //buffer to save the image
/* Handles OpenGL-CUDA exchange. */
cudaGraphicsResource *cuda_texture;
cudaArray* memDevice;

/* Registers (+ resizes) CUDA-Texture (aka Renderbuffer). */
void resizeTexture(int w, int h) {
    static GLuint gl_texture = 0;

    /* Delete old CUDA-Texture. */
    if (gl_texture) {
        cudaGraphicsUnregisterResource(cuda_texture);
        glDeleteTextures(1, &gl_texture);
    }else{
    	glEnable(GL_TEXTURE_2D);
    }

    glGenTextures(1, &gl_texture);
    glBindTexture(GL_TEXTURE_2D, gl_texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, w, h, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);

    cudaGraphicsGLRegisterImage(&cuda_texture, gl_texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
}

void updateMemDevice() {

    cudaMemcpyToArray(memDevice, 0, 0, char_buf,
    		image_width * image_height, cudaMemcpyHostToDevice);
}

void editTexture(int w, int h) {
    cudaGraphicsMapResources(1, &cuda_texture);
    cudaGraphicsSubResourceGetMappedArray(&memDevice, cuda_texture, 0, 0);
    updateMemDevice();
    cudaGraphicsUnmapResources(1, &cuda_texture);
}


void windowResizeFunc(int w, int h) {
    glViewport(-h, -w, w * 2 , h * 2);
	//glViewport(0, 0, w, h);
}

void displayFunc() {
    glBegin(GL_QUADS);
    glTexCoord2i(0, 0); glVertex2i(0, 0);
    glTexCoord2i(1, 0); glVertex2i(1, 0);
    glTexCoord2i(1, 1); glVertex2i(1, 1);
    glTexCoord2i(0, 1); glVertex2i(0, 1);
    glEnd();

    glFlush();
}
void updateFrame(void)
{
    glutPostRedisplay();       //call the registered display callback at its next opportunity
}

void mouse(int button, int state, int x, int y)
{
    switch (button) {
        case GLUT_LEFT_BUTTON:
            if (state == GLUT_DOWN)//when mouse button is down
                glutIdleFunc(updateFrame);
            break;
        case GLUT_RIGHT_BUTTON:
            if (state == GLUT_UP)//when mouse button is up
                glutIdleFunc(NULL);
            break;
        case GLUT_MIDDLE_BUTTON:
            glutPostRedisplay();
        default:
            break;
    }
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key) {
        case 27: //ASCII CODE for Escape
        	checkCudaErrors(cudaFreeArray(memDevice));
            delete[]char_buf;
        	cudaDeviceReset();
            exit(0);
            break;
    }
}

int main(int argc, char *argv[]) {

	ifstream ifs;
	ifs.open("1024byte.dat", ios:: in | ios:: binary);

	if(!ifs.is_open()){cout << "input file error" << endl; return 1;}

	ifs.read(char_buf, frameSize);

    /* Initialize OpenGL context. */
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(image_width, image_height);
    glutCreateWindow("Bitmap in Device Memory");
    glutReshapeFunc(windowResizeFunc);

    glutMouseFunc(mouse);
    glutKeyboardFunc(keyboard);
    glutDisplayFunc(displayFunc);

    glewInit();
    cudaGLSetGLDevice(0);

    int width = 1024, height = 1024;
    resizeTexture(width, height);
    editTexture(width, height);

    glutMainLoop();

    ifs.close();
    return 0;
}
