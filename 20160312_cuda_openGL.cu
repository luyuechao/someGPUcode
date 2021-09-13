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
//

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

const int image_width = 1024;
const int image_height = 1024;

/* Handles OpenGL-CUDA exchange. */
cudaGraphicsResource *cuda_texture;

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

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    cudaGraphicsGLRegisterImage(&cuda_texture, gl_texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
}

void updateMemDevice(cudaArray *memDevice, int w, int h) {
    struct Color {
        unsigned char r, g, b, a;
    } *memHost = new Color[w * h];

    memset(memHost, 128, w * h * 4); // memset(*ptr, int value, size_t num);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            Color &c = memHost[y * w + x];
            c.r = c.b = 255 * x / w;
        }
    }

    cudaMemcpyToArray(memDevice, 0, 0, memHost, w * h * 4, cudaMemcpyHostToDevice);
    delete [] memHost;
}

void editTexture(int w, int h) {
    cudaGraphicsMapResources(1, &cuda_texture);
    cudaArray* memDevice;
    cudaGraphicsSubResourceGetMappedArray(&memDevice, cuda_texture, 0, 0);
    updateMemDevice(memDevice, w, h);
    cudaGraphicsUnmapResources(1, &cuda_texture);
}


void windowResizeFunc(int w, int h) {
    glViewport(-w, -h, w * 2, h * 2);
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

int main(int argc, char *argv[]) {

    /* Initialize OpenGL context. */
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB);
    glutInitWindowSize(400, 300);
    glutCreateWindow("Bitmap in Device Memory");
    glutReshapeFunc(windowResizeFunc);
    glutDisplayFunc(displayFunc);

    glewInit();
    cudaGLSetGLDevice(0);

    int width = 10, height = 5;
    resizeTexture(width, height);
    editTexture(width, height);

    glutMainLoop();
    return 0;
}
