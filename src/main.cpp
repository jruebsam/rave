#include "glad/glad.h"

#include <GLFW/glfw3.h>
#include <iostream>

#include "Window.h"
#include "Shader.h"
#include "Canvas.h"

#include "turbo_colormap.cpp"

void update(unsigned char* data, int nx, int ny, double time){
    double x, y;
    int v, idx;

    for (int i=0; i < ny; i++)
    {
        for(int j=0; j < nx; j++)
        {
            idx = (nx*i + j)*3;
            y = -1 + (i + 1)/double(nx);
            x = - 1 +(j + 1)/double(ny);
            v = (int) (sin(10*x)*sin(10*y)*cos(time)*128 + 128);
            data[idx + 0] = turbo_srgb_bytes[v][0];
            data[idx + 1] = turbo_srgb_bytes[v][1];
            data[idx + 2] = turbo_srgb_bytes[v][2];
        }
    }
}

int main()
{
    double deltaTime = 0.0;
    double lastTime = 0.0;

    static const char* vShader = "shaders/shader.vert";
    static const char* fShader = "shaders/shader.frag";

    Window mainWindow = Window(1366, 768);
    mainWindow.Initialise();


    Shader* shader = new Shader();
    shader->CreateFromFiles(vShader, fShader);

    int width = 800, height = 600;
    Canvas* canvas = new Canvas(width, height, 0.05f);

    unsigned char* data = canvas->getBufferHandle();

    double inc = 0;
    while(!mainWindow.getShouldClose()){
        double now = glfwGetTime();
        deltaTime = now - lastTime;
        lastTime = now;
        glfwPollEvents();

        update(data, width, height, inc);

        shader->UseShader();
        canvas->Render();
        mainWindow.swapBuffers();
        inc += 0.01;
    }

    return 0;
}

