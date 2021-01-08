#include "glad/glad.h"

#include <GLFW/glfw3.h>
#include <iostream>
#include <stdlib.h>     



#include "Window.h"
#include "Shader.h"
#include "Canvas.h"

#include "Simulation.h"
#include "State.h"



int main()
{
    double deltaTime = 0.0;
    double lastTime = 0.0;
    srand(42);

    static const char* vShader = "shaders/shader.vert";
    static const char* fShader = "shaders/shader.frag";

    Window mainWindow = Window(1200, 1200);
    mainWindow.Initialise();


    Shader* shader = new Shader();
    shader->CreateFromFiles(vShader, fShader);

    int nx = 512, ny = 512;
    Canvas* canvas = new Canvas(nx, ny, 0.05f);

    GLuint texId = canvas->getTextureID();
    Simulation* sim = new Simulation(texId, nx, ny);

    for(int i=100; i< nx - 100; i++){
        for(int j=100; j< ny - 100; j++){
            float noise = (rand() % 100)/100. - 0.5;
            sim->state.T.host[j + nx*i] += noise + (j/nx- 0.5);
        }
    }
    sim->state.toDevice();


    while(!mainWindow.getShouldClose()){
        double now = glfwGetTime();
        deltaTime = now - lastTime;
        lastTime = now;
        glfwPollEvents();

        sim->Step();
        shader->UseShader();
        canvas->Render();

        mainWindow.swapBuffers();
    }

    return 0;
}

