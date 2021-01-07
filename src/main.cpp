#include "glad/glad.h"

#include <GLFW/glfw3.h>
#include <iostream>

#include "Window.h"
#include "Shader.h"
#include "Canvas.h"

#include "Simulation.h"
#include "State.h"



int main()
{
    double deltaTime = 0.0;
    double lastTime = 0.0;

    static const char* vShader = "shaders/shader.vert";
    static const char* fShader = "shaders/shader.frag";

    Window mainWindow = Window(1200, 1200);
    mainWindow.Initialise();


    Shader* shader = new Shader();
    shader->CreateFromFiles(vShader, fShader);

    int width = 1024, height = 1024;
    Canvas* canvas = new Canvas(width, height, 0.05f);

    GLuint texId = canvas->getTextureID();
    Simulation* sim = new Simulation(texId, width, height);

    for(int i=300; i<600; i++){
        for(int j=300; j<600; j++){
            sim->state.T.host[j + 1024*i] = 1.0f;
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

