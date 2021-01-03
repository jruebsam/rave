#include "glad/glad.h"

#include <GLFW/glfw3.h>
#include <iostream>

#include "Window.h"
#include "Shader.h"
#include "Canvas.h"
#include "Simulation.h"

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

    GLuint texId = canvas->getTextureID();
    Simulation* sim = new Simulation(texId, width, height);


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

