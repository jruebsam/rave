#include "glad/glad.h"

#include <GLFW/glfw3.h>
#include <iostream>

#include "Window.h"
#include "Shader.h"
#include "Canvas.h"

Window mainWindow;

double deltaTime = 0.0;
double lastTime = 0.0;
static const char* vShader = "shaders/shader.vert";
static const char* fShader = "shaders/shader.frag";

int main(){
    mainWindow = Window(1366, 768);
    mainWindow.Initialise();

    Shader* shader = new Shader();
    shader->CreateFromFiles(vShader, fShader);

    Canvas* canvas = new Canvas();

    while(!mainWindow.getShouldClose()){
        double now = glfwGetTime();
        deltaTime = now - lastTime;
        lastTime = now;
        glfwPollEvents();

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        shader->UseShader();
        canvas->Render();
        mainWindow.swapBuffers();
    }

    return 0;
}

