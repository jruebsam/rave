#include "glad/glad.h"
#include <GLFW/glfw3.h>

#include "Window.h"

Window mainWindow;
double deltaTime = 0.0f;
double lastTime = 0.0f;

int main(){
    mainWindow = Window(1366, 768);
    mainWindow.Initialise();

    float vertices[] = {-0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f, 0.0f,  0.5f, 0.0f};
    GLuint VBO;
    glGenBuffers(1, &VBO);


    while(!mainWindow.getShouldClose()){
        double now = glfwGetTime();
        deltaTime = now - lastTime;
        lastTime = now;
        glfwPollEvents();

        mainWindow.swapBuffers();
    }

    return 0;
}

