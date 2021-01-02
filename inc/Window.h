#pragma once

#include "glad/glad.h"
#include <GLFW/glfw3.h>

class Window
{
private:
    GLFWwindow *mainWindow;

    GLint width, height, bufferWidth, bufferHeight;

    bool keys[1024];

    GLfloat lastX;
    GLfloat lastY;
    GLfloat xChange = 0.0f;
    GLfloat yChange = 0.0f;
    bool mousedFirstMoved;

    void createCallbacks();
    static void handleKeys(GLFWwindow* window, int key, int code, int action, int mode);
    static void handleMouse(GLFWwindow* window, double xPos, double yPos); 

public:
    Window();
    Window(GLint windowWidth, GLint windowHeight);
    ~Window();

    int Initialise();

    GLint getBufferWidth() {return bufferWidth; }
    GLint getBufferHeight() {return bufferHeight; }

    bool getShouldClose() { return glfwWindowShouldClose(mainWindow); }

    bool* getKeys() {return keys; }
    GLfloat getXChange();
    GLfloat getYChange();

    void swapBuffers() { return glfwSwapBuffers(mainWindow); }
};
