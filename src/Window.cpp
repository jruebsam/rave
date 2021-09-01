#include "Window.h"

#include <iostream>


Window::Window() {
    width = 800;
    height = 600;

    for(size_t i = 0; i < 1024; i++){
        keys[i] = 0;
    }
}

Window::Window(GLint windowWidth, GLint windowHeight) {
    width = windowWidth;
    height = windowHeight;

    for(size_t i = 0; i < 1024; i++){
        keys[i] = 0;
    }
}

void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}


int Window::Initialise() {
    glfwSetErrorCallback(error_callback);

    if (!glfwInit()){
        std::cout << "GLFW initialisation failed!" << std::endl;
        glfwTerminate();
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    mainWindow = glfwCreateWindow(width, height, "Test Window", NULL, NULL);

    if(!mainWindow){
        std::cout << "GLFW windows creation failed!" << std::endl;
        glfwTerminate();
        return 1;
    }

    glfwGetFramebufferSize(mainWindow, &bufferWidth, &bufferHeight);
    glfwMakeContextCurrent(mainWindow);

    createCallbacks();

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        glfwDestroyWindow(mainWindow);
        glfwTerminate();
        return -1;
    }    
    
    // Setup viewport size
    glViewport(0, 0, bufferWidth, bufferHeight);
    glfwSetWindowUserPointer(mainWindow, this);
}

void Window::createCallbacks() {
    glfwSetKeyCallback(mainWindow, handleKeys);
    glfwSetCursorPosCallback(mainWindow, handleMouse);
}

GLfloat Window::getXChange() {
    GLfloat theChange = xChange;
    xChange = 0.0f;
    return theChange;
}
GLfloat Window::getYChange() {
    GLfloat theChange = yChange;
    yChange = 0.0f;
    return theChange;
}

void Window::handleKeys(GLFWwindow* window, int key, int code, int action, int mode) {
    Window* theWindow = static_cast<Window*>(glfwGetWindowUserPointer(window));

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
    if (key >= 0 && key < 1024) {
        if (action == GLFW_PRESS) {
            theWindow->keys[key] = true;
        }
        else if (action == GLFW_RELEASE) {
            theWindow->keys[key] = false;
        }
    } 
}

void Window::handleMouse(GLFWwindow* window, double xPos, double yPos) {
    Window* theWindow = static_cast<Window*>(glfwGetWindowUserPointer(window));

    if (theWindow->mousedFirstMoved)
    {
        theWindow->lastX = xPos;
        theWindow->lastY = yPos;
        theWindow->mousedFirstMoved = false;
    }

    theWindow->xChange = xPos - theWindow->lastX;
    theWindow->yChange = theWindow->lastY - yPos;
    theWindow->lastX = xPos;
    theWindow->lastY = yPos;

}

Window::~Window() {
    glfwDestroyWindow(mainWindow);
    glfwTerminate();
}