#include "glad/glad.h"

#include <GLFW/glfw3.h>
#include <iostream>

#include "Window.h"
#include "Shader.h"
#include "Canvas.h"

#include "Simulation.h"
#include "State.h"

#include "xtensor/xio.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include <iostream>


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


    xt::view(sim->state.host.T, xt::range(300, 400), xt::range(300, 400)) = 1;
    xt::view(sim->state.host.T, xt::range(0, 1), xt::range(0, 1)) = 1;
    std::cout << sim->state.host.T <<std::endl;

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

