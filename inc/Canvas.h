#pragma once

#include "glad/glad.h"

class Canvas
{
private:
    GLuint VBO = 0, VAO = 0, EBO = 0;
public:
    Canvas(const float border = 0.05);
    ~Canvas();
    void Canvas::Render();
};

