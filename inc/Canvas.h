#pragma once

#include "glad/glad.h"

class Canvas
{
private:
    GLuint VBO = 0, VAO = 0, EBO = 0, texture;
    int width, height;
public:
    Canvas(const int width, const int height, const float border = 0.05);
    ~Canvas();
    void Render();
    GLuint getTextureID();
};

