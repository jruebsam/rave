#include "Canvas.h"

Canvas::Canvas(const int _width, const int _height, const float border)
: width(_width), height(_height)
{
    float size = 1.0f - border;
    data = new unsigned char[width*height*3];

    GLfloat vertices[] = {-size, -size, 0.0f, 0.0f, 0.0f,
                          -size,  size, 0.0f, 0.0f, 1.0f,
                           size, -size, 0.0f, 1.0f, 0.0f,
                           size,  size, 0.0f, 1.0f, 1.0f};
    unsigned int indices[] = {0, 1, 2, 1, 2, 3};

    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glGenVertexArrays(1, &VAO);
    glGenTextures(1, &texture);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3*sizeof(float)));

    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Canvas::Render(){
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0); 

    glBindBuffer(0, EBO);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

unsigned char* Canvas::getBufferHandle()
{
    return data;
}

Canvas::~Canvas()
{
    if(EBO != 0) {
        glDeleteBuffers(1, &EBO);
        EBO = 0;
    }
    if(VBO != 0) {
        glDeleteBuffers(1, &VBO);
        VBO = 0;
    }
    if(VAO != 0) {
        glDeleteVertexArrays(1, &VAO);
        VAO = 0;
    }
    delete[] data;
}