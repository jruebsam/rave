#include "Canvas.h"

Canvas::Canvas(const float border)
{
    float size = 1.0f - border;

    GLfloat vertices[] = {-size, -size, 0.0f, 
                          -size,  size, 0.0f,
                           size, -size, 0.0f,
                           size,  size, 0.0f};
    unsigned int indices[] = {0, 1, 2, 1, 2, 3};

    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glGenVertexArrays(1, &VAO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Canvas::Render(){
    glBindVertexArray(VAO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0); 
    glBindBuffer(0, EBO);
    glBindVertexArray(0);
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
}