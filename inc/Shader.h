#pragma once

#include "glad/glad.h"
#include <string>

class Shader
{
private:
    GLuint shaderID;
    void CompileShader(const char * vertexCode, const char* fragmentCode);
    void CompileProgram();
    void AddShader(GLuint theProgram, const char* shaderCode, GLenum shaderType);
public:
    Shader();
    void CreateFromString(const char* vertexCode, const char* fragmentCode);
    void CreateFromFiles(const char* vertexLocation, const char* fragmentLocation);
    void Validate();


    std::string ReadFile(const char* fileLocation);

    void UseShader();
    void ClearShader();
    ~Shader();
};
