#include "Shader.h"
#include <fstream>

Shader::Shader()
{
    shaderID = 0;
}

void Shader::CreateFromString(const char * vertexCode, const char* fragmentCode) {
    CompileShader(vertexCode, fragmentCode);
}

std::string Shader::ReadFile(const char* fileLocation){
    std::string content;
    std::ifstream fileStream(fileLocation, std::ios::in);

    if(!fileStream.is_open()){
        printf("Failed to read %s! File doesn't exist.", fileLocation);
        return "";
    }

    std::string line = "";
    while (!fileStream.eof()){
        std::getline(fileStream, line);
        content.append(line + "\n");
    }

    fileStream.close();
    return content;
}

void Shader::CreateFromFiles(const char* vertexLocation, const char* fragmentLocation){
    std::string vertexString = ReadFile(vertexLocation);
    std::string fragmentString = ReadFile(fragmentLocation);
    const char* vertexCode = vertexString.c_str();
    const char* fragmentCode = fragmentString.c_str();
    CompileShader(vertexCode, fragmentCode);
}

void Shader::CompileShader(const char * vertexCode, const char* fragmentCode) {
    shaderID = glCreateProgram();
    
    if(!shaderID){
        printf("Error creating shader programm!\n");
        return;
    }
    AddShader(shaderID, vertexCode, GL_VERTEX_SHADER);
    AddShader(shaderID, fragmentCode, GL_FRAGMENT_SHADER);

    CompileProgram();
}

void Shader::Validate()
{
    GLint result = 0;
    GLchar eLog[1024] = { 0 };

    glValidateProgram(shaderID);
    glGetProgramiv(shaderID, GL_VALIDATE_STATUS, &result);
    if (!result){
        glGetProgramInfoLog(shaderID, sizeof(eLog), NULL, eLog);
        printf("Error validating program: '%s' \n", eLog);
        return;
    }

}

void Shader::CompileProgram()
{
    GLint result = 0;
    GLchar eLog[1024] = { 0 };

    glLinkProgram(shaderID);
    glGetProgramiv(shaderID, GL_LINK_STATUS, &result);
    if (!result){
        glGetProgramInfoLog(shaderID, sizeof(eLog), NULL, eLog);
        printf("Error linking program: '%s' \n", eLog);
        return;
    }
}

void Shader::AddShader(GLuint theProgram, const char* shaderCode, GLenum shaderType){
    GLuint theShader = glCreateShader(shaderType);

    const GLchar* theCode[1];
    theCode[0] = shaderCode;
    GLint codeLength[1];
    codeLength[0] = strlen(shaderCode);

    glShaderSource(theShader, 1, theCode, codeLength);
    glCompileShader(theShader);

    GLint result = 0;
    GLchar eLog[1024] = { 0 };

    glGetShaderiv(theShader, GL_COMPILE_STATUS, &result);
    if (!result){
        glGetShaderInfoLog(theShader, sizeof(eLog), NULL, eLog);
        printf("Error compiling the %d shader: '%s' \n", shaderType, eLog);
        return;
    }

    glAttachShader(theProgram, theShader);
}


void Shader::UseShader(){
    glUseProgram(shaderID);
}
void Shader::ClearShader(){
    if (shaderID != 0){
        glDeleteProgram(shaderID);
        shaderID = 0;
    }
}

Shader::~Shader()
{
    ClearShader();
}



