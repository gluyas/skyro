#version 330

layout(std140) uniform Mvp {
    mat4 modelview;
    mat4 projection;
};

in vec3 position;

in vec4 vertex_color;
out vec4 color;

out gl_PerVertex
{
    vec4 gl_Position;
};

void main() {
    gl_Position = projection * modelview * vec4(position, 1.0);
    color = vertex_color;
}
