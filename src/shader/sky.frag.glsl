#version 330

in vec4 color;

out vec4 gl_FragColor;

void main() {
    gl_FragColor = color;
    if (!gl_FrontFacing) gl_FragColor.a *= 0.5;
}
