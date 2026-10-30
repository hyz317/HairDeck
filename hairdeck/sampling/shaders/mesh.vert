
#version 330 core
// Copied from mesh_to_sdf 0.0.15 (https://github.com/marian42/mesh_to_sdf), shaders/mesh.vert; unchanged but for this notice.
// MIT License, Copyright (c) 2020 Marian Kleineberg.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright
// notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Vertex Attributes
layout(location = 0) in vec3 position;
layout(location = NORMAL_LOC) in vec3 normal;
layout(location = INST_M_LOC) in mat4 inst_m;

// Uniforms
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

// Outputs
out vec3 frag_position;
out vec3 frag_normal;

void main()
{
    gl_Position = P * V * M * inst_m * vec4(position, 1);
    frag_position = vec3(M * inst_m * vec4(position, 1.0));

    mat4 N = transpose(inverse(M * inst_m));
    frag_normal = normalize(vec3(N * vec4(normal, 0.0)));
}