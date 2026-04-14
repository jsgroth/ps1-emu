const VERTICES: array<vec4f, 4> = array<vec4f, 4>(
    vec4f(-1.0, -1.0, 0.0, 1.0),
    vec4f(1.0, -1.0, 0.0, 1.0),
    vec4f(-1.0, 1.0, 0.0, 1.0),
    vec4f(1.0, 1.0, 0.0, 1.0),
);

const TEXTURE_COORDS: array<vec2f, 4> = array<vec2f, 4>(
    vec2f(0.0, 1.0),
    vec2f(1.0, 1.0),
    vec2f(0.0, 0.0),
    vec2f(1.0, 0.0),
);

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) texture_coords: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    return VertexOutput(VERTICES[vertex_index], TEXTURE_COORDS[vertex_index]);
}

@group(0) @binding(0)
var frame_sampler: sampler;
@group(1) @binding(0)
var frame: texture_2d<f32>;

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    return textureSample(frame, frame_sampler, input.texture_coords);
}