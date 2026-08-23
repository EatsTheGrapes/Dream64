struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) destination: vec4<f32>,
    @location(1) atlas_uv: vec4<f32>,
    @location(2) clip: vec4<f32>,
    @location(3) viewport: vec2<f32>,
    @location(4) tint: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) screen_position: vec2<f32>,
    @location(2) @interpolate(flat) clip: vec4<f32>,
    @location(3) @interpolate(flat) tint: vec4<f32>,
};

@group(0) @binding(0) var sprite_atlas: texture_2d<f32>;
@group(0) @binding(1) var sprite_sampler: sampler;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    let corner = corners[input.vertex_index];
    let screen = input.destination.xy + corner * input.destination.zw;
    var output: VertexOutput;
    output.position = vec4<f32>(
        screen.x / input.viewport.x * 2.0 - 1.0,
        1.0 - screen.y / input.viewport.y * 2.0,
        0.0,
        1.0,
    );
    output.uv = input.atlas_uv.xy + corner * input.atlas_uv.zw;
    output.screen_position = screen;
    output.clip = input.clip;
    output.tint = input.tint;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if (input.screen_position.x < input.clip.x ||
        input.screen_position.y < input.clip.y ||
        input.screen_position.x >= input.clip.z ||
        input.screen_position.y >= input.clip.w) {
        discard;
    }
    return textureSample(sprite_atlas, sprite_sampler, input.uv) * input.tint;
}
