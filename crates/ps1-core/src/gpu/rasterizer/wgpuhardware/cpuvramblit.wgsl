// Must match ShaderCpuVramBlitArgs in Rust
struct CpuVramBlitArgs {
    position: vec2u,
    size: vec2u,
    force_mask_bit: u32,
    check_mask_bit: u32,
}

@group(0) @binding(0)
var native_vram: texture_storage_2d<r32uint, read_write>;
@group(1) @binding(0)
var<storage> blit_buffer: array<u32>;

var<immediate> args: CpuVramBlitArgs;

@compute
@workgroup_size(16, 16, 1)
fn cpu_vram_blit(@builtin(global_invocation_id) invocation: vec3u) {
    if invocation.x >= args.size.x || invocation.y >= args.size.y {
        return;
    }

    let tex_x = (invocation.x + args.position.x) & 1023;
    let tex_y = (invocation.y + args.position.y) & 511;
    if args.check_mask_bit != 0 {
        let texel = textureLoad(native_vram, vec2u(tex_x, tex_y)).r;
        if (texel & 0x8000) != 0 {
            return;
        }
    }

    let buffer_idx = args.size.x * invocation.y + invocation.x;
    let value = blit_buffer[buffer_idx] | (args.force_mask_bit << 15);

    textureStore(native_vram, vec2u(tex_x, tex_y), vec4u(value, 0, 0, 0));
}