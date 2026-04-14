struct VramFillArgs {
    position: vec2u,
    size: vec2u,
    color: u32,
}

@group(0) @binding(0)
var native_vram: texture_storage_2d<r32uint, write>;

var<immediate> args: VramFillArgs;

@compute
@workgroup_size(16, 16, 1)
fn vram_fill(@builtin(global_invocation_id) invocation: vec3u) {
    if invocation.x >= args.size.x || invocation.y >= args.size.y {
        return;
    }

    let x = (invocation.x + args.position.x) & 1023;
    let y = (invocation.y + args.position.y) & 511;
    textureStore(native_vram, vec2u(x, y), vec4u(args.color, 0, 0, 0));
}