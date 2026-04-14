struct VramCopyArgs {
    source: vec2u,
    destination: vec2u,
    size: vec2u,
    force_mask_bit: u32,
    check_mask_bit: u32,
    resolution_scale: u32,
}

@group(0) @binding(0)
var scaled_vram: texture_storage_2d<rgba8unorm, read_write>;

var<immediate> args: VramCopyArgs;

@compute
@workgroup_size(16, 16, 1)
fn vram_copy(@builtin(global_invocation_id) invocation: vec3u) {
    if invocation.x >= args.size.x || invocation.y >= args.size.y {
        return;
    }

    let vram_width = args.resolution_scale * 1024;
    let vram_height = args.resolution_scale * 512;

    let destination_x = (args.destination.x + invocation.x) % vram_width;
    let destination_y = (args.destination.y + invocation.y) % vram_height;
    let destination = vec2u(destination_x, destination_y);

    if args.check_mask_bit != 0 {
        let existing_texel = textureLoad(scaled_vram, destination);
        if existing_texel.a != 0.0 {
            return;
        }
    }

    let source_x = (args.source.x + invocation.x) % vram_width;
    let source_y = (args.source.y + invocation.y) % vram_height;
    let source = vec2u(source_x, source_y);

    var texel = textureLoad(scaled_vram, source);
    if args.force_mask_bit != 0 {
        texel.a = 1.0;
    }

    textureStore(scaled_vram, destination, texel);
}