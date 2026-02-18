//! GP0 command processing
//!
//! GP0 commands are primarily related to drawing graphics

use crate::gpu::rasterizer::{
    CpuVramBlitArgs, DrawLineArgs, DrawRectangleArgs, DrawTriangleArgs, LineShading,
    RectangleTextureMapping, TextureMappingMode, TriangleShading, TriangleTextureMapping,
    VramVramBlitArgs,
};
use crate::gpu::{Color, Gpu, Vertex};
use crate::num::U32Ext;
use crate::pgxp::PreciseVertex;
use bincode::{Decode, Encode};
use std::array;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
pub enum PolygonVertices {
    Three,
    Four,
}

impl PolygonVertices {
    fn from_bit(bit: bool) -> Self {
        if bit { Self::Four } else { Self::Three }
    }
}

impl From<PolygonVertices> for u8 {
    fn from(value: PolygonVertices) -> Self {
        match value {
            PolygonVertices::Three => 3,
            PolygonVertices::Four => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
pub enum RectangleSize {
    Variable,
    One,
    Eight,
    Sixteen,
}

impl RectangleSize {
    fn from_bits(bits: u32) -> Self {
        match bits & 3 {
            0 => Self::Variable,
            1 => Self::One,
            2 => Self::Eight,
            3 => Self::Sixteen,
            _ => unreachable!("value & 3 is always <= 3"),
        }
    }
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub struct LineCommandParameters {
    pub gouraud_shading: bool,
    pub polyline: bool,
    pub semi_transparent: bool,
    pub color: Color,
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub struct PolygonCommandParameters {
    pub vertices: PolygonVertices,
    pub gouraud_shading: bool,
    pub textured: bool,
    pub semi_transparent: bool,
    pub raw_texture: bool,
    pub color: Color,
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub struct RectangleCommandParameters {
    pub size: RectangleSize,
    pub textured: bool,
    pub semi_transparent: bool,
    pub raw_texture: bool,
    pub color: Color,
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub enum DrawCommand {
    Fill(Color),
    DrawLine(LineCommandParameters),
    DrawPolygon(PolygonCommandParameters),
    DrawRectangle(RectangleCommandParameters),
    VramToVramBlit,
    CpuToVramBlit,
    VramToCpuBlit,
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub struct VramTransferFields {
    destination_x: u32,
    destination_y: u32,
    x_size: u32,
    y_size: u32,
    halfwords_remaining: u32,
}

#[derive(Debug, Clone, Copy, Encode, Decode, Default)]
pub enum Gp0CommandState {
    #[default]
    WaitingForCommand,
    WaitingForParameters {
        command: DrawCommand,
        index: u8,
        remaining: u8,
    },
    WaitingForPolyline(LineCommandParameters),
    ReceivingFromCpu(VramTransferFields),
    SendingToCpu {
        buffer_idx: u32,
        halfwords_remaining: u32,
    },
}

impl Gp0CommandState {
    const VRAM_TO_VRAM_BLIT: Self =
        Self::WaitingForParameters { command: DrawCommand::VramToVramBlit, index: 0, remaining: 3 };

    const CPU_TO_VRAM_BLIT: Self =
        Self::WaitingForParameters { command: DrawCommand::CpuToVramBlit, index: 0, remaining: 2 };

    const VRAM_TO_CPU_BLIT: Self =
        Self::WaitingForParameters { command: DrawCommand::VramToCpuBlit, index: 0, remaining: 2 };

    fn fill(value: u32) -> Self {
        let color = parse_command_color(value);
        Self::WaitingForParameters { command: DrawCommand::Fill(color), index: 0, remaining: 2 }
    }

    fn draw_line(value: u32) -> Self {
        let color = parse_command_color(value);

        let gouraud_shading = value.bit(28);
        let polyline = value.bit(27);
        let semi_transparent = value.bit(25);

        let parameters = 2 + u8::from(gouraud_shading);

        Self::WaitingForParameters {
            command: DrawCommand::DrawLine(LineCommandParameters {
                gouraud_shading,
                polyline,
                semi_transparent,
                color,
            }),
            index: 0,
            remaining: parameters,
        }
    }

    fn draw_polygon(value: u32) -> Self {
        let gouraud_shading = value.bit(28);
        let vertices = PolygonVertices::from_bit(value.bit(27));
        let textured = value.bit(26);
        let semi_transparent = value.bit(25);
        let raw_texture = value.bit(24);
        let color = parse_command_color(value);

        let vertex_count: u8 = vertices.into();

        // Each vertex requires 1-3 parameters:
        // - 1 parameter for the coordinates
        // - 1 parameter for the color (only if Gouraud shading is enabled, and not for the first
        //   vertex because it uses the color from the command word)
        // - 1 parameter for the U/V texture coordinates (only for textured polygons)
        let parameters = vertex_count * (1 + u8::from(textured))
            + (vertex_count - 1) * u8::from(gouraud_shading);

        let command = DrawCommand::DrawPolygon(PolygonCommandParameters {
            vertices,
            gouraud_shading,
            textured,
            semi_transparent,
            raw_texture,
            color,
        });

        Self::WaitingForParameters { command, index: 0, remaining: parameters }
    }

    fn draw_rectangle(value: u32) -> Self {
        let size = RectangleSize::from_bits(value >> 27);
        let textured = value.bit(26);
        let semi_transparent = value.bit(25);
        let raw_texture = value.bit(24);
        let color = parse_command_color(value);

        let parameters = 1 + u8::from(textured) + u8::from(size == RectangleSize::Variable);

        let command = DrawCommand::DrawRectangle(RectangleCommandParameters {
            size,
            textured,
            semi_transparent,
            raw_texture,
            color,
        });

        Self::WaitingForParameters { command, index: 0, remaining: parameters }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Encode, Decode)]
pub enum SemiTransparencyMode {
    // B/2 + F/2
    #[default]
    Average = 0,
    // B + F
    Add = 1,
    // B - F
    Subtract = 2,
    // B + F/4
    AddQuarter = 3,
}

impl SemiTransparencyMode {
    fn from_bits(bits: u32) -> Self {
        match bits & 3 {
            0 => Self::Average,
            1 => Self::Add,
            2 => Self::Subtract,
            3 => Self::AddQuarter,
            _ => unreachable!("value & 3 is always <= 3"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Encode, Decode)]
pub enum TextureColorDepthBits {
    #[default]
    Four = 0,
    Eight = 1,
    Fifteen = 2,
}

impl TextureColorDepthBits {
    fn from_bits(bits: u32) -> Self {
        match bits & 3 {
            0 => Self::Four,
            1 => Self::Eight,
            // Setting 3 ("reserved") functions the same as setting 2 (15-bit)
            2 | 3 => Self::Fifteen,
            _ => unreachable!("value & 3 is always <= 3"),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Encode, Decode)]
pub struct TexturePage {
    // In 64-halfword steps
    pub x_base: u32,
    // 0 or 256 (can technically also be 512 or 768 but those are not supported on retail consoles)
    pub y_base: u32,
    pub semi_transparency_mode: SemiTransparencyMode,
    pub color_depth: TextureColorDepthBits,
    pub rectangle_x_flip: bool,
    pub rectangle_y_flip: bool,
}

impl TexturePage {
    fn from_command_word(command: u32) -> Self {
        Self {
            x_base: command & 0xF,
            y_base: 256 * ((command >> 4) & 1),
            semi_transparency_mode: SemiTransparencyMode::from_bits(command >> 5),
            color_depth: TextureColorDepthBits::from_bits(command >> 7),
            rectangle_x_flip: command.bit(12),
            rectangle_y_flip: command.bit(13),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Encode, Decode)]
pub struct TextureWindow {
    // All values in 8-pixel steps
    pub x_mask: u32,
    pub y_mask: u32,
    pub x_offset: u32,
    pub y_offset: u32,
}

impl TextureWindow {
    fn from_command_word(command: u32) -> Self {
        Self {
            x_mask: command & 0x1F,
            y_mask: (command >> 5) & 0x1F,
            x_offset: (command >> 10) & 0x1F,
            y_offset: (command >> 15) & 0x1F,
        }
    }

    pub fn to_word(self) -> u32 {
        self.x_mask | (self.y_mask << 5) | (self.x_offset << 10) | (self.y_offset << 15)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Encode, Decode)]
pub struct DrawSettings {
    pub drawing_in_display_allowed: bool,
    pub dithering_enabled: bool,
    pub draw_area_top_left: Vertex,
    pub draw_area_bottom_right: Vertex,
    pub draw_offset: Vertex,
    pub force_mask_bit: bool,
    pub check_mask_bit: bool,
}

const PARAMETERS_LEN: usize = 11;

#[derive(Debug, Clone, Encode, Decode)]
pub struct Gp0State {
    pub command_state: Gp0CommandState,
    pub parameters: [u32; PARAMETERS_LEN],
    pub pgxp_parameters: [PreciseVertex; PARAMETERS_LEN],
    pub global_texture_page: TexturePage,
    pub texture_window: TextureWindow,
    pub draw_settings: DrawSettings,
    pub blit_buffer: Vec<u16>,
}

impl Gp0State {
    pub fn new() -> Self {
        Self {
            command_state: Gp0CommandState::default(),
            parameters: array::from_fn(|_| 0),
            pgxp_parameters: array::from_fn(|_| PreciseVertex::default()),
            global_texture_page: TexturePage::default(),
            texture_window: TextureWindow::default(),
            draw_settings: DrawSettings::default(),
            blit_buffer: Vec::with_capacity(1024 * 512),
        }
    }
}

impl Gpu {
    pub(super) fn read_vram_word_for_cpu(
        &mut self,
        buffer_idx: u32,
        mut halfwords_remaining: u32,
    ) -> u32 {
        let mut word = 0_u32;
        for i in 0..2 {
            let halfword =
                self.gp0.blit_buffer.get((buffer_idx + i) as usize).copied().unwrap_or(0);
            word |= u32::from(halfword) << (16 * i);

            halfwords_remaining -= 1;
            if halfwords_remaining == 0 {
                log::trace!("VRAM-to-CPU blit finished, sending word {word:08X} to CPU");

                self.gp0.command_state = Gp0CommandState::WaitingForCommand;
                return word;
            }
        }

        log::trace!("VRAM-to-CPU blit in progress, sending word {word:08X} to CPU");

        self.gp0.command_state =
            Gp0CommandState::SendingToCpu { buffer_idx: buffer_idx + 2, halfwords_remaining };
        word
    }

    #[allow(clippy::match_same_arms)]
    pub(super) fn handle_gp0_write(&mut self, value: u32, pgxp_vertex: PreciseVertex) {
        log::trace!("GP0 command write: {value:08X}");

        self.gp0.command_state = match self.gp0.command_state {
            Gp0CommandState::WaitingForCommand => match value >> 29 {
                0 => {
                    match value >> 24 {
                        0x00 | 0x03..=0x1E => {
                            // GP0($00): Apparently a no-op? Functionally unknown
                            log::debug!("No-op/invalid GP0 command: {value:08X}");
                            Gp0CommandState::WaitingForCommand
                        }
                        0x01 => {
                            // GP0($01): Clear texture cache
                            // TODO emulate texture cache?
                            self.rasterizer.clear_texture_cache();
                            log::debug!("Clear texture cache command: {value:08X}");
                            Gp0CommandState::WaitingForCommand
                        }
                        0x02 => {
                            // GP0($02): VRAM fill
                            Gp0CommandState::fill(value)
                        }
                        0x1F => {
                            // GP0($1F): Set GPU IRQ flag
                            // Apparently nothing uses this feature? Except for one game that seems
                            // to accidentally send a GP0($1F) command
                            todo!("GP0($1F) - set GPU IRQ")
                        }
                        _ => unreachable!(
                            "highest 3 bits are 0, highest byte must be in 0x00..=0x1F"
                        ),
                    }
                }
                1 => Gp0CommandState::draw_polygon(value),
                2 => Gp0CommandState::draw_line(value),
                3 => Gp0CommandState::draw_rectangle(value),
                4 => Gp0CommandState::VRAM_TO_VRAM_BLIT,
                5 => Gp0CommandState::CPU_TO_VRAM_BLIT,
                6 => Gp0CommandState::VRAM_TO_CPU_BLIT,
                7 => {
                    // All commands starting with 111 are settings commands that take no parameters
                    self.execute_settings_command(value);
                    Gp0CommandState::WaitingForCommand
                }
                _ => unreachable!("highest 3 bits must be <= 7"),
            },
            Gp0CommandState::WaitingForParameters { command, index, remaining } => {
                self.gp0.parameters[index as usize] = value;
                self.gp0.pgxp_parameters[index as usize] = pgxp_vertex;

                if remaining == 1 {
                    self.execute_draw_command(command)
                } else {
                    Gp0CommandState::WaitingForParameters {
                        command,
                        index: index + 1,
                        remaining: remaining - 1,
                    }
                }
            }
            Gp0CommandState::WaitingForPolyline(parameters) => {
                if value & 0xF000F000 == 0x50005000 {
                    // Polyline command end marker
                    Gp0CommandState::WaitingForCommand
                } else {
                    self.gp0.parameters[1] = value;
                    self.gp0.pgxp_parameters[1] = pgxp_vertex;

                    if parameters.gouraud_shading {
                        // Need to read one more word for the second vertex coordinate
                        Gp0CommandState::WaitingForParameters {
                            command: DrawCommand::DrawLine(parameters),
                            index: 2,
                            remaining: 1,
                        }
                    } else {
                        self.draw_line(parameters)
                    }
                }
            }
            Gp0CommandState::ReceivingFromCpu(fields) => {
                self.receive_vram_word_from_cpu(value, fields)
            }
            Gp0CommandState::SendingToCpu { .. } => {
                panic!("unexpected write to GP0 command buffer during VRAM-to-CPU blit")
            }
        };
    }

    fn execute_draw_command(&mut self, command: DrawCommand) -> Gp0CommandState {
        log::debug!("Executing GP0 command {command:?}");

        match command {
            DrawCommand::Fill(color) => {
                self.vram_fill(color);

                Gp0CommandState::WaitingForCommand
            }
            DrawCommand::DrawLine(parameters) => self.draw_line(parameters),
            DrawCommand::DrawPolygon(parameters) => {
                self.draw_polygon(parameters);

                Gp0CommandState::WaitingForCommand
            }
            DrawCommand::DrawRectangle(parameters) => {
                self.draw_rectangle(parameters);

                Gp0CommandState::WaitingForCommand
            }
            DrawCommand::VramToVramBlit => {
                self.execute_vram_copy();

                Gp0CommandState::WaitingForCommand
            }
            DrawCommand::CpuToVramBlit => {
                let (destination_x, destination_y) = parse_vram_position(self.gp0.parameters[0]);
                let (x_size, y_size) = parse_vram_size(self.gp0.parameters[1]);

                self.gp0.blit_buffer.clear();

                Gp0CommandState::ReceivingFromCpu(VramTransferFields {
                    destination_x,
                    destination_y,
                    x_size,
                    y_size,
                    halfwords_remaining: x_size * y_size,
                })
            }
            DrawCommand::VramToCpuBlit => {
                let (source_x, source_y) = parse_vram_position(self.gp0.parameters[0]);
                let (x_size, y_size) = parse_vram_size(self.gp0.parameters[1]);

                self.gp0.blit_buffer.clear();
                self.rasterizer.vram_to_cpu_blit(
                    source_x,
                    source_y,
                    x_size,
                    y_size,
                    &mut self.gp0.blit_buffer,
                );

                Gp0CommandState::SendingToCpu {
                    buffer_idx: 0,
                    halfwords_remaining: x_size * y_size,
                }
            }
        }
    }

    fn execute_settings_command(&mut self, command: u32) {
        // Highest 8 bits determine operation
        match command >> 24 {
            0xE1 => {
                // GP0($E1): Texture page & draw mode settings
                self.gp0.global_texture_page = TexturePage::from_command_word(command);
                self.gp0.draw_settings.drawing_in_display_allowed = command.bit(10);
                self.gp0.draw_settings.dithering_enabled = command.bit(9);

                log::debug!("Executed texture page / draw mode command: {command:08X}");
                log::debug!("  Global texture page: {:?}", self.gp0.global_texture_page);
                log::debug!(
                    "  Drawing allowed in display area: {}",
                    self.gp0.draw_settings.drawing_in_display_allowed
                );
                log::debug!(
                    "  Dithering from 24-bit to 15-bit enabled: {}",
                    self.gp0.draw_settings.dithering_enabled
                );
            }
            0xE2 => {
                // GP0($E2): Texture window settings
                self.gp0.texture_window = TextureWindow::from_command_word(command);

                log::debug!("Executed texture window settings command: {command:08X}");
                log::debug!("  Texture window: {:?}", self.gp0.texture_window);
            }
            0xE3 => {
                // GP0($E3): Drawing area top-left coordinates
                let x1 = (command & 0x3FF) as i32;
                let y1 = ((command >> 10) & 0x1FF) as i32;
                self.gp0.draw_settings.draw_area_top_left = Vertex { x: x1, y: y1 };

                log::debug!("Executed drawing area top-left command: {command:08X}");
                log::debug!("  (X1, Y1) = {:?}", self.gp0.draw_settings.draw_area_top_left);
            }
            0xE4 => {
                // GP0($E4): Drawing area bottom-right coordinates
                let x2 = (command & 0x3FF) as i32;
                let y2 = ((command >> 10) & 0x1FF) as i32;
                self.gp0.draw_settings.draw_area_bottom_right = Vertex { x: x2, y: y2 };

                log::debug!("Executed drawing area bottom-right command: {command:08X}");
                log::debug!("  (X2, Y2) = {:?}", self.gp0.draw_settings.draw_area_bottom_right);
            }
            0xE5 => {
                // GP0($E5): Drawing offset
                // Both values are signed 11-bit integers (-1024 to +1023)
                let x_offset = parse_signed_11_bit(command);
                let y_offset = parse_signed_11_bit(command >> 11);
                self.gp0.draw_settings.draw_offset = Vertex { x: x_offset, y: y_offset };

                log::debug!("Executed draw offset command: {command:08X}");
                log::debug!("  (X offset, Y offset) = {:?}", self.gp0.draw_settings.draw_offset);
            }
            0xE6 => {
                // GP0($E6): Mask bit settings
                self.gp0.draw_settings.force_mask_bit = command.bit(0);
                self.gp0.draw_settings.check_mask_bit = command.bit(1);

                log::debug!("Executed mask bit settings command: {command:08X}");
                log::debug!("  Force mask bit: {}", self.gp0.draw_settings.force_mask_bit);
                log::debug!("  Check mask bit on draw: {}", self.gp0.draw_settings.check_mask_bit);
            }
            0xE0 | 0xE7..=0xFF => {
                // Unknown/no-op commands?
                log::debug!("No-op/invalid GP0 command: {command:08X}");
            }
            _ => {
                panic!("execute_settings_command() called with high bits not all 1: {command:08X}")
            }
        }
    }

    fn vram_fill(&mut self, color: Color) {
        let x = self.gp0.parameters[0] & 0xFFFF;
        let y = self.gp0.parameters[0] >> 16;
        let width = self.gp0.parameters[1] & 0xFFFF;
        let height = self.gp0.parameters[1] >> 16;

        log::debug!("Executing VRAM fill with X={x}, Y={y}, width={width}, height={height}");

        self.rasterizer.vram_fill(x, y, width, height, color);
    }

    fn draw_line(&mut self, command_parameters: LineCommandParameters) -> Gp0CommandState {
        let line_args = parse_draw_line_parameters(
            command_parameters,
            &self.gp0.parameters,
            self.gp0.global_texture_page.semi_transparency_mode,
        );

        log::debug!("Executing draw line command: {line_args:?}");

        let v1 = line_args.vertices[1];
        let shading = line_args.shading;

        self.rasterizer.draw_line(line_args, &self.gp0.draw_settings);

        if command_parameters.polyline {
            // Pretend that the previous second vertex/color is now the first vertex/color
            self.gp0.parameters[0] = ((v1.x & 0xFFFF) | (v1.y << 16)) as u32;
            let new_first_color = match shading {
                LineShading::Flat(color) | LineShading::Gouraud([_, color]) => color,
            };
            return Gp0CommandState::WaitingForPolyline(LineCommandParameters {
                color: new_first_color,
                ..command_parameters
            });
        }

        Gp0CommandState::WaitingForCommand
    }

    fn draw_polygon(&mut self, command_parameters: PolygonCommandParameters) {
        let (first_args, second_args) = parse_draw_polygon_parameters(
            command_parameters,
            &self.gp0.parameters,
            &self.gp0.pgxp_parameters,
            self.gp0.global_texture_page.semi_transparency_mode,
            self.gp0.texture_window,
        );

        if let Some(texture_mapping) = &first_args.texture_mapping {
            // Drawing a textured polygon appears to change most of the current GP0($E1) settings.
            // Valkyrie Profile depends on this to change semi-transparency mode or most text boxes
            // will be the wrong color
            self.gp0.global_texture_page = TexturePage {
                rectangle_x_flip: self.gp0.global_texture_page.rectangle_x_flip,
                rectangle_y_flip: self.gp0.global_texture_page.rectangle_y_flip,
                ..texture_mapping.texpage
            };
        }

        log::debug!("Drawing polygon with params {first_args:?}");

        self.rasterizer.draw_triangle(first_args, &self.gp0.draw_settings);
        if let Some(second_args) = second_args {
            log::debug!("Drawing second polygon with params {second_args:?}");
            self.rasterizer.draw_triangle(second_args, &self.gp0.draw_settings);
        }
    }

    fn draw_rectangle(&mut self, command_parameters: RectangleCommandParameters) {
        let rectangle_args = parse_draw_rectangle_parameters(
            command_parameters,
            &self.gp0.parameters,
            self.gp0.global_texture_page,
            self.gp0.texture_window,
        );

        log::debug!("Drawing rectangle with parameters {rectangle_args:?}");

        self.rasterizer.draw_rectangle(rectangle_args, &self.gp0.draw_settings);
    }

    fn execute_vram_copy(&mut self) {
        let source_x = self.gp0.parameters[0] & 0x3FF;
        let source_y = (self.gp0.parameters[0] >> 16) & 0x1FF;
        let dest_x = self.gp0.parameters[1] & 0x3FF;
        let dest_y = (self.gp0.parameters[1] >> 16) & 0x1FF;
        let width = (self.gp0.parameters[2].wrapping_sub(1) & 0x3FF) + 1;
        let height = ((self.gp0.parameters[2] >> 16).wrapping_sub(1) & 0x1FF) + 1;

        log::trace!(
            "Executing VRAM copy from X={source_x} / Y={source_y} to X={dest_x} / Y={dest_y}, width={width} and height={height}"
        );

        self.rasterizer.vram_to_vram_blit(VramVramBlitArgs {
            source_x,
            source_y,
            dest_x,
            dest_y,
            width,
            height,
            force_mask_bit: self.gp0.draw_settings.force_mask_bit,
            check_mask_bit: self.gp0.draw_settings.check_mask_bit,
        });
    }

    fn receive_vram_word_from_cpu(
        &mut self,
        value: u32,
        mut fields: VramTransferFields,
    ) -> Gp0CommandState {
        for halfword in [value & 0xFFFF, value >> 16] {
            self.gp0.blit_buffer.push(halfword as u16);

            fields.halfwords_remaining -= 1;
            if fields.halfwords_remaining == 0 {
                self.rasterizer.cpu_to_vram_blit(
                    CpuVramBlitArgs {
                        x: fields.destination_x,
                        y: fields.destination_y,
                        width: fields.x_size,
                        height: fields.y_size,
                        force_mask_bit: self.gp0.draw_settings.force_mask_bit,
                        check_mask_bit: self.gp0.draw_settings.check_mask_bit,
                    },
                    &self.gp0.blit_buffer,
                );
                return Gp0CommandState::WaitingForCommand;
            }
        }

        Gp0CommandState::ReceivingFromCpu(fields)
    }
}

fn parse_vram_position(value: u32) -> (u32, u32) {
    let x = value & 0x3FF;
    let y = (value >> 16) & 0x1FF;
    (x, y)
}

fn parse_vram_size(value: u32) -> (u32, u32) {
    let x = (value.wrapping_sub(1) & 0x3FF) + 1;
    let y = ((value >> 16).wrapping_sub(1) & 0x1FF) + 1;
    (x, y)
}

fn parse_command_color(value: u32) -> Color {
    let r = value as u8;
    let g = (value >> 8) as u8;
    let b = (value >> 16) as u8;

    Color { r, g, b }
}

fn parse_vertex_coordinates(parameter: u32) -> Vertex {
    // Vertex coordinates are signed 11-bit values, X in the low halfword and Y in the high halfword
    let x = parse_signed_11_bit(parameter);
    let y = parse_signed_11_bit(parameter >> 16);

    Vertex { x, y }
}

fn parse_signed_11_bit(word: u32) -> i32 {
    ((word as i32) << 21) >> 21
}

struct Gp0Parameters<'a>(&'a [u32]);

impl Gp0Parameters<'_> {
    fn next(&mut self) -> u32 {
        let value = self.0[0];
        self.0 = &self.0[1..];
        value
    }

    fn peek(&self) -> u32 {
        self.0[0]
    }
}

struct Gp0ParametersPgxp<'a>(&'a [u32], &'a [PreciseVertex]);

impl Gp0ParametersPgxp<'_> {
    fn next(&mut self) -> (u32, PreciseVertex) {
        let value = self.0[0];
        let vertex = self.1[0];

        self.0 = &self.0[1..];
        self.1 = &self.1[1..];

        (value, vertex)
    }

    fn next_word(&mut self) -> u32 {
        let value = self.0[0];
        self.0 = &self.0[1..];
        self.1 = &self.1[1..];
        value
    }

    fn peek(&self) -> u32 {
        self.0[0]
    }
}

fn parse_draw_line_parameters(
    command_parameters: LineCommandParameters,
    parameters: &[u32],
    semi_transparency_mode: SemiTransparencyMode,
) -> DrawLineArgs {
    let mut parameters = Gp0Parameters(parameters);

    let v0 = parse_vertex_coordinates(parameters.next());

    let shading = if command_parameters.gouraud_shading {
        let second_color = parse_command_color(parameters.next());
        LineShading::Gouraud([command_parameters.color, second_color])
    } else {
        LineShading::Flat(command_parameters.color)
    };

    let v1 = parse_vertex_coordinates(parameters.next());

    DrawLineArgs {
        vertices: [v0, v1],
        shading,
        semi_transparent: command_parameters.semi_transparent,
        semi_transparency_mode,
    }
}

fn parse_draw_polygon_parameters(
    command_parameters: PolygonCommandParameters,
    parameters: &[u32],
    pgxp_parameters: &[PreciseVertex],
    global_semi_transparency_mode: SemiTransparencyMode,
    texture_window: TextureWindow,
) -> (DrawTriangleArgs, Option<DrawTriangleArgs>) {
    let mut parameters = Gp0ParametersPgxp(parameters, pgxp_parameters);

    let mut vertices = [Vertex::default(); 4];
    let mut pgxp_vertices = [PreciseVertex::default(); 4];
    let mut pgxp_match = true;
    let mut colors = [Color::default(); 4];
    let mut u = [0; 4];
    let mut v = [0; 4];
    let mut clut_x = 0;
    let mut clut_y = 0;
    let mut texpage = TexturePage::default();

    colors[0] = command_parameters.color;

    for vertex_idx in 0..command_parameters.vertices.into() {
        if vertex_idx != 0 && command_parameters.gouraud_shading {
            colors[vertex_idx as usize] = parse_command_color(parameters.next_word());
        }

        let (vertex_word, pgxp_vertex) = parameters.next();
        let vertex = parse_vertex_coordinates(vertex_word);
        vertices[vertex_idx as usize] = vertex;
        pgxp_vertices[vertex_idx as usize] = pgxp_vertex;

        pgxp_match &= pgxp_vertex.is_valid() && pgxp_vertex.matches(vertex);

        if command_parameters.textured {
            match vertex_idx {
                0 => {
                    (clut_x, clut_y) = parse_clut_attribute(parameters.peek());
                }
                1 => {
                    texpage = TexturePage::from_command_word(parameters.peek() >> 16);
                }
                _ => {}
            }

            u[vertex_idx as usize] = parameters.peek() as u8;
            v[vertex_idx as usize] = (parameters.peek() >> 8) as u8;
            parameters.next();
        }
    }

    let texture_mode = if command_parameters.raw_texture {
        TextureMappingMode::Raw
    } else {
        TextureMappingMode::Modulated
    };

    let semi_transparency_mode = if command_parameters.textured {
        texpage.semi_transparency_mode
    } else {
        global_semi_transparency_mode
    };

    let first_parameters = DrawTriangleArgs {
        vertices: [vertices[0], vertices[1], vertices[2]],
        pgxp_vertices: pgxp_match.then_some([pgxp_vertices[0], pgxp_vertices[1], pgxp_vertices[2]]),
        shading: if command_parameters.gouraud_shading {
            TriangleShading::Gouraud([colors[0], colors[1], colors[2]])
        } else {
            TriangleShading::Flat(colors[0])
        },
        semi_transparent: command_parameters.semi_transparent,
        semi_transparency_mode,
        texture_mapping: command_parameters.textured.then_some(TriangleTextureMapping {
            mode: texture_mode,
            texpage,
            window: texture_window,
            clut_x,
            clut_y,
            u: [u[0], u[1], u[2]],
            v: [v[0], v[1], v[2]],
        }),
    };

    match command_parameters.vertices {
        PolygonVertices::Three => (first_parameters, None),
        PolygonVertices::Four => {
            let second_parameters = DrawTriangleArgs {
                vertices: [vertices[1], vertices[2], vertices[3]],
                pgxp_vertices: pgxp_match.then_some([
                    pgxp_vertices[1],
                    pgxp_vertices[2],
                    pgxp_vertices[3],
                ]),
                shading: if command_parameters.gouraud_shading {
                    TriangleShading::Gouraud([colors[1], colors[2], colors[3]])
                } else {
                    TriangleShading::Flat(colors[0])
                },
                semi_transparent: command_parameters.semi_transparent,
                semi_transparency_mode,
                texture_mapping: command_parameters.textured.then_some(TriangleTextureMapping {
                    mode: texture_mode,
                    texpage,
                    window: texture_window,
                    clut_x,
                    clut_y,
                    u: [u[1], u[2], u[3]],
                    v: [v[1], v[2], v[3]],
                }),
            };

            (first_parameters, Some(second_parameters))
        }
    }
}

fn parse_draw_rectangle_parameters(
    command_parameters: RectangleCommandParameters,
    parameters: &[u32],
    global_texpage: TexturePage,
    texture_window: TextureWindow,
) -> DrawRectangleArgs {
    let mut parameters = Gp0Parameters(parameters);

    let position = parse_vertex_coordinates(parameters.next());

    let texture_mapping = command_parameters.textured.then(|| {
        let texture_mode = if command_parameters.raw_texture {
            TextureMappingMode::Raw
        } else {
            TextureMappingMode::Modulated
        };
        let (clut_x, clut_y) = parse_clut_attribute(parameters.peek());
        let u = parameters.peek() as u8;
        let v = (parameters.peek() >> 8) as u8;
        parameters.next();

        RectangleTextureMapping {
            mode: texture_mode,
            texpage: global_texpage,
            window: texture_window,
            clut_x,
            clut_y,
            u: [u],
            v: [v],
        }
    });

    let (width, height) = match command_parameters.size {
        RectangleSize::One => (1, 1),
        RectangleSize::Eight => (8, 8),
        RectangleSize::Sixteen => (16, 16),
        RectangleSize::Variable => {
            let width = parameters.peek() & 0x3FF;
            let height = (parameters.peek() >> 16) & 0x1FF;
            (width, height)
        }
    };

    DrawRectangleArgs {
        top_left: position,
        width,
        height,
        color: command_parameters.color,
        semi_transparent: command_parameters.semi_transparent,
        semi_transparency_mode: global_texpage.semi_transparency_mode,
        texture_mapping,
    }
}

fn parse_clut_attribute(value: u32) -> (u16, u16) {
    let clut_x = (value >> 16) & 0x3F;
    let clut_y = (value >> 22) & 0x1FF;
    (clut_x as u16, clut_y as u16)
}
