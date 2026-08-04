**This repository has moved to [Codeberg](https://codeberg.org/jsgroth/CoffeePSX)**

# CoffeePSX

Work-in-progress attempt at a PlayStation emulator.

Some games are fully playable, but some do not boot or have major issues.

## Why?

Why write a PS1 emulator at all? For fun and learning, of course!

For a more interesting question, why is this a standalone emulator instead of a backend core in [jgenesis](https://github.com/jsgroth/jgenesis/)?

Short version is that I wanted more freedom to experiment with the interface between the backend (the actual console emulator) and the frontend (the thing that displays frames / plays audio / etc). jgenesis has an interface that is designed around consoles like Genesis/SNES where from an emulation perspective it's massively preferable to render frames on the host CPU rather than the GPU. For later consoles like PS1/Saturn/N64/etc., there's a lot of benefit to rendering frames on the host GPU instead of the host CPU, and I wanted the interface freedom to make it possible to take advantage of that.

I'm also trying out some alternative forms of video/audio synchronization, which is a lot easier to do when there's only 1 backend core.

I may merge the two projects in the future - there are definitely ways to create a single API that will work for both 2D consoles and PS1 - but for the time being this is its own thing.

## Status

### Implemented

* CPU
  * Currently implemented using a plain interpreter; performance could be a lot better
* GTE (3D math coprocessor)
* GPU, with both software and hardware rasterizers
  * Hardware rasterizer uses [wgpu](https://wgpu.rs/) with native extensions; should work on Vulkan, DirectX 12, and Metal (has not been tested on MacOS/Metal)
  * Hardware rasterizer supports higher resolutions up to 16x native as well as 24bpp high color rendering
  * Supports basic PGXP (Parallel/Precision Geometry Transform Pipeline), which reduces model wobble and texture warping in many 3D games
    * "CPU mode" is not yet implemented so the PGXP implementation is not compatible with some games (e.g. Spyro series, Metal Gear Solid, Resident Evil 3, Tony Hawk's Pro Skater series)
* SPU (sound processor)
* Most of the CD-ROM controller
* Support for loading CUE/BIN disc images, CHD disc images, and PS1 EXE files
* MDEC (hardware image decompressor)
* Hardware timers
* NTSC/60Hz and PAL/50Hz support
* Digital and analog controllers
* Memory cards

### Not Yet Implemented

* Hotkey configuration
* DualShock rumble support
* Some sort of memory card management UI
* Additional graphical enhancements for the hardware rasterizer (e.g. PGXP CPU mode, texture filtering, downsampling)
* More efficient CPU emulation (cached interpreter, recompiler)
* More accurate timings for DMA/GPU/MDEC; some games that depend on DMA timing work, but timings are quite inaccurate right now
* Some CD-ROM functionality including infrequently used commands and 8-bit CD-XA audio
  * There are possibly no games that use 8-bit CD-XA audio samples?
* Various emulator enhancements like increased disc drive speed, CPU overclocking, rewind, save state slots
* Accurate timing for memory writes (i.e. emulating the CPU write queue)
  * It seems like maybe nothing depends on this?

## Software Rasterizer AVX2 Dependency

The software rasterizer makes very heavy use of x86_64 [AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions_2) instructions. These have been supported in Intel CPUs since Haswell (Q2 2013) and AMD CPUs since Excavator (Q2 2015). There is a fallback rasterizer that does not use any x86_64 intrinsics but it is extremely slow and will probably not run 3D games at full speed.

The hardware rasterizer has no such dependency.

## Build Dependencies

### Rust

This project requires the latest stable version of the [Rust toolchain](https://doc.rust-lang.org/book/ch01-01-installation.html) to build.

### SDL2

This project uses [SDL2](https://www.libsdl.org/) for audio and gamepad inputs.

Linux (Debian-based):
```shell
sudo apt install libsdl2-dev
```

MacOS:
```shell
brew install sdl2
```

Windows:
* https://github.com/libsdl-org/SDL/releases (use a 2.x version)

## Build & Run

To run the GUI:
```shell
cargo run --release
```

To run in headless mode (no GUI window, will exit when the emulator window is closed):
```shell
cargo run --release -- --headless -f /path/to/file.cue
```

To build with fat LTOs (link time optimizations), which slightly improves performance and decreases binary size but increases compile time:
```shell
cargo build --profile release-lto
# Executable located in target/release-lto/
```

## Hotkey Bindings
* Save state: F5 key
* Load state: F6 key
* Fast forward: Tab key
* Pause: P key
* Step to next frame: N key
* Select hardware rasterizer: 0 key
* Select software rasterizer: - key (Minus)
* Decrease resolution scale: [ key (Left square bracket)
* Increase resolution scale: ] key (Right square bracket)
* Toggle VRAM view: ' key (Quote)
* Exit: Esc key

## Screenshots

![ff9](https://github.com/user-attachments/assets/106b97eb-7198-48c1-a233-20e619d5c791)
![mmx4](https://github.com/user-attachments/assets/dc56666e-fc70-4f9b-b605-8138753df6f6)
![crash2](https://github.com/user-attachments/assets/4cce3bfa-50c7-41b9-b21e-92902ff8c0be)
![valkyrie-profile](https://github.com/user-attachments/assets/58f18208-833e-4876-b7d5-19b4007da50e)
