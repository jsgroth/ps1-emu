use clap::Parser;
use coffeepsx_frontend::app::App;
use coffeepsx_frontend::config::AppConfig;
use coffeepsx_frontend::emustate::EmulatorState;
use coffeepsx_frontend::guistate::GuiState;
use coffeepsx_frontend::{OpenFileType, UserEvent};
use env_logger::Env;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use winit::event::Event;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};

#[derive(Debug, Parser)]
struct Args {
    /// Run in headless mode. Will not open the GUI window, will immediately launch the specified
    /// file, and will exit when the emulator window is closed
    #[arg(long, default_value_t)]
    headless: bool,

    /// File path to use when running in headless mode. Will run the BIOS if not set
    #[arg(long, short = 'f')]
    headless_file: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(
        Env::default().default_filter_or("info,naga=warn,wgpu_core=warn,wgpu_hal=warn"),
    )
    .init();

    let args = Args::parse();

    let event_loop = EventLoop::with_user_event().build()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new("coffeepsx-config.toml".into());

    if args.headless {
        return run_headless(args.headless_file.as_ref(), app.config_mut(), event_loop);
    }

    let mut emu_state = EmulatorState::new(&app.config().input)?;
    let mut gui_state = GuiState::new(app, &event_loop)?;

    let sigint = install_ctrl_c_handler()?;

    let event_loop_proxy = event_loop.create_proxy();
    #[allow(deprecated)]
    event_loop.run(move |event, elwt| {
        if sigint.load(Ordering::Relaxed) {
            elwt.exit();
            return;
        }

        if let Event::UserEvent(UserEvent::Close) = &event {
            elwt.exit();
            return;
        }

        if let Err(err) =
            emu_state.handle_event(&event, elwt, &event_loop_proxy, gui_state.app_config_mut())
        {
            log::error!("Emulator error: {err}");
        }

        gui_state.handle_event(&event, elwt, &emu_state, &event_loop_proxy);

        throttle_if_necessary(&event, elwt);
    })?;

    Ok(())
}

fn run_headless(
    path: Option<&PathBuf>,
    config: &mut AppConfig,
    event_loop: EventLoop<UserEvent>,
) -> anyhow::Result<()> {
    let event_loop_proxy = event_loop.create_proxy();

    let mut emu_state = EmulatorState::new(&config.input)?;
    let mut first_event = Some(match path {
        Some(path) => {
            Event::UserEvent(UserEvent::FileOpened(OpenFileType::Open, Some(path.clone())))
        }
        None => Event::UserEvent(UserEvent::RunBios),
    });

    let sigint = install_ctrl_c_handler()?;

    #[allow(deprecated)]
    event_loop.run(move |event, elwt| {
        if let Some(first_event) = first_event.take()
            && let Err(err) = emu_state.handle_event(&first_event, elwt, &event_loop_proxy, config)
        {
            log::error!("Error initializing emulator: {err}");
            elwt.exit();
            return;
        }

        if sigint.load(Ordering::Relaxed) {
            elwt.exit();
            return;
        }

        if let Event::UserEvent(UserEvent::Close) = &event {
            elwt.exit();
            return;
        }

        if let Err(err) = emu_state.handle_event(&event, elwt, &event_loop_proxy, config) {
            log::error!("Emulator error: {err}");
        }

        if !emu_state.is_emulator_running() {
            elwt.exit();
            return;
        }

        throttle_if_necessary(&event, elwt);
    })?;

    Ok(())
}

fn throttle_if_necessary(event: &Event<UserEvent>, elwt: &ActiveEventLoop) {
    // Wait for 1ms every time the event queue is exhausted to prevent pegging a CPU core at
    // 100% while the app is running
    if matches!(event, Event::AboutToWait) {
        elwt.set_control_flow(ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(1)));
    }
}

fn install_ctrl_c_handler() -> anyhow::Result<Arc<AtomicBool>> {
    let sigint = Arc::new(AtomicBool::new(false));
    {
        let sigint = Arc::clone(&sigint);
        ctrlc::set_handler(move || sigint.store(true, Ordering::Relaxed))?;
    }

    Ok(sigint)
}
