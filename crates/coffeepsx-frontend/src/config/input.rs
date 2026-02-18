use serde::de::{Error, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::{Display, Formatter};
use std::str::FromStr;
use winit::keyboard::KeyCode;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisDirection {
    Positive,
    Negative,
}

impl AxisDirection {
    #[must_use]
    pub fn max_value(self) -> i16 {
        match self {
            Self::Positive => i16::MAX,
            Self::Negative => i16::MIN,
        }
    }
}

impl Display for AxisDirection {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Positive => write!(f, "+"),
            Self::Negative => write!(f, "-"),
        }
    }
}

impl FromStr for AxisDirection {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "+" => Ok(Self::Positive),
            "-" => Ok(Self::Negative),
            _ => Err(format!("Invalid axis direction: '{s}'")),
        }
    }
}

// See `sdl2::controller::Button` and `sdl2::controller::Axis`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SdlGamepadInput {
    A,
    B,
    X,
    Y,
    Back,
    Guide,
    Start,
    LeftStick,
    RightStick,
    LeftShoulder,
    RightShoulder,
    DPadUp,
    DPadDown,
    DPadLeft,
    DPadRight,
    Misc1,
    Paddle1,
    Paddle2,
    Paddle3,
    Paddle4,
    Touchpad,
    LeftX(AxisDirection),
    LeftY(AxisDirection),
    RightX(AxisDirection),
    RightY(AxisDirection),
    TriggerLeft(AxisDirection),
    TriggerRight(AxisDirection),
}

impl SdlGamepadInput {
    #[must_use]
    pub fn from_sdl_button(button: sdl2::controller::Button) -> Self {
        use sdl2::controller::Button;

        match button {
            Button::A => Self::A,
            Button::B => Self::B,
            Button::X => Self::X,
            Button::Y => Self::Y,
            Button::Back => Self::Back,
            Button::Guide => Self::Guide,
            Button::Start => Self::Start,
            Button::LeftStick => Self::LeftStick,
            Button::RightStick => Self::RightStick,
            Button::LeftShoulder => Self::LeftShoulder,
            Button::RightShoulder => Self::RightShoulder,
            Button::DPadUp => Self::DPadUp,
            Button::DPadDown => Self::DPadDown,
            Button::DPadLeft => Self::DPadLeft,
            Button::DPadRight => Self::DPadRight,
            Button::Misc1 => Self::Misc1,
            Button::Paddle1 => Self::Paddle1,
            Button::Paddle2 => Self::Paddle2,
            Button::Paddle3 => Self::Paddle3,
            Button::Paddle4 => Self::Paddle4,
            Button::Touchpad => Self::Touchpad,
        }
    }

    #[must_use]
    pub fn from_sdl_axis(axis: sdl2::controller::Axis, value: i16) -> Self {
        use sdl2::controller::Axis;

        let direction = if value >= 0 { AxisDirection::Positive } else { AxisDirection::Negative };

        match axis {
            Axis::LeftX => Self::LeftX(direction),
            Axis::LeftY => Self::LeftY(direction),
            Axis::RightX => Self::RightX(direction),
            Axis::RightY => Self::RightY(direction),
            Axis::TriggerLeft => Self::TriggerLeft(direction),
            Axis::TriggerRight => Self::TriggerRight(direction),
        }
    }
}

impl Display for SdlGamepadInput {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::A => write!(f, "A"),
            Self::B => write!(f, "B"),
            Self::X => write!(f, "X"),
            Self::Y => write!(f, "Y"),
            Self::Back => write!(f, "Back"),
            Self::Guide => write!(f, "Guide"),
            Self::Start => write!(f, "Start"),
            Self::LeftStick => write!(f, "LeftStick"),
            Self::RightStick => write!(f, "RightStick"),
            Self::LeftShoulder => write!(f, "LeftShoulder"),
            Self::RightShoulder => write!(f, "RightShoulder"),
            Self::DPadUp => write!(f, "DPadUp"),
            Self::DPadDown => write!(f, "DPadDown"),
            Self::DPadLeft => write!(f, "DPadLeft"),
            Self::DPadRight => write!(f, "DPadRight"),
            Self::Misc1 => write!(f, "Misc1"),
            Self::Paddle1 => write!(f, "Paddle1"),
            Self::Paddle2 => write!(f, "Paddle2"),
            Self::Paddle3 => write!(f, "Paddle3"),
            Self::Paddle4 => write!(f, "Paddle4"),
            Self::Touchpad => write!(f, "Touchpad"),
            Self::LeftX(direction) => write!(f, "LeftX {direction}"),
            Self::LeftY(direction) => write!(f, "LeftY {direction}"),
            Self::RightX(direction) => write!(f, "RightX {direction}"),
            Self::RightY(direction) => write!(f, "RightY {direction}"),
            Self::TriggerLeft(direction) => write!(f, "TriggerLeft {direction}"),
            Self::TriggerRight(direction) => write!(f, "TriggerRight {direction}"),
        }
    }
}

impl FromStr for SdlGamepadInput {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "A" => Ok(Self::A),
            "B" => Ok(Self::B),
            "X" => Ok(Self::X),
            "Y" => Ok(Self::Y),
            "Back" => Ok(Self::Back),
            "Guide" => Ok(Self::Guide),
            "Start" => Ok(Self::Start),
            "LeftStick" => Ok(Self::LeftStick),
            "RightStick" => Ok(Self::RightStick),
            "LeftShoulder" => Ok(Self::LeftShoulder),
            "RightShoulder" => Ok(Self::RightShoulder),
            "DPadUp" => Ok(Self::DPadUp),
            "DPadDown" => Ok(Self::DPadDown),
            "DPadLeft" => Ok(Self::DPadLeft),
            "DPadRight" => Ok(Self::DPadRight),
            "Misc1" => Ok(Self::Misc1),
            "Paddle1" => Ok(Self::Paddle1),
            "Paddle2" => Ok(Self::Paddle2),
            "Paddle3" => Ok(Self::Paddle3),
            "Paddle4" => Ok(Self::Paddle4),
            "Touchpad" => Ok(Self::Touchpad),
            _ => match s.split_once(' ') {
                Some(("LeftX", direction)) => Ok(Self::LeftX(direction.parse()?)),
                Some(("LeftY", direction)) => Ok(Self::LeftY(direction.parse()?)),
                Some(("RightX", direction)) => Ok(Self::RightX(direction.parse()?)),
                Some(("RightY", direction)) => Ok(Self::RightY(direction.parse()?)),
                Some(("TriggerLeft", direction)) => Ok(Self::TriggerLeft(direction.parse()?)),
                Some(("TriggerRight", direction)) => Ok(Self::TriggerRight(direction.parse()?)),
                _ => Err(format!("Invalid SDL button/axis string: '{s}'")),
            },
        }
    }
}

impl Serialize for SdlGamepadInput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

struct GamepadInputDeserializeVisitor;

impl Visitor<'_> for GamepadInputDeserializeVisitor {
    type Value = SdlGamepadInput;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "String representing an SDL gamepad input")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: Error,
    {
        v.parse().map_err(|err| Error::custom(err))
    }
}

impl<'de> Deserialize<'de> for SdlGamepadInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(GamepadInputDeserializeVisitor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SingleInput {
    Keyboard { keycode: KeyCode },
    SdlGamepad { controller_idx: u32, sdl_input: SdlGamepadInput },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControllerConfig {
    pub d_pad_up: Option<SingleInput>,
    pub d_pad_down: Option<SingleInput>,
    pub d_pad_left: Option<SingleInput>,
    pub d_pad_right: Option<SingleInput>,
    pub cross: Option<SingleInput>,
    pub circle: Option<SingleInput>,
    pub square: Option<SingleInput>,
    pub triangle: Option<SingleInput>,
    pub l1: Option<SingleInput>,
    pub l2: Option<SingleInput>,
    pub r1: Option<SingleInput>,
    pub r2: Option<SingleInput>,
    pub start: Option<SingleInput>,
    pub select: Option<SingleInput>,
    pub analog: Option<SingleInput>,
    pub l_stick_up: Option<SingleInput>,
    pub l_stick_down: Option<SingleInput>,
    pub l_stick_left: Option<SingleInput>,
    pub l_stick_right: Option<SingleInput>,
    pub r_stick_up: Option<SingleInput>,
    pub r_stick_down: Option<SingleInput>,
    pub r_stick_left: Option<SingleInput>,
    pub r_stick_right: Option<SingleInput>,
    pub l3: Option<SingleInput>,
    pub r3: Option<SingleInput>,
    pub gamepad_axis_deadzone: i16,
    pub gamepad_trigger_threshold: i16,
}

const DEFAULT_TRIGGER_PRESS_THRESHOLD: i16 = i16::MAX / 4;

impl ControllerConfig {
    #[must_use]
    pub fn default_p1_keyboard() -> Self {
        Self {
            d_pad_up: Some(SingleInput::Keyboard { keycode: KeyCode::ArrowUp }),
            d_pad_down: Some(SingleInput::Keyboard { keycode: KeyCode::ArrowDown }),
            d_pad_left: Some(SingleInput::Keyboard { keycode: KeyCode::ArrowLeft }),
            d_pad_right: Some(SingleInput::Keyboard { keycode: KeyCode::ArrowRight }),
            cross: Some(SingleInput::Keyboard { keycode: KeyCode::KeyX }),
            circle: Some(SingleInput::Keyboard { keycode: KeyCode::KeyS }),
            square: Some(SingleInput::Keyboard { keycode: KeyCode::KeyZ }),
            triangle: Some(SingleInput::Keyboard { keycode: KeyCode::KeyA }),
            l1: Some(SingleInput::Keyboard { keycode: KeyCode::KeyW }),
            l2: Some(SingleInput::Keyboard { keycode: KeyCode::KeyQ }),
            r1: Some(SingleInput::Keyboard { keycode: KeyCode::KeyE }),
            r2: Some(SingleInput::Keyboard { keycode: KeyCode::KeyR }),
            start: Some(SingleInput::Keyboard { keycode: KeyCode::Enter }),
            select: Some(SingleInput::Keyboard { keycode: KeyCode::ShiftRight }),
            analog: Some(SingleInput::Keyboard { keycode: KeyCode::KeyT }),
            l_stick_up: None,
            l_stick_down: None,
            l_stick_left: None,
            l_stick_right: None,
            r_stick_up: None,
            r_stick_down: None,
            r_stick_left: None,
            r_stick_right: None,
            l3: None,
            r3: None,
            gamepad_axis_deadzone: 0,
            gamepad_trigger_threshold: DEFAULT_TRIGGER_PRESS_THRESHOLD,
        }
    }

    #[must_use]
    pub fn default_p1_gamepad() -> Self {
        Self {
            d_pad_up: sdl_input_idx_0(SdlGamepadInput::DPadUp),
            d_pad_down: sdl_input_idx_0(SdlGamepadInput::DPadDown),
            d_pad_left: sdl_input_idx_0(SdlGamepadInput::DPadLeft),
            d_pad_right: sdl_input_idx_0(SdlGamepadInput::DPadRight),
            cross: sdl_input_idx_0(SdlGamepadInput::A),
            circle: sdl_input_idx_0(SdlGamepadInput::B),
            square: sdl_input_idx_0(SdlGamepadInput::X),
            triangle: sdl_input_idx_0(SdlGamepadInput::Y),
            l1: sdl_input_idx_0(SdlGamepadInput::LeftShoulder),
            l2: sdl_input_idx_0(SdlGamepadInput::TriggerLeft(AxisDirection::Positive)),
            r1: sdl_input_idx_0(SdlGamepadInput::RightShoulder),
            r2: sdl_input_idx_0(SdlGamepadInput::TriggerRight(AxisDirection::Positive)),
            start: sdl_input_idx_0(SdlGamepadInput::Start),
            select: sdl_input_idx_0(SdlGamepadInput::Back),
            analog: sdl_input_idx_0(SdlGamepadInput::Guide),
            l_stick_up: sdl_input_idx_0(SdlGamepadInput::LeftY(AxisDirection::Negative)),
            l_stick_down: sdl_input_idx_0(SdlGamepadInput::LeftY(AxisDirection::Positive)),
            l_stick_left: sdl_input_idx_0(SdlGamepadInput::LeftX(AxisDirection::Negative)),
            l_stick_right: sdl_input_idx_0(SdlGamepadInput::LeftX(AxisDirection::Positive)),
            r_stick_up: sdl_input_idx_0(SdlGamepadInput::RightY(AxisDirection::Negative)),
            r_stick_down: sdl_input_idx_0(SdlGamepadInput::RightY(AxisDirection::Positive)),
            r_stick_left: sdl_input_idx_0(SdlGamepadInput::RightX(AxisDirection::Negative)),
            r_stick_right: sdl_input_idx_0(SdlGamepadInput::RightX(AxisDirection::Positive)),
            l3: sdl_input_idx_0(SdlGamepadInput::LeftStick),
            r3: sdl_input_idx_0(SdlGamepadInput::RightStick),
            gamepad_axis_deadzone: 0,
            gamepad_trigger_threshold: DEFAULT_TRIGGER_PRESS_THRESHOLD,
        }
    }

    #[must_use]
    pub fn none() -> Self {
        Self {
            d_pad_up: None,
            d_pad_down: None,
            d_pad_left: None,
            d_pad_right: None,
            cross: None,
            circle: None,
            square: None,
            triangle: None,
            l1: None,
            l2: None,
            r1: None,
            r2: None,
            start: None,
            select: None,
            analog: None,
            l_stick_up: None,
            l_stick_down: None,
            l_stick_left: None,
            l_stick_right: None,
            r_stick_up: None,
            r_stick_down: None,
            r_stick_left: None,
            r_stick_right: None,
            l3: None,
            r3: None,
            gamepad_axis_deadzone: 0,
            gamepad_trigger_threshold: DEFAULT_TRIGGER_PRESS_THRESHOLD,
        }
    }
}

#[allow(clippy::unnecessary_wraps)]
const fn sdl_input_idx_0(sdl_input: SdlGamepadInput) -> Option<SingleInput> {
    Some(SingleInput::SdlGamepad { controller_idx: 0, sdl_input })
}
