//! Pure controller logic of the run controls and the clickable inputs (no egui, no GPU).
//!
//! The toolbar ([`super::toolbar`]) turns user gestures into [`ToolbarAction`]s and
//! [`SimControl::commands`] maps them to [`Command`]s for the sim thread. Nothing here waits on
//! the simulation: every action is a non-blocking `SimHost::send`.
//!
//! [`InputModel`] remembers input values that were sent but not yet confirmed by a snapshot, so
//! clicking an input twice within one publication interval toggles it twice.

use std::collections::HashMap;

use panda_kernel::{Mode, StepUnit};
use panda_netlist::{Logic, Value};

use crate::simhost::{Command, Speed};

/// A gesture on the run controls.
#[derive(Clone, Debug, PartialEq)]
pub enum ToolbarAction {
    /// Select Timing or Fast mode.
    SetMode(Mode),
    /// Start running continuously.
    Run,
    /// Stop running.
    Pause,
    /// Run if paused, pause if running.
    ToggleRun,
    /// Step by the configured count and unit.
    Step,
    /// Step `step_n` units of a specific kind (tick / phase / cycle buttons); also makes it the
    /// unit of [`ToolbarAction::Step`].
    StepBy(StepUnit),
    /// Switch between unlimited and the target rate.
    SetLimited(bool),
    /// Change the target rate (cycles per second).
    SetRate(f64),
    /// Turn the limit on at this rate (a preset of the rate menu).
    LimitTo(f64),
    /// Power-on reset.
    Reset,
}

/// Run-control state kept by the UI (what the toolbar shows between snapshots).
#[derive(Clone, Debug, PartialEq)]
pub struct SimControl {
    /// Unit of the "Step" button.
    pub step_unit: StepUnit,
    /// Count of the "Step" button.
    pub step_n: u64,
    /// The speed limit is on.
    pub limited: bool,
    /// Target rate (cycles per second, phases for clockless designs) while limited.
    pub rate: f64,
}

impl Default for SimControl {
    fn default() -> Self {
        SimControl {
            step_unit: StepUnit::Cycle,
            step_n: 1,
            limited: false,
            rate: 10.0,
        }
    }
}

/// Bounds of the target rate.
pub const RATE_RANGE: (f64, f64) = (0.1, 1e9);

/// Presets of the rate menu (cycles per second).
pub const RATE_PRESETS: [f64; 7] = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6];

impl SimControl {
    /// The speed that follows from `limited` / `rate`.
    pub fn speed(&self) -> Speed {
        if self.limited {
            Speed::CyclesPerSec(self.rate)
        } else {
            Speed::Unlimited
        }
    }

    /// Adopt the speed reported by a snapshot (e.g. set by another panel or a test).
    pub fn sync_speed(&mut self, s: Speed) {
        match s {
            Speed::Unlimited => self.limited = false,
            Speed::CyclesPerSec(r) => {
                self.limited = true;
                self.rate = r;
            }
        }
    }

    /// A step of `step_n` `unit`s. Stepping implies pausing: a step while running would be lost
    /// in the run.
    fn step(&self, unit: StepUnit, running: bool) -> Vec<Command> {
        let step = Command::Step {
            n: self.step_n.max(1),
            unit,
        };
        if running {
            vec![Command::Pause, step]
        } else {
            vec![step]
        }
    }

    /// The commands for `action`, given the current snapshot's `mode` and `running` flag.
    /// Updates the control state (step unit, speed) as a side effect.
    pub fn commands(&mut self, action: ToolbarAction, mode: Mode, running: bool) -> Vec<Command> {
        match action {
            ToolbarAction::SetMode(m) if m == mode => Vec::new(),
            ToolbarAction::SetMode(m) => {
                let mut v = vec![Command::SetMode(m)];
                if self.limited {
                    // Restart the pacing schedule: cycles the slower mode could not reach must
                    // not be caught up in a burst by the faster one.
                    v.push(Command::SetSpeed(self.speed()));
                }
                v
            }
            ToolbarAction::Run if running => Vec::new(),
            ToolbarAction::Run => vec![Command::Run],
            ToolbarAction::Pause if !running => Vec::new(),
            ToolbarAction::Pause => vec![Command::Pause],
            ToolbarAction::ToggleRun => {
                vec![if running {
                    Command::Pause
                } else {
                    Command::Run
                }]
            }
            ToolbarAction::Step => self.step(self.step_unit, running),
            ToolbarAction::StepBy(unit) => {
                self.step_unit = unit;
                self.step(unit, running)
            }
            ToolbarAction::SetLimited(on) => {
                self.limited = on;
                vec![Command::SetSpeed(self.speed())]
            }
            ToolbarAction::SetRate(r) => {
                self.rate = r.clamp(RATE_RANGE.0, RATE_RANGE.1);
                if self.limited {
                    vec![Command::SetSpeed(self.speed())]
                } else {
                    Vec::new()
                }
            }
            ToolbarAction::LimitTo(r) => {
                self.limited = true;
                self.rate = r.clamp(RATE_RANGE.0, RATE_RANGE.1);
                vec![Command::SetSpeed(self.speed())]
            }
            ToolbarAction::Reset => vec![Command::Reset],
        }
    }
}

/// Human-readable rate: `"1.70 M"`, `"12.3 k"`, `"950"`.
pub fn si(v: f64) -> String {
    let a = v.abs();
    if a >= 1e9 {
        format!("{:.2} G", v / 1e9)
    } else if a >= 1e6 {
        format!("{:.2} M", v / 1e6)
    } else if a >= 1e4 {
        format!("{:.1} k", v / 1e3)
    } else if a >= 100.0 || a == 0.0 {
        format!("{v:.0}")
    } else {
        format!("{v:.1}")
    }
}

/// Values of top-level inputs as the user last set them (until a snapshot confirms them).
#[derive(Clone, Debug, Default)]
pub struct InputModel {
    sent: HashMap<String, Value>,
}

impl InputModel {
    /// The value to show / toggle from: the last sent one, else the live one.
    pub fn current(&self, pin: &str, live: Option<&Value>) -> Option<Value> {
        self.sent.get(pin).cloned().or_else(|| live.cloned())
    }

    /// Forget a sent value once the live value shows it.
    pub fn confirm(&mut self, pin: &str, live: Option<&Value>) {
        if let Some(l) = live
            && self.sent.get(pin) == Some(l)
        {
            self.sent.remove(pin);
        }
    }

    /// The value sent to `pin` that the live value does not show yet.
    pub fn pending(&self, pin: &str, live: Option<&Value>) -> Option<&Value> {
        self.sent.get(pin).filter(|v| live != Some(*v))
    }

    /// Sent values awaiting confirmation.
    pub fn sent(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.sent.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// True if no sent value awaits confirmation.
    pub fn is_empty(&self) -> bool {
        self.sent.is_empty()
    }

    /// Forget everything (new design).
    pub fn clear(&mut self) {
        self.sent.clear();
    }

    /// Record `value` for `pin` and return the command setting it.
    pub fn set(&mut self, pin: &str, value: Value) -> Command {
        self.sent.insert(pin.to_string(), value.clone());
        Command::SetInput {
            pin: pin.to_string(),
            value,
        }
    }

    /// Toggle a 1-bit input: `1 → 0`, anything else (`0`, `X`, `Z`, unknown) `→ 1`.
    pub fn toggle(&mut self, pin: &str, live: Option<&Value>) -> Command {
        let cur = self.current(pin, live);
        let one = cur.is_some_and(|v| v.width() == 1 && v.get(0) == Logic::One);
        self.set(pin, Value::from_bool(!one))
    }
}

/// Parse the text of the multi-bit input editor: decimal, `0x…`, `0b…` (with `x`/`z`), a lone
/// `x` / `z`, or a negative decimal (two's complement in `width` bits).
pub fn parse_input(width: u32, text: &str) -> Result<Value, String> {
    let t = text.trim();
    if let Some(neg) = t.strip_prefix('-') {
        let n: u64 = neg
            .trim()
            .parse()
            .map_err(|e| format!("bad number '{t}': {e}"))?;
        if width == 0 || width > 64 || (width < 64 && n > 1u64 << (width - 1)) {
            return Err(format!("{t} does not fit in {width} signed bits"));
        }
        let mask = if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        return Ok(Value::from_u64(width, n.wrapping_neg() & mask));
    }
    Value::parse(width, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dbg(v: &[Command]) -> Vec<String> {
        v.iter().map(|c| format!("{c:?}")).collect()
    }

    #[test]
    fn toolbar_actions_map_to_commands() {
        let mut c = SimControl::default();
        let m = Mode::Timing;
        assert!(
            c.commands(ToolbarAction::SetMode(Mode::Timing), m, false)
                .is_empty()
        );
        assert_eq!(
            dbg(&c.commands(ToolbarAction::SetMode(Mode::Fast), m, false)),
            ["SetMode(Fast)"]
        );
        assert_eq!(dbg(&c.commands(ToolbarAction::Run, m, false)), ["Run"]);
        assert!(c.commands(ToolbarAction::Run, m, true).is_empty());
        assert_eq!(
            dbg(&c.commands(ToolbarAction::ToggleRun, m, true)),
            ["Pause"]
        );
        assert_eq!(
            dbg(&c.commands(ToolbarAction::ToggleRun, m, false)),
            ["Run"]
        );
        assert!(c.commands(ToolbarAction::Pause, m, false).is_empty());
        // Stepping while running pauses first.
        c.step_n = 5;
        assert_eq!(
            dbg(&c.commands(ToolbarAction::Step, m, true)),
            ["Pause", "Step(5, Cycle)"]
        );
        assert_eq!(
            dbg(&c.commands(ToolbarAction::StepBy(StepUnit::Tick), m, false)),
            ["Step(5, Tick)"]
        );
        assert_eq!(c.step_unit, StepUnit::Tick);
        c.step_n = 0; // clamped to 1
        assert_eq!(
            dbg(&c.commands(ToolbarAction::StepBy(StepUnit::Phase), m, false)),
            ["Step(1, Phase)"]
        );
        assert_eq!(dbg(&c.commands(ToolbarAction::Reset, m, false)), ["Reset"]);
    }

    #[test]
    fn speed_limit_is_paced_by_the_sim_thread() {
        let mut c = SimControl::default();
        // Changing the rate while unlimited sends nothing.
        assert!(
            c.commands(ToolbarAction::SetRate(1000.0), Mode::Fast, true)
                .is_empty()
        );
        assert_eq!(
            dbg(&c.commands(ToolbarAction::SetLimited(true), Mode::Fast, true)),
            ["SetSpeed(CyclesPerSec(1000.0))"]
        );
        assert_eq!(
            dbg(&c.commands(ToolbarAction::SetRate(0.0), Mode::Fast, true)),
            ["SetSpeed(CyclesPerSec(0.1))"]
        );
        assert_eq!(
            dbg(&c.commands(ToolbarAction::SetLimited(false), Mode::Fast, true)),
            ["SetSpeed(Unlimited)"]
        );
        assert_eq!(
            dbg(&c.commands(ToolbarAction::LimitTo(1e5), Mode::Fast, true)),
            ["SetSpeed(CyclesPerSec(100000.0))"]
        );
        assert!(c.limited && c.rate == 1e5);
        // A mode switch while limited restarts the pacing schedule.
        assert_eq!(
            dbg(&c.commands(ToolbarAction::SetMode(Mode::Timing), Mode::Fast, true)),
            ["SetMode(Timing)", "SetSpeed(CyclesPerSec(100000.0))"]
        );
        assert_eq!(
            dbg(&c.commands(ToolbarAction::SetLimited(false), Mode::Timing, true)),
            ["SetSpeed(Unlimited)"]
        );
        assert_eq!(
            dbg(&c.commands(ToolbarAction::SetMode(Mode::Fast), Mode::Timing, true)),
            ["SetMode(Fast)"]
        );
        c.sync_speed(Speed::CyclesPerSec(42.0));
        assert!(c.limited && c.rate == 42.0);
        c.sync_speed(Speed::Unlimited);
        assert!(!c.limited);
    }

    #[test]
    fn inputs_toggle_and_confirm() {
        let mut m = InputModel::default();
        let zero = Value::from_bool(false);
        let one = Value::from_bool(true);
        // Unknown → 1.
        assert_eq!(
            format!("{:?}", m.toggle("EN", None)),
            "SetInput(EN, Value(1'0b1))"
        );
        // Toggling again before the snapshot shows it uses the sent value.
        assert_eq!(
            format!("{:?}", m.toggle("EN", Some(&zero))),
            "SetInput(EN, Value(1'0b0))"
        );
        assert_eq!(m.current("EN", Some(&one)), Some(zero.clone()));
        assert_eq!(m.pending("EN", Some(&one)), Some(&zero));
        assert_eq!(m.pending("EN", Some(&zero)), None);
        m.confirm("EN", Some(&zero));
        assert!(m.is_empty());
        assert_eq!(m.current("EN", Some(&one)), Some(one.clone()));
        // X → 1.
        let x = Value::xs(1);
        assert_eq!(
            format!("{:?}", m.toggle("A", Some(&x))),
            "SetInput(A, Value(1'0b1))"
        );
    }

    #[test]
    fn multi_bit_input_parsing() {
        assert_eq!(parse_input(8, "0x2a").unwrap().to_u64(), Some(42));
        assert_eq!(parse_input(8, " 200 ").unwrap().to_u64(), Some(200));
        assert_eq!(parse_input(8, "-1").unwrap().to_u64(), Some(255));
        assert_eq!(parse_input(8, "-128").unwrap().to_u64(), Some(128));
        assert!(parse_input(8, "-129").is_err());
        assert!(parse_input(8, "256").is_err());
        assert_eq!(parse_input(4, "0b1x0z").unwrap().to_string(), "0b1x0z");
        assert!(parse_input(4, "hello").is_err());
    }

    #[test]
    fn si_formatting() {
        assert_eq!(si(1_703_000.0), "1.70 M");
        assert_eq!(si(12_345.0), "12.3 k");
        assert_eq!(si(950.0), "950");
        assert_eq!(si(2.5), "2.5");
        assert_eq!(si(0.0), "0");
    }
}
