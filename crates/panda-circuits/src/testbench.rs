//! Testbench helpers: wrap a DUT in a top def that supplies `Clock` / `PowerOnReset` sources.

use panda_netlist::{DefBuilder, DefRef, DesignBuilder, Dir, Sig};

/// Parameters of [`testbench_with`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestbenchOptions {
    /// Clock period in ticks (`≥ 2`).
    pub period: u32,
    /// Ticks the clock is high per period (`0 < high < period`).
    pub high: u32,
    /// Tick of the first rising edge (`< period`). Rising edges are at `phase + k * period`.
    pub phase: u32,
    /// `PowerOnReset` duration: the reset pin is 1 for ticks `0..reset_ticks`.
    pub reset_ticks: u32,
    /// Name of the DUT's clock input pin.
    pub clock_pin: String,
    /// Name of the DUT's reset input pin.
    pub reset_pin: String,
}

impl TestbenchOptions {
    /// Defaults for a clock period: 50 % duty, first rising edge at `period / 2` (so the clock
    /// starts low), and reset held for two full cycles, released a quarter period after a falling
    /// edge (`2 * period + period / 4`), i.e. during a low phase — away from any edge. Pins `CLK`
    /// and `RST`.
    pub fn new(period: u32) -> Self {
        assert!(period >= 4, "TestbenchOptions: period must be ≥ 4");
        TestbenchOptions {
            period,
            high: period / 2,
            phase: period / 2,
            reset_ticks: 2 * period + period / 4,
            clock_pin: "CLK".into(),
            reset_pin: "RST".into(),
        }
    }

    /// Tick of the `k`-th rising clock edge (`k = 0` is the first).
    pub fn rising_edge(&self, k: u64) -> u64 {
        u64::from(self.phase) + k * u64::from(self.period)
    }

    /// Index of the first rising edge after the reset is released (the first edge at which a
    /// synchronously reset design leaves reset).
    pub fn first_edge_after_reset(&self) -> u64 {
        let r = u64::from(self.reset_ticks);
        let p = u64::from(self.phase);
        // An edge at the release tick itself still samples the reset value.
        if r < p { 0 } else { (r - p) / u64::from(self.period) + 1 }
    }
}

/// [`testbench_with`] using [`TestbenchOptions::new(clock_period)`](TestbenchOptions::new).
pub fn testbench(db: &mut DesignBuilder, dut: &DefRef, clock_period: u32) -> DefRef {
    testbench_with(db, dut, &TestbenchOptions::new(clock_period))
}

/// Wrap `dut` (instance id `dut`) in a top-level def suitable for [`DesignBuilder::build`].
///
/// Id `"tb_{dut}_p{period}_h{high}_ph{phase}_r{reset_ticks}"`, with `"_c{clock_pin}_rp{reset_pin}"`
/// appended when the pin names differ from the defaults `CLK` / `RST`. For every DUT pin, in order:
/// * the `In` pin named `opts.clock_pin` is driven by a `Clock` source (net `clk`);
/// * the `In` pin named `opts.reset_pin` is driven by a `PowerOnReset` source (net `rst`);
/// * every other `In` pin becomes a top-level `In` pin of the same name (user-settable);
/// * every `Out` pin becomes a top-level `Out` pin of the same name;
/// * every `InOut` pin becomes a top-level `InOut` pin of the same name.
///
/// A DUT without a clock (reset) pin simply gets no clock (reset) source.
pub fn testbench_with(db: &mut DesignBuilder, dut: &DefRef, opts: &TestbenchOptions) -> DefRef {
    let TestbenchOptions { period, high, phase, reset_ticks, .. } = *opts;
    assert!(period >= 2 && high > 0 && high < period && phase < period, "testbench: bad clock parameters");
    let mut id = format!("tb_{}_p{period}_h{high}_ph{phase}_r{reset_ticks}", dut.id);
    // Non-default pin choices change the wiring, so they must be part of the id (otherwise a
    // second testbench of the same DUT would silently reuse the first one's def).
    if opts.clock_pin != "CLK" || opts.reset_pin != "RST" {
        id.push_str(&format!("_c{}_rp{}", opts.clock_pin, opts.reset_pin));
    }
    if let Some(r) = db.get(&id) {
        return r;
    }
    let mut b = DefBuilder::new(&id);
    let mut ins: Vec<Sig> = Vec::new();
    for p in dut.pins.iter().filter(|p| p.dir != Dir::Out) {
        let s = match p.dir {
            Dir::In if p.name == opts.clock_pin && p.width == 1 => {
                let c = b.clock(period, high, phase);
                b.name(c, "clk")
            }
            Dir::In if p.name == opts.reset_pin && p.width == 1 => {
                let r = b.power_on_reset(reset_ticks);
                b.name(r, "rst")
            }
            Dir::In => b.input(&p.name, p.width),
            _ => b.inout(&p.name, p.width),
        };
        ins.push(s);
    }
    let outs = b.inst_named("dut", dut, &ins);
    for (p, s) in dut.pins.iter().filter(|p| p.dir == Dir::Out).zip(outs) {
        b.output(&p.name, s);
    }
    db.add(b.finish())
}
