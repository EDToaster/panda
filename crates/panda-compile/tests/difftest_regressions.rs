//! Regressions found by the panda-difftest differential fuzzer (FastRef vs Timing mode).

use std::sync::Arc;

use panda_compile::{FastOptions, FastRef, KernelKind};
use panda_netlist::{DefBuilder, DesignBuilder, Logic, Termination, Value, flatten};
use panda_timing::{TimingOptions, TimingSim};

/// A keeper-terminated net whose drivers are all constants is a Keeper kernel whose inputs never
/// change. Its only transition is the power-on one; with `old inputs = new inputs` the kernel
/// simulator kept the forced all-X state (a forced resolver output is only recomputed when one of
/// its drivers changes), so FastRef read `X` forever while Timing mode resolves the net to `1` at
/// power-on. (Minimized from a random design: tri-state bus drivers replaced by constants.)
#[test]
fn keeper_net_driven_by_constants_resolves_at_power_on() {
    for (drivers, want) in [
        (vec![1u64, 1, 1], Logic::One),
        (vec![0, 0], Logic::Zero),
        (vec![1, 0], Logic::X), // contention
    ] {
        let mut db = DesignBuilder::new();
        let mut b = DefBuilder::new("keeper_consts");
        let net = b.wire(1);
        b.terminate(net, Termination::Keeper);
        for &d in &drivers {
            let c = b.lit(1, d);
            b.connect(c, net);
        }
        let y = b.buf(net);
        b.output("Y", y);
        b.output("K", net);
        let top = db.add(b.finish());
        let flat = Arc::new(flatten(&db.build(&top)).expect("flatten"));
        let mut fr = FastRef::new(flat.clone(), FastOptions::default()).expect("FastRef");
        assert!(
            fr.kernels()
                .kernels
                .iter()
                .any(|k| k.kind == KernelKind::Keeper),
            "expected a Keeper kernel"
        );
        let mut sim = TimingSim::new(flat.clone(), TimingOptions::default());
        sim.settle(100);
        for pin in ["K", "Y"] {
            let t = sim.read_pin(pin).unwrap();
            let f = fr.read_pin(pin).unwrap();
            assert_eq!(t, Value::from_logic(want), "{drivers:?}: Timing {pin}");
            assert_eq!(f, t, "{drivers:?}: FastRef {pin} differs from Timing mode");
        }
        // Still correct after a reset.
        fr.reset();
        assert_eq!(fr.read_pin("Y").unwrap(), Value::from_logic(want));
    }
}
