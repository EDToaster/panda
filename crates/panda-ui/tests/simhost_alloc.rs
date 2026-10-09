//! Allocation discipline of the sim thread: publishing snapshots (and shadow-capture
//! checkpoints) must not allocate per publication once buffers have grown.
//!
//! A counting global allocator keeps a per-thread count; a [`Command::Custom`] closure reads the
//! sim thread's own counter.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::{Duration, Instant};

use panda_kernel::Mode;
use panda_netlist::flat::NetIdx;
use panda_ui::app::resolve_net;
use panda_ui::examples;
use panda_ui::simhost::{Command, SimEvent, SimHost, SimOptions, WatchSet, WaveConfig, WaveSignal};

struct Counting;

thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

fn bump() {
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
}

// SAFETY: forwards to the system allocator; the counter is a const-initialized thread-local
// without destructor (no allocation, no reentrancy).
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The sim thread's allocation count and the publication counter at that moment.
fn sim_allocs(h: &mut SimHost) -> (u64, u64) {
    h.send(Command::Custom(Box::new(|_| {
        Some(SimEvent::Custom(Box::new(ALLOCS.with(Cell::get))))
    })));
    let t0 = Instant::now();
    loop {
        assert!(t0.elapsed() < Duration::from_secs(10), "no answer");
        if let Some(SimEvent::Custom(b)) = h.wait_event(Duration::from_millis(10)) {
            let n = *b.downcast::<u64>().unwrap();
            return (n, h.snapshot().seq);
        }
    }
}

fn wait<F: FnMut(&panda_ui::simhost::Snapshot) -> bool>(h: &mut SimHost, mut f: F) {
    let t0 = Instant::now();
    while !f(h.snapshot()) {
        assert!(t0.elapsed() < Duration::from_secs(120), "timeout");
        h.poll_events();
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Run Panda-8 flat out for a second and return (sim-thread allocations, publications).
fn measure(waves: bool) -> (u64, u64) {
    let mut h = SimHost::spawn(SimOptions::default());
    h.send(Command::Load(Box::new(examples::panda8())));
    let (g, flat) = loop {
        if let Some(SimEvent::Netlist { design_gen, flat }) = h.wait_event(Duration::from_secs(30))
        {
            break (design_gen, flat);
        }
    };
    h.send(Command::SetMode(Mode::Fast));
    wait(&mut h, |s| s.jit);
    h.send(Command::SetWatch(WatchSet {
        id: 1,
        design_gen: g,
        nets: (0..flat.num_nets as NetIdx).collect(),
    }));
    if waves {
        let signals = ["PC", "IR", "OUT", "R0", "FLAGS", "PHASE"]
            .iter()
            .map(|n| WaveSignal::new(&flat, *n, resolve_net(&flat, n).unwrap()))
            .collect();
        h.send(Command::SetWave(WaveConfig {
            id: 2,
            design_gen: g,
            signals,
        }));
    }
    h.send(Command::Run);
    wait(&mut h, |s| s.running && s.watch_id == 1);
    // Warm up: buffers grow, the state pool fills.
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_millis(500) {
        h.poll_events();
        std::thread::sleep(Duration::from_millis(16));
    }
    let (a0, p0) = sim_allocs(&mut h);
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(1) {
        h.poll_events();
        std::thread::sleep(Duration::from_millis(16));
    }
    let (a1, p1) = sim_allocs(&mut h);
    (a1 - a0, p1 - p0)
}

#[test]
fn publishing_does_not_allocate() {
    let (allocs, pubs) = measure(false);
    eprintln!("no waves: {allocs} sim-thread allocations over {pubs} publications");
    assert!(pubs > 20, "{pubs} publications");
    // The probe itself allocates a few (its answer event); nothing per publication.
    assert!(allocs <= 4 + pubs / 50, "{allocs} allocations for {pubs} publications");
}

#[test]
fn shadow_checkpoints_allocate_at_most_one_small_block() {
    let (allocs, pubs) = measure(true);
    eprintln!("shadow capture: {allocs} sim-thread allocations over {pubs} publications");
    assert!(pubs > 20, "{pubs} publications");
    // One `Arc` header per checkpoint (the state buffer itself comes from the pool) plus the
    // channel's amortized block allocation.
    assert!(
        allocs <= 8 + pubs + pubs / 8,
        "{allocs} allocations for {pubs} publications"
    );
}
