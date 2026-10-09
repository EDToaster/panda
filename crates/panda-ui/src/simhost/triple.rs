//! A lock-free single-producer single-consumer triple buffer.
//!
//! The writer always has a private slot to fill; [`Writer::publish`] swaps it with the shared
//! "middle" slot in one atomic operation and marks it fresh. The reader swaps the middle slot
//! with its own when it is fresh ([`Reader::read`]). Neither side ever waits for the other, and
//! slots are reused (no allocation per publication once their buffers have grown).

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

/// Bit set in the shared state when the middle slot holds an unread publication.
const FRESH: u8 = 0b100;
/// Mask of the middle slot index.
const INDEX: u8 = 0b011;

struct Shared<T> {
    slots: [UnsafeCell<T>; 3],
    /// Index of the middle slot | FRESH.
    state: AtomicU8,
}

// SAFETY: each slot is accessed by at most one side at a time: the writer owns `Writer::idx`,
// the reader owns `Reader::idx`, and the middle slot is only touched after an atomic swap moved
// it to one side (AcqRel orders the slot contents with the hand-over).
unsafe impl<T: Send> Sync for Shared<T> {}

/// Producer half.
pub struct Writer<T> {
    shared: Arc<Shared<T>>,
    idx: u8,
}

/// Consumer half.
pub struct Reader<T> {
    shared: Arc<Shared<T>>,
    idx: u8,
}

/// Create a triple buffer whose three slots start as clones of `init`.
pub fn triple_buffer<T: Clone + Send>(init: T) -> (Writer<T>, Reader<T>) {
    let shared = Arc::new(Shared {
        slots: [
            UnsafeCell::new(init.clone()),
            UnsafeCell::new(init.clone()),
            UnsafeCell::new(init),
        ],
        state: AtomicU8::new(1),
    });
    (
        Writer {
            shared: shared.clone(),
            idx: 0,
        },
        Reader { shared, idx: 2 },
    )
}

impl<T: Send> Writer<T> {
    /// The writer's private slot (holds an older publication: overwrite every field).
    pub fn slot(&mut self) -> &mut T {
        // SAFETY: slot `idx` belongs to the writer until the next `publish`.
        unsafe { &mut *self.shared.slots[self.idx as usize].get() }
    }

    /// Publish the private slot; the writer gets the previous middle slot back.
    pub fn publish(&mut self) {
        let old = self.shared.state.swap(self.idx | FRESH, Ordering::AcqRel);
        self.idx = old & INDEX;
    }
}

impl<T: Send> Reader<T> {
    /// The latest publication (or the previous one if nothing new was published). Never blocks.
    pub fn read(&mut self) -> &T {
        if self.shared.state.load(Ordering::Relaxed) & FRESH != 0 {
            let old = self.shared.state.swap(self.idx, Ordering::AcqRel);
            self.idx = old & INDEX;
        }
        // SAFETY: slot `idx` belongs to the reader until the next swap in `read`.
        unsafe { &*self.shared.slots[self.idx as usize].get() }
    }

    /// True if a publication arrived since the last [`Reader::read`].
    pub fn has_fresh(&self) -> bool {
        self.shared.state.load(Ordering::Acquire) & FRESH != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reader_sees_latest() {
        let (mut w, mut r) = triple_buffer(0u64);
        assert_eq!(*r.read(), 0);
        *w.slot() = 1;
        w.publish();
        *w.slot() = 2;
        w.publish();
        assert!(r.has_fresh());
        assert_eq!(*r.read(), 2);
        assert!(!r.has_fresh());
        assert_eq!(*r.read(), 2);
    }

    #[test]
    fn concurrent_publications_are_consistent_and_monotone() {
        // Each publication is a vector whose elements all equal its sequence number.
        let (mut w, mut r) = triple_buffer(vec![0u64; 64]);
        let t = std::thread::spawn(move || {
            for i in 1..=200_000u64 {
                w.slot().fill(i);
                w.publish();
            }
        });
        let mut last = 0;
        while last < 200_000 {
            let v = r.read();
            let first = v[0];
            assert!(v.iter().all(|&x| x == first), "torn read");
            assert!(first >= last, "went backwards");
            last = first;
        }
        t.join().unwrap();
    }
}
