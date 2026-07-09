// SPDX-License-Identifier: AGPL-3.0-or-later
// ──────────────────────────────────────────────────────────────────────
// SCPN Control — Zero-Allocation Slab
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// ──────────────────────────────────────────────────────────────────────

use std::cell::UnsafeCell;
use std::ptr;
use std::sync::atomic::{fence, AtomicBool, Ordering};

/// Maximum inflight frames supported by the transport queue.
pub const SLAB_CAPACITY: usize = 128;

#[repr(C)]
pub struct PacketSlab<const PACKET_SIZE: usize, const CAPACITY: usize = SLAB_CAPACITY> {
    memory: UnsafeCell<[[u8; PACKET_SIZE]; CAPACITY]>,
    status_flags: [AtomicBool; CAPACITY],
}

impl<const PACKET_SIZE: usize, const CAPACITY: usize> PacketSlab<PACKET_SIZE, CAPACITY> {
    pub const fn new() -> Self {
        Self {
            memory: UnsafeCell::new([[0u8; PACKET_SIZE]; CAPACITY]),
            status_flags: [const { AtomicBool::new(false) }; CAPACITY],
        }
    }

    /// Writes `data` into a free slot and returns the slot index.
    pub fn lease_and_write(&self, data: &[u8]) -> Option<usize> {
        if data.len() != PACKET_SIZE {
            return None;
        }

        for i in 0..CAPACITY {
            if self.status_flags[i]
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                // SAFETY: the compare_exchange above claimed slot `i` exclusively
                // (false -> true, AcqRel), so this thread has sole access to its
                // PACKET_SIZE bytes. `i < CAPACITY` and `data.len() == PACKET_SIZE`
                // (both checked), so `dest = base + i * PACKET_SIZE` and the
                // PACKET_SIZE-byte copy stay within the contiguous
                // CAPACITY * PACKET_SIZE backing array; source and dest never overlap.
                unsafe {
                    let dest = self.memory.get().cast::<u8>().add(i * PACKET_SIZE);
                    ptr::copy_nonoverlapping(data.as_ptr(), dest, PACKET_SIZE);
                }
                fence(Ordering::Release);
                return Some(i);
            }
        }
        None
    }

    pub fn get_ptr_for_io(&self, index: usize) -> Option<*const u8> {
        if index >= CAPACITY {
            return None;
        }

        // SAFETY: `index < CAPACITY` (checked above), so indexing the backing
        // array is in bounds. The pointer aliases slot `index`; the caller
        // contract (only queried for a slot it currently leases) keeps that slot
        // stable until released, so no concurrent writer touches those bytes.
        Some(unsafe { (*self.memory.get())[index].as_ptr() })
    }

    /// Releases a leased buffer slot.
    pub fn release(&self, index: usize) {
        if index < CAPACITY {
            self.status_flags[index].store(false, Ordering::Release);
        }
    }

    #[allow(dead_code)]
    pub fn in_use_count(&self) -> usize {
        self.status_flags
            .iter()
            .filter(|flag| flag.load(Ordering::Acquire))
            .count()
    }
}

// SAFETY: every access to the `UnsafeCell` payload is gated by the per-slot
// `status_flags` atomics acting as a lock. A slot's bytes are written only after
// its flag is claimed (compare_exchange false -> true, AcqRel) and read only via a
// pointer handed out for a slot the caller still leases, with a Release fence after
// each write and Acquire on each claim. No two threads ever touch the same slot's
// bytes concurrently, so sharing `&PacketSlab` across threads is data-race-free.
unsafe impl<const PACKET_SIZE: usize, const CAPACITY: usize> Sync
    for PacketSlab<PACKET_SIZE, CAPACITY>
{
}
