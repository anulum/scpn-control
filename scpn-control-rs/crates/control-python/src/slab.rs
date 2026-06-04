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

unsafe impl<const PACKET_SIZE: usize, const CAPACITY: usize> Sync
    for PacketSlab<PACKET_SIZE, CAPACITY>
{
}
