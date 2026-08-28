/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Control — Pulsed scheduler liveness proof.
-/

namespace SCPNControl.PulsedFSM

/-!
# Pulsed scheduler finite-state model

This module proves local transition and eight-step-cycle properties for the
declared scheduler. The proofs cover only this finite-state abstraction; they
do not establish actuator timing, plant safety, or plasma stability.
-/

/-- The eight ordered phases in one abstract pulsed-control cycle. -/
inductive State where
  | idle
  | rampUp
  | flatTop
  | burn
  | expansion
  | dump
  | recharge
  | coolDown
  deriving DecidableEq, Repr

/-- Return the sole admitted successor of a scheduler state. -/
def next : State -> State
  | State.idle => State.rampUp
  | State.rampUp => State.flatTop
  | State.flatTop => State.burn
  | State.burn => State.expansion
  | State.expansion => State.dump
  | State.dump => State.recharge
  | State.recharge => State.coolDown
  | State.coolDown => State.idle

/-- Map each phase to its zero-based position in the abstract cycle. -/
def actionRank : State -> Nat
  | State.idle => 0
  | State.rampUp => 1
  | State.flatTop => 2
  | State.burn => 3
  | State.expansion => 4
  | State.dump => 5
  | State.recharge => 6
  | State.coolDown => 7

/-- Apply `next` exactly `n` times to `state`. -/
def stepN : Nat -> State -> State
  | 0, state => state
  | Nat.succ n, state => stepN n (next state)

/-- State that `toState` is exactly the declared successor of `fromState`. -/
def legalTransition (fromState toState : State) : Prop :=
  next fromState = toState

/-- Two legal successors of the same state are equal. -/
theorem adjacent_transition_deterministic
    (fromState toState candidateState : State)
    (h₁ : legalTransition fromState toState)
    (h₂ : legalTransition fromState candidateState) :
    toState = candidateState := by
  unfold legalTransition at h₁ h₂
  rw [← h₁, h₂]

/-- Every declared state reaches `idle` in at most eight abstract steps. -/
theorem pulsed_fsm_eventually_returns_to_idle (state : State) :
    ∃ n : Nat, n ≤ 8 ∧ stepN n state = State.idle := by
  cases state with
  | idle =>
      exact ⟨0, by decide, by rfl⟩
  | rampUp =>
      exact ⟨7, by decide, by rfl⟩
  | flatTop =>
      exact ⟨6, by decide, by rfl⟩
  | burn =>
      exact ⟨5, by decide, by rfl⟩
  | expansion =>
      exact ⟨4, by decide, by rfl⟩
  | dump =>
      exact ⟨3, by decide, by rfl⟩
  | recharge =>
      exact ⟨2, by decide, by rfl⟩
  | coolDown =>
      exact ⟨1, by decide, by rfl⟩

/-- Starting at `idle`, eight abstract steps complete one full cycle. -/
theorem idle_returns_to_idle_after_full_cycle :
    stepN 8 State.idle = State.idle := by
  rfl

/-- The declared transition relation forbids a direct `idle` to `burn` jump. -/
theorem manual_transition_cannot_skip_burn_from_idle :
    ¬ legalTransition State.idle State.burn := by
  intro h
  unfold legalTransition at h
  cases h

end SCPNControl.PulsedFSM
