# Synthetic scalar controller benchmark

This benchmark compares controller responses on the scalar model
`dx/dt = -0.5*x + u + d`, starting each episode at zero. State/reference use arbitrary state units; input and disturbance use state
units per second, with decay coefficient 0.5/s. It does not simulate a tokamak, validate
a facility, or establish a real-time guarantee. The historical `PIDWrapper`
implements PI control; it has no derivative term or anti-windup.

## Discrete metric contract

For interval `k`, the controller receives state `x[k]` and reference `r[k]` at
`t[k] = k*dt`. Its applied output advances the plant by forward Euler:
`x[k+1] = x[k] + dt*(-0.5*x[k] + u[k] + d[k])`.
The reported error is `e[k] = x[k+1] - r[k]`. Over an episode:

- IAE is `sum(abs(e[k])*dt)`, in state units times seconds.
- ISE is `sum(e[k]**2*dt)`, in squared state units times seconds.
- ITAE is `sum((k+1)*dt*abs(e[k])*dt)`, using right-endpoint seconds.
- Control effort is `sum(u[k]**2*dt)` using the applied, possibly clipped input.
- Violations count post-step samples with state greater than 3.5. This is an
  illustrative scalar limit, not a plasma beta constraint.
- Computation time measures only the actual controller call using the host's
  monotonic performance counter. It excludes plant integration and report
  generation, and is not a deployment latency bound.

Squared integrals scale each sample by `sqrt(dt)` before squaring; absolute
errors are weighted before summation. This avoids overflow or underflow in
unweighted intermediate squares when the weighted result is representable.
Unrepresentable final metrics still abort the run.

Integral metrics and measured call time are averaged over episodes. Violations
are **summed** over episodes, preserving each observed violation. Time grids
must contain an integral positive number of intervals; a fractional tail is
refused rather than silently dropped. Grid resolution and explicit Euler can
change the result; no convergence or stability certificate is implied.

## Final-reference response

Response metrics use only the last held-reference segment. The final step
amplitude is the final reference minus the preceding sampled reference; when
there is no change, the preceding reference is the initial state, zero.
Overshoot is the maximum nonnegative excursion beyond the final reference in
the direction of that step, divided by its absolute amplitude, times 100.
A downward step therefore measures excursion below the target. A return to
zero remains well-defined; a zero-amplitude percentage is JSON `null`.

The settling band is 2% of the absolute final step amplitude. The settling time
is the first **post-step sample** inside that band that remains inside through
the end of the recorded segment, measured from the final reference change.
Even immediately successful tracking is first observed at `dt`, not zero.
With zero amplitude, the band has zero width. An episode that remains outside
at its last sample has no observed settling time. The aggregate time is `null`
if any episode does not settle; `settled_episodes` records how many did.
An observed settling time is limited by the finite sampled window; it does not
prove continued settling beyond the observation horizon.

## State, inputs and evidence

Every run samples each scenario reference once and shares those values across
controllers. Callbacks should be pure and return finite real scalars. Each
controller resets before every episode and receives private shape-(1,) input
snapshots. Output must be a finite real shape-(1,) vector; broadcasting is
refused. Invalid inputs or arithmetic abort the run without publishing partial
results. Controller side effects cannot be rolled back.

A successful run replaces the runner's result list and returns a separate list
of frozen result records. The runner and controllers are sequential objects;
do not use them concurrently. Seeded observation noise is local to the scenario:
each controller receives the same sequence, without consuming NumPy's global
random state. Host timing remains nondeterministic.

JSON remains a list of result objects. Each object declares
`synthetic-control-benchmark/v2`, the time grid, episode count and seed, with
fixed synthetic/non-facility/non-real-time markers. Consumers must interpret
the version: legacy metrics used start-time ITAE, truncated mean violation
counts, and zero-valued unobserved settling. Those values are not interchangeable
with v2 and old reports must not be silently relabelled.

`save_json` and `save_markdown` enforce the existing recorded-campaign custody
requirement before writing persistent repository evidence. Scratch exports do
not grant evidence authority. Writers replace existing files; filesystem errors
propagate. See the [recorded benchmark workflow](benchmarks.md).

## Native API and executable example

The example below runs against the real public runner in the dedicated test
module. The reference renderer reads these owning docstrings directly.

::: validation.control_benchmark_suite
    options:
      members:
        - BenchmarkScenario
        - ControllerWrapper
        - BenchmarkResults
        - BenchmarkRunner
        - setpoint_tracking
        - PIDWrapper
