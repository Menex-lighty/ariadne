# Post-mortem: the Kepler-9 fit — a wrong result *and* a wrong correction

A worked example of a subtle wrong result surviving several rounds of "honest
caveats," and then of the *first fix being wrong too*. The second point is the
more useful one: a plausible mechanism is not a verified mechanism, even when an
adversarial process produced it.

## What we reported vs what is true

For the Kepler-9 real-data anchor we fit a 2-D 3-body model to the observed TTVs
and reported **χ²/dof ≈ 598** (masses ~35/25 M⊕), building the conditioning
number, degeneracy scan, and prior analysis on top of it, with careful caveats
attributing the misfit to "2-D / near-resonant model inadequacy."

The truth: a proper multistart fit reaches **χ²/dof ≈ 180–192**, and the misfit
is **not** where we said. The real structure is a broad **mass-scale degeneracy**:
within 2× the best χ², **m_b ranges 15–75 M⊕ (a factor of 5)** while the **mass
ratio is pinned to ~2% (1.43–1.53)**. The local Fisher error (m_b ± 0.19 M⊕) is
*illusory precision* — the global valley is 15–75.

## Error #1 — the original: under-optimization

The Kepler-9 script ran **one `least_squares` from one cold start** (masses 43/30,
zero eccentricity phases). The repo's own `CLAUDE.md` says *"Nonconvex search…
never trust a single fit,"* and the M1 fitter uses multistart — we didn't apply it
here. That single fit stuck at χ²598. Multistart over several starts reaches
~180–192. The 598 was a bad local minimum, full stop.

## Error #2 — the correction that was also wrong

The *first* post-mortem (written overnight, via an adversarial two-agent debate)
concluded the 598 came from fitting the **wrong transit node**: the 2-D and 3-D
finders use opposite "in-front" sign conventions, and on a *fixed* trajectory that
flips χ² dramatically (180 vs 3298). That mechanism is real in isolation — but it
was **not the cause of our 598**, and blaming it was overreach.

The multistart re-run settles it: the **original node** (`front=+1`) reaches χ²192;
the other node reaches χ²579. So the good node was the one we already had; the 598
was under-optimization *at that node*, not a wrong-node artifact. The node
convention shifts χ² at the margins, but the 598→180 gap was optimization. The
3-D fit's 30/21 (χ²180) and the 2-D fit's 40/27 (χ²192) are simply two points in
the *same* mass-scale valley (same ratio ~1.47) — not two different nodes.

An adversarial debate correctly proved the two forward models were numerically
identical and that 180 was reachable — but then over-attributed the *why* to the
node, the most salient difference it had found. A salient mechanism that explains
the sign of an effect is not the same as the mechanism that produced the
magnitude.

## What is actually true (the corrected result)

- Best fit **χ²/dof ≈ 180–192** (multistart, both nodes searched, `front=+1` wins).
- **Mass ratio ≈ 1.47, constrained to ~2%** across the whole valley — the robust,
  well-identified combination.
- **Mass scale degenerate:** m_b anywhere in ~15–75 M⊕ at comparable χ² — the
  direction the TTVs barely constrain (conditioning correctly flags m_b+m_c moving
  together as the least-constrained direction).
- **Prior-aware:** at the full 71-transit baseline an external m_b prior buys
  almost nothing (data-only σ already tight *locally*); on sparse early data
  (~14 transits) a 20% prior collapses σ from ~30 to ~8 M⊕. The prior earns its
  keep only below ~28 transits.

So Kepler-9 reproduces the documented "illusory precision" honestly: a tight local
error and a tight *ratio*, over a mass *scale* that roams 5×.

## The lessons (the second is the point)

- **"It converged" ≠ "it's right."** 598 was a local minimum. Nonconvex fits need
  multistart — and we knew that, and skipped it for the one real case.
- **A verified *contradiction* is not a verified *cause*.** The debate proved 180
  was reachable (solid) and then named the node as the reason (wrong). Proving that
  a better answer exists is different from proving *why* the worse one happened.
  The fix needed its own re-run to correct.
- **Verification is iterative, not a single pass.** Each layer here (the caveats,
  then the node post-mortem, then the multistart) corrected the one before. Stop
  too early at any layer and you ship a confident wrong story.
- **Illusory precision is real:** trust the *global* scan (m_b 15–75) over the
  *local* Fisher (± 0.19), and report the constrained combination (the ratio), not
  the scale.

## Fix applied

- `find_transit_times` takes an explicit `front` (node) argument.
- `run_kepler9_identifiability.py` runs a **multistart** fit over **both nodes** and
  keeps the best; the conditioning, scan, and prior-demo run at that fit. The
  revised numbers propagate to `docs/identifiability.md`.
- Residual open question (unchanged): which node is Earth's true geometry can't be
  settled by detrended mid-times alone — freeing the inclinations off 90° would do
  it. It does not affect the mass-scale conclusion.
