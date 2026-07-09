# Overnight report — 2026-07-08

> **⚠ Superseded 2026-07-09 (Kepler-9 section).** The "sub-optimal transit node"
> explanation below is **wrong**. The χ²≈598 was **under-optimization** (single
> cold-start fit), not a node error: a multistart fit at the *original* node
> reaches χ²/dof ≈ 180–192, and the real structure is a mass-scale degeneracy
> (m_b spans 15–75 M⊕ at a fixed ~1.47 ratio). The node convention was a red
> herring. See [kepler9_node_postmortem.md](kepler9_node_postmortem.md) and the
> ✔-corrected banner in [identifiability.md](identifiability.md). Everything below
> is left as originally written for the record.

Two tasks: run the 3-D Kepler-9 tests, and start M3 (Path B). Both done. One
finding needs your judgment (the Kepler-9 node issue). Everything is in the
working tree; git is still detached (publish-ready).

## TL;DR

- **M3 (Path B) — done and verified.** Density-field identifiability works: a
  single satellite can't constrain ρ(altitude, local time); a constellation can.
  New: `perturber/density.py`, `scripts/run_vleo_identifiability.py`, figure,
  `docs/m3_plan.md`. The identifiability thread now spans **three domains**.
- **3-D Kepler-9 — done, with a consequential finding.** The system is coplanar
  (3-D inclination adds nothing). But an adversarial cross-check revealed our
  **2-D fit was at a sub-optimal transit node** (χ²/dof ≈ 598); the better node
  reaches **χ²/dof ≈ 180** (masses ~30/21). **The Kepler-9 identifiability numbers
  need revising** — flagged in `docs/identifiability.md`. Qualitative conclusions
  expected to hold.

## M3 (VLEO) — Path B, first result

`ρ(h, local time)` is sampled by a satellite only along its track, so recovering
the field is the same design-matrix problem as force-form recovery. Result
(`scripts/run_vleo_identifiability.py`, figure `docs/figures/vleo_restoration.png`):

| constellation | condition # | field recovery |
|---|---|---|
| 1 satellite | ~370 | 12–60% |
| 2 satellites | ~93 | ~90–98% |
| 3+ satellites | ~4–5 | 100% |

The multi-object joint constraint (M3's reason to exist) breaks the single-
satellite degeneracy, quantified with the same tooling as force laws and
Kepler-9 masses. See `docs/m3_plan.md` for the full framing (we chose Path B —
infer/quantify — over Path A, the STORM-AI forecasting leaderboard) and the next
steps (a velocity-dependent drag forward model; the ρ–ballistic-coefficient
scale degeneracy on top of the field-shape one). STORM-AI data characterised;
`data/stormai/` gitignored (download from Harvard Dataverse DOI 10.7910/DVN/U6K6MJ).

## 3-D Kepler-9 — the finding that needs you

**Result:** the fit drives mutual inclination to ~0° and 3-D freedom does not
improve the fit (coplanar 185 → 3-D 191). Kepler-9 is coplanar — expected.

**The consequential part (found by adversarial debate):** building the 3-D model
surfaced that our **2-D Kepler-9 fit was stuck at a worse transit node**. The 2-D
and 3-D transit finders use opposite "in-front" sign conventions
(`transits.py:31` `y<0` vs `threed.py:70` `dZ>0`); for an exactly edge-on model
these are opposite orbital nodes, and the two nodes fit the real O-C very
differently — **χ²/dof ≈ 598 (our node) vs ≈ 180 (the other)**. The two forward
models are otherwise numerically identical (proven: same trajectory → same χ² to
5 sig figs), so 180 is a real, reachable, better fit — our 2-D pipeline just
locked to the worse node and had no multistart.

**Why I did not auto-fix it:** this is a subtle 2-fold node/degeneracy question,
exactly the kind where an unsupervised "fix" risks compounding the error. So I
flagged it (banner in `docs/identifiability.md`) rather than silently re-running
the Kepler-9 conditioning / scan / prior-demo at the new fit.

**Recommended fix (your call):**
1. Give `scripts/run_kepler9_identifiability.py` a **multistart** (the M1 fitter
   already has one; this script does a single cold start — the root cause).
2. **Settle the node:** the data prefers the 180 node; to *prove* which is Earth's
   geometry, free the inclinations slightly off 90° (the two nodes then get
   different impact parameters / transit durations) or anchor an absolute epoch.
3. Re-run the conditioning / degeneracy-scan / prior-demo at the χ²≈180 fit and
   update the three numbers in `docs/identifiability.md`. Expect the *story* to
   hold (ratio tight, scale loosest, prior helps only when sparse); numbers move.

## The adversarial debate (your rule earned its keep)

"When in doubt, run an adversarial debate" — saved to memory and applied. Tonight
it caught **three** things I would otherwise have reported wrong:
1. an edge-on **kink bug** in the 3-D transit finder (parabolic fit on a V-shaped
   |dX|) — fixed (now O-C ~1e-8);
2. a **phase-convention** mismatch that made the first checkpoint fail loudly
   (good — it was supposed to);
3. the **node issue** above — two agents argued opposite sides; side-B found the
   real mechanism, side-A proved the models are otherwise identical.

## State of the tree

- New/changed: `perturber/{threed,density}.py`, `scripts/{run_kepler9_3d,
  run_vleo_identifiability}.py`, `docs/{m3_plan,identifiability}.md`, figures,
  `.gitignore` (stormai). All module self-checks pass (`threed`, `density`
  included). Git detached; history backed up in the session scratchpad.
- Scratch verification scripts (finder precision, 2-D refit, numerics sweep) are
  in the session scratchpad, not the repo.
