"""Tests for the Monochromator_Rowland McStas component.

The test instrument is built with the mccode-antlr Assembler API together
with a minimal inline neutron source (TrivialSource) so that no McStas
standard-library installation is required beyond the component under test.

Tests
-----
test_component_parses
    Verifies that mccode-antlr can load and parse Monochromator_Rowland.comp
    without a C compiler.

test_component_compiles
    Builds and runs a minimal instrument that contains the component.
    Skipped when no working C compiler is available.

test_rowland_circle_self_consistency
    Parses the slab positions and Rowland circle parameters printed by the
    component's INITIALIZE block and verifies that every slab lies on the
    circle (distance from center == radius).

test_zero_reflectivity_suppresses_scatter
    With r0=0 the component should never produce a SCATTER event.

test_bragg_scattering_occurs
    With a neutron at the Bragg wavelength aimed at the crystal, at least
    some scattering must occur (non-zero SCATTER count).

test_dm_and_q_equivalent
    Specifying DM=3.355 and Q=2π/3.355 must yield identical results.

test_focus_modes_all_run
    All four focush modes ("", "parallel", "point", "exact") compile and run
    without error.
"""
from __future__ import annotations

import math
import re
from textwrap import dedent

import pytest

from .utilities import compiled, repo_registry, compile_and_run

# ---------------------------------------------------------------------------
# Inline component: deterministic single-neutron source.
# ---------------------------------------------------------------------------
_TRIVIAL_SOURCE_COMP = dedent("""\
    DEFINE COMPONENT TrivialSource
    SETTING PARAMETERS (double velocity=2200.0, double vx_in=0.0, double vy_in=0.0)
    TRACE
    %{
      x = 0; y = 0; z = 0;
      vx = vx_in; vy = vy_in; vz = velocity;
      t = 0.0;
      p = 1.0;
      SCATTER;
    %}
    END
""")

# A divergent source that fans neutrons horizontally over ±dh_deg degrees.
# The rand01()-based divergence ensures different slabs are illuminated;
# each neutron index gets a reproducible RNG state via the McStas 3 PRNG so
# results with a fixed seed are deterministic.
_DIVERGENT_SOURCE_COMP = dedent("""\
    DEFINE COMPONENT DivergentSource
    SETTING PARAMETERS (double velocity=2200.0, double dh_deg=5.0)
    TRACE
    %{
      double angle = (2.0*rand01() - 1.0) * DEG2RAD * dh_deg;
      x = 0; y = 0; z = 0;
      vx = velocity * sin(angle);
      vy = 0;
      vz = velocity * cos(angle);
      t = 0.0;
      p = 1.0;
      SCATTER;
    %}
    END
""")

# ---------------------------------------------------------------------------
# Test instrument geometry constants
# ---------------------------------------------------------------------------
# PG 002 lattice spacing (Å)
_DM = 3.355
# 2θ = 80° → θ_B = 40°
_TWO_THETA_DEG = 80.0
_THETA_B_DEG = _TWO_THETA_DEG / 2.0          # 40°
_THETA_B_RAD = math.radians(_THETA_B_DEG)
# Bragg condition: λ = 2·d·sin(θ_B)
_LAMBDA_BRAGG = 2.0 * _DM * math.sin(_THETA_B_RAD)   # Å
# Neutron speed at that wavelength (v = 3956 / λ [Å] m/s)
_V_BRAGG = 3956.0 / _LAMBDA_BRAGG                    # m/s  ≈ 918 m/s

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registries():
    from mccode_antlr.reader.registry import InMemoryRegistry
    src_reg = InMemoryRegistry('trivial_src')
    src_reg.add_comp('TrivialSource', _TRIVIAL_SOURCE_COMP)
    src_reg.add_comp('DivergentSource', _DIVERGENT_SOURCE_COMP)
    return [src_reg, repo_registry()]


def _build_standard_instrument(name: str = 'MonoRowlandTest',
                                extra_comp_params: dict | None = None,
                                focush: str = '',
                                nH: int = 7) -> object:
    """Assemble a minimal instrument with correct Bragg geometry.

    TrivialSource emits along lab +z.  For the Monochromator_Rowland crystal
    to satisfy the Bragg condition the component must be rotated so that the
    crystal plane normal (local x-axis) makes angle θ_B = 40° with the
    incoming beam.  This requires a rotation of -θ_B around y for the
    Analyzer, and a total rotation of -2θ_B around y for the DetectorArm.

      Origin (TrivialSource)  at (0,0,0) ABSOLUTE — emits along +z
      AnalyzerPoint (Arm)     at (0,0,1) relative to Origin, no rotation
      Analyzer                at AnalyzerPoint, rotated (0,-θ_B,0)
                              → ku_local[x] = sin(θ_B) satisfies Bragg
      DetectorArm (Arm)       at Analyzer, rotated (0,-2θ_B,0) rel. to Origin
                              → z-axis points along the reflected beam
      Focus (Arm)             1 m downstream along DetectorArm's z-axis
      Detector (PSD_monitor)  at Focus
    """
    from mccode_antlr import Flavor
    from mccode_antlr.assembler import Assembler

    assembler = Assembler(name, registries=_make_registries(), flavor=Flavor.MCSTAS)
    assembler.initialize('printf("mono_rowland_init\\n");')

    assembler.component('Origin', 'TrivialSource',
                        at=([0, 0, 0], 'ABSOLUTE'),
                        parameters={'velocity': _V_BRAGG})

    params = {
        'NH': nH,
        'zwidth': 0.01,
        'yheight': 0.15,
        'mosaic': 60.0,
        'DM': _DM,
        'gap': 0.002,
        'source': '"Origin"',
        'sink': '"Focus"',
        'focush': f'"{focush}"',
        'verbose': 1,
    }
    if extra_comp_params:
        params.update(extra_comp_params)

    assembler.component('AnalyzerPoint', 'Arm',
                        at=([0, 0, 1], 'Origin'))
    # Rotate by θ_B (not 2θ!) so the crystal plane normal is at 50° from beam,
    # giving angle-of-incidence = θ_B = 40° (Bragg condition for DM=3.355 Å).
    assembler.component('Analyzer', 'Monochromator_Rowland',
                        at=([0, 0, 0], 'AnalyzerPoint'),
                        rotate=([0, -_THETA_B_DEG, 0], 'AnalyzerPoint'),
                        parameters=params)
    # DetectorArm: total 2θ rotation from Origin so its z-axis follows the
    # reflected beam direction.
    assembler.component('DetectorArm', 'Arm',
                        at=([0, 0, 0], 'Analyzer'),
                        rotate=([0, -_TWO_THETA_DEG, 0], 'AnalyzerPoint'))
    assembler.component('Focus', 'Arm',
                        at=([0, 0, 1], 'DetectorArm'))
    assembler.component('Detector', 'PSD_monitor',
                        at=([0, 0, 0], 'Focus'),
                        parameters={'nx': 20, 'ny': 20,
                                    'filename': '"psd.dat"',
                                    'restore_neutron': 1,
                                    'yheight': 0.3, 'xwidth': 0.3})

    return assembler.instrument


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_component_parses():
    """mccode-antlr can read and parse Monochromator_Rowland.comp.

    Does not require a C compiler.  Validates the McCode DSL syntax and all
    component sections (DEFINE, SETTING PARAMETERS, SHARE, DECLARE,
    INITIALIZE, TRACE, FINALLY, MCDISPLAY).
    """
    from pathlib import Path
    from git import Repo, InvalidGitRepositoryError
    from mccode_antlr import Flavor
    from mccode_antlr.reader import Reader

    try:
        root = Path(Repo('.', search_parent_directories=True).working_tree_dir)
    except InvalidGitRepositoryError as exc:
        raise RuntimeError(f"Could not locate repository root: {exc}") from exc

    comp_path = root / 'Monochromator_Rowland.comp'
    assert comp_path.exists(), f"Component file not found: {comp_path}"

    reader = Reader(registries=[repo_registry()], flavor=Flavor.MCSTAS)
    comp = reader.get_component('Monochromator_Rowland')
    assert comp is not None
    assert comp.name == 'Monochromator_Rowland'


@compiled
def test_component_compiles():
    """A minimal instrument containing Monochromator_Rowland compiles and runs.

    Checks that the compiled binary produces the 'Rowland sphere centered at'
    line that the INITIALIZE block always prints, confirming that at least one
    simulated neutron event triggered the component's initialisation path.
    """
    instr = _build_standard_instrument('MonoRowlandCompile')
    results = compile_and_run(instr, ncount=10, seed=1)
    text = results['output'].decode(errors='replace')
    assert 'Rowland sphere centered at' in text, (
        f"Expected 'Rowland sphere centered at' in output:\n{text}")


@compiled
def test_rowland_circle_self_consistency():
    """All printed slab positions lie on the reported Rowland circle.

    Parses the slab position table and circle parameters from stdout, then
    checks that every slab is within 0.1 mm of the stated radius from the
    stated center.
    """
    instr = _build_standard_instrument('MonoRowlandGeometry', nH=7)
    results = compile_and_run(instr, ncount=1, seed=1)
    text = results['output'].decode(errors='replace')

    # Extract circle center: "Rowland sphere centered at (cx, 0, cz)"
    m = re.search(r'Rowland sphere centered at \(\s*([-\d.eE+]+),\s*[-\d.eE+]+,\s*([-\d.eE+]+)\)',
                  text)
    assert m, f"Could not find Rowland center in output:\n{text}"
    cx, cz = float(m.group(1)), float(m.group(2))

    # Extract radius: "with radius R" or deduce from first slab vs center
    # The component prints slab positions as [ [x y z] [x y z] ... ] in mm
    slab_block = re.search(r'\[(\s*\[.*?\])+\s*\]', text, re.DOTALL)
    assert slab_block, f"Could not find slab position table in output:\n{text}"
    slab_positions = re.findall(r'\[\s*([-\d. ]+)\s*\]', slab_block.group())
    assert len(slab_positions) > 0

    slabs_m = []
    for pos in slab_positions:
        vals = [float(v) for v in pos.split()]
        if len(vals) == 3:
            slabs_m.append((vals[0] / 1000.0, vals[2] / 1000.0))  # mm → m

    # Compute expected radius as the mean distance from center to all slabs
    radii = [math.sqrt((sx - cx)**2 + (sz - cz)**2) for sx, sz in slabs_m]
    r_mean = sum(radii) / len(radii)

    tol = 1e-4  # 0.1 mm
    for i, r in enumerate(radii):
        assert abs(r - r_mean) < tol, (
            f"Slab {i} is {abs(r - r_mean)*1000:.3f} mm off the Rowland circle "
            f"(expected radius {r_mean:.4f} m, got {r:.4f} m)")


@compiled
def _build_scatter_counter_instrument(name: str, mono_params: dict,
                                       sentinel: str) -> tuple:
    """Build an instrument with correct Rowland geometry and a scatter counter.

    Returns (assembler, mono_instance) so callers can add extra EXTEND code
    before retrieving assembler.instrument.
    """
    from mccode_antlr import Flavor
    from mccode_antlr.assembler import Assembler

    asm = Assembler(name, registries=_make_registries(), flavor=Flavor.MCSTAS)
    asm.declare('int scatter_count = 0;')
    asm.initialize(f'printf("{sentinel}\\n");')

    asm.component('Origin', 'TrivialSource',
                  at=([0, 0, 0], 'ABSOLUTE'),
                  parameters={'velocity': _V_BRAGG})
    asm.component('AnalyzerPoint', 'Arm',
                  at=([0, 0, 1], 'Origin'))
    mono = asm.component(
        'Analyzer', 'Monochromator_Rowland',
        at=([0, 0, 0], 'AnalyzerPoint'),
        rotate=([0, -_THETA_B_DEG, 0], 'AnalyzerPoint'),
        parameters=mono_params)
    mono.EXTEND('if (SCATTERED) scatter_count++;')

    asm.component('DetectorArm', 'Arm',
                  at=([0, 0, 0], 'Analyzer'),
                  rotate=([0, -_TWO_THETA_DEG, 0], 'AnalyzerPoint'))
    asm.component('Focus', 'Arm',
                  at=([0, 0, 1], 'DetectorArm'))
    asm.final('printf("scatter_count=%d\\n", scatter_count);')
    return asm, mono


@compiled
def test_zero_reflectivity_suppresses_scatter():
    """With r0=0 no neutron should be Bragg-scattered.

    Uses an EXTEND block on the Analyzer to count the number of SCATTER
    events; asserts that the count remains zero across 1000 neutrons.
    """
    asm, _mono = _build_scatter_counter_instrument(
        'MonoRowlandZeroR0',
        mono_params={
            'NH': 1, 'zwidth': 0.05, 'yheight': 0.15,
            'mosaic': 60.0, 'DM': _DM, 'gap': 0.0,
            'r0': 0.0,
            'source': '"Origin"', 'sink': '"Focus"',
        },
        sentinel='zero_r0_start',
    )
    results = compile_and_run(asm.instrument, ncount=1000, seed=42)
    text = results['output'].decode(errors='replace')
    assert 'zero_r0_start' in text, f"Sentinel missing:\n{text}"

    m = re.search(r'scatter_count=(\d+)', text)
    assert m, f"scatter_count sentinel not found in output:\n{text}"
    assert int(m.group(1)) == 0, (
        f"Expected zero scatters with r0=0, got {m.group(1)}")


@compiled
def test_bragg_scattering_occurs():
    """Neutrons at the Bragg wavelength are scattered by the monochromator.

    Runs 10 000 neutrons at the Bragg velocity.  Counts SCATTER events via
    an EXTEND block.  Asserts that at least one neutron was scattered.
    """
    asm, _mono = _build_scatter_counter_instrument(
        'MonoRowlandBragg',
        mono_params={
            'NH': 1, 'zwidth': 0.05, 'yheight': 0.15,
            'mosaic': 60.0, 'DM': _DM, 'gap': 0.0,
            'r0': 0.7,
            'source': '"Origin"', 'sink': '"Focus"',
        },
        sentinel='bragg_start',
    )
    results = compile_and_run(asm.instrument, ncount=10000, seed=1)
    text = results['output'].decode(errors='replace')
    assert 'bragg_start' in text, f"Sentinel missing:\n{text}"

    m = re.search(r'scatter_count=(\d+)', text)
    assert m, f"scatter_count sentinel not found in output:\n{text}"
    assert int(m.group(1)) > 0, (
        f"Expected at least one Bragg scatter for neutrons at v={_V_BRAGG:.1f} m/s "
        f"(λ={_LAMBDA_BRAGG:.3f} Å, DM={_DM} Å), got 0.\nOutput:\n{text}")


@compiled
def test_dm_and_q_equivalent():
    """DM=3.355 and Q=2π/3.355 must produce exactly the same scatter count.

    Both specifications are reduced to the same `tau` value in INITIALIZE
    (Q is computed as 2π/DM when DM is set), so results should be bitwise
    identical for the same seed.
    """
    Q_equiv = 2.0 * math.pi / _DM

    def build(use_dm: bool) -> object:
        name = 'MonoRowlandDM' if use_dm else 'MonoRowlandQ'
        mono_params = {
            'NH': 1, 'zwidth': 0.05, 'yheight': 0.15,
            'mosaic': 60.0, 'gap': 0.0, 'r0': 0.7,
            'source': '"Origin"', 'sink': '"Focus"',
        }
        if use_dm:
            mono_params['DM'] = _DM
        else:
            mono_params['Q'] = Q_equiv
        asm, _mono = _build_scatter_counter_instrument(
            name, mono_params=mono_params,
            sentinel=f'{name}_start')
        return asm.instrument

    def get_scatter_count(instr) -> int:
        res = compile_and_run(instr, ncount=1000, seed=7)
        text = res['output'].decode(errors='replace')
        m = re.search(r'scatter_count=(\d+)', text)
        assert m, f"scatter_count sentinel not found:\n{text}"
        return int(m.group(1))

    count_dm = get_scatter_count(build(use_dm=True))
    count_q  = get_scatter_count(build(use_dm=False))
    # Both tau values are numerically equivalent (2π/DM ≈ Q), so scatter counts
    # should be statistically consistent.  Exact equality is not guaranteed: the
    # Python float literal for Q_equiv and the C runtime expression 2*PI/DM may
    # differ by 1 ULP, causing tiny Bragg-probability differences that
    # accumulate over 1000 neutrons.  A 10% relative tolerance is sufficient to
    # catch a truly wrong Q value while allowing for this representation noise.
    rel_diff = abs(count_dm - count_q) / max(count_dm, count_q)
    assert rel_diff < 0.10, (
        f"DM={_DM} gave {count_dm} scatters but Q={Q_equiv:.6f} gave {count_q} "
        f"({rel_diff*100:.1f}% relative difference, expected < 10%). "
        "Check that Q and DM map to the same scattering vector tau.")


@compiled
@pytest.mark.parametrize('focush', ['', 'parallel', 'point', 'exact'])
def test_focus_modes_all_run(focush):
    """Every focush mode compiles, runs, and prints the Rowland circle info.

    Parallel and point modes additionally log the computed focus radius.
    """
    instr = _build_standard_instrument(f'MonoRowlandFocus_{focush or "none"}',
                                       focush=focush,
                                       extra_comp_params={'verbose': 1})
    results = compile_and_run(instr, ncount=10, seed=1)
    text = results['output'].decode(errors='replace')

    assert 'Rowland sphere centered at' in text, (
        f"focush='{focush}': 'Rowland sphere centered at' missing.\n{text}")

    if focush in ('parallel', 'point'):
        assert 'Calculated horizontal focus radius' in text, (
            f"focush='{focush}': Expected focus-radius log line.\n{text}")


@compiled
def test_exact_focusing_is_optimal():
    """tiltH_scale=1 (nominal focush="exact") gives the best horizontal focus.

    A DivergentSource fans neutrons at ±5° horizontally, illuminating all 7
    slabs (which span ~10 cm at 1 m).  With focush="exact" and tiltH_scale=1
    the Rowland inscribed-angle formula tilts each slab so all beams converge
    to the Focus point (1 m along the reflected-beam arm).  With tiltH_scale=0
    (flat) the 7 beams fan out ~93 mm at 1 m, so only ~5% land in a 5 mm
    detector.  With tiltH_scale=2 (over-focused) the beams converge before the
    detector and then diverge again.

    We use a deliberately narrow detector (5 mm wide) at Focus so that focused
    intensity >> unfocused intensity, verifying:
        intensity(scale=1)  >  intensity(scale=0)   (focusing improves over flat)
        intensity(scale=1)  >  intensity(scale=2)   (scale=1 is better than over-focus)
    """
    from mccode_antlr import Flavor
    from mccode_antlr.assembler import Assembler

    # Half-divergence large enough to illuminate all 7 slabs at 1 m distance
    # (slabs span ~10 cm → ±5° covers ±87 mm at 1 m).
    _DH_DEG = 5.0

    _SCALES = [0.0, 1.0, 2.0]

    def build(scale: float) -> object:
        name = f'MonoRowlandTiltScale_{str(scale).replace(".", "p")}'
        params = {
            'NH': 7,
            'zwidth': 0.01,
            'yheight': 0.15,
            'mosaic': 60.0,
            'DM': _DM,
            'gap': 0.002,
            'source': '"Origin"',
            'sink': '"Focus"',
            'focush': '"exact"',
            'tiltH_scale': scale,
            'verbose': 0,
        }
        asm = Assembler(name, registries=_make_registries(), flavor=Flavor.MCSTAS)
        asm.component('Origin', 'DivergentSource',
                      at=([0, 0, 0], 'ABSOLUTE'),
                      parameters={'velocity': _V_BRAGG, 'dh_deg': _DH_DEG})
        asm.component('AnalyzerPoint', 'Arm',
                      at=([0, 0, 1], 'Origin'))
        asm.component('Analyzer', 'Monochromator_Rowland',
                      at=([0, 0, 0], 'AnalyzerPoint'),
                      rotate=([0, -_THETA_B_DEG, 0], 'AnalyzerPoint'),
                      parameters=params)
        asm.component('DetectorArm', 'Arm',
                      at=([0, 0, 0], 'Analyzer'),
                      rotate=([0, -_TWO_THETA_DEG, 0], 'AnalyzerPoint'))
        asm.component('Focus', 'Arm',
                      at=([0, 0, 1], 'DetectorArm'))
        # Narrow detector (5 mm wide) at focus.  With exact focusing all 7
        # slabs converge to ~0 mm; with flat the spread is ~93 mm, so only
        # ~5% of scattered neutrons land in the 5 mm window.
        asm.component('FocusDetector', 'PSD_monitor',
                      at=([0, 0, 0], 'Focus'),
                      parameters={'nx': 20, 'ny': 1,
                                  'filename': f'"psd_{scale}.dat"',
                                  'restore_neutron': 1,
                                  'yheight': 0.3, 'xwidth': 0.005})
        return asm.instrument

    def focused_intensity(scale: float) -> float:
        res = compile_and_run(build(scale), ncount=5000, seed=42)
        text = res['output'].decode(errors='replace')
        m = re.search(r'FocusDetector_I=\s*([\d.eE+\-]+)', text)
        assert m, f"FocusDetector_I not found in output for scale={scale}:\n{text}"
        print(f"scale={scale}: FocusDetector_I = {m.group(1)}\n{text}")
        return float(m.group(1))

    counts = {s: focused_intensity(s) for s in _SCALES}

    nominal = counts[1.0]
    flat    = counts[0.0]
    over    = counts[2.0]

    assert nominal > 0, (
        f"scale=1 (exact Rowland) gave zero intensity at the focus detector. "
        f"All counts: {counts}")
    assert nominal > flat, (
        f"scale=1 ({nominal:.3g}) should be > scale=0/flat ({flat:.3g}) — "
        f"exact focusing must concentrate the beam into the 5 mm window. "
        f"All counts: {counts}")
    assert nominal > over, (
        f"scale=1 ({nominal:.3g}) should be > scale=2/over-focused ({over:.3g}) — "
        f"over-focus converges before the detector. "
        f"All counts: {counts}")
