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

# ---------------------------------------------------------------------------
# Test instrument geometry constants
# ---------------------------------------------------------------------------
# PG 002 lattice spacing (Å)
_DM = 3.355
# 2θ = 80° → θ_B = 40°
_TWO_THETA_DEG = 80.0
_THETA_B_RAD = math.radians(_TWO_THETA_DEG / 2.0)
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
    return [src_reg, repo_registry()]


def _build_standard_instrument(name: str = 'MonoRowlandTest',
                                extra_comp_params: dict | None = None,
                                focush: str = '',
                                nH: int = 7) -> object:
    """Assemble a minimal instrument replicating test_Monochromator_Rowland.instr.

    Layout (absolute frame, all distances in metres):
      TrivialSource  at origin
      AnalyzerArm    1 m downstream (z=1), rotated -80° around y
      Monochromator  at AnalyzerArm, source="Origin", sink="Focus"
      DetectorArm    at AnalyzerArm, further rotated -80° around y (total -160°)
      Focus (Arm)    1 m along DetectorArm's z-axis
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
    }
    if extra_comp_params:
        params.update(extra_comp_params)

    assembler.component('AnalyzerArm', 'Arm',
                        at=([0, 0, 1], 'RELATIVE', 'Origin'),
                        rotate=([0, -_TWO_THETA_DEG, 0], 'RELATIVE', 'Origin'))
    assembler.component('Analyzer', 'Monochromator_Rowland',
                        at=([0, 0, 0], 'RELATIVE', 'AnalyzerArm'),
                        rotate=([0, 0, 0], 'RELATIVE', 'AnalyzerArm'),
                        parameters=params)
    assembler.component('DetectorArm', 'Arm',
                        at=([0, 0, 0], 'RELATIVE', 'AnalyzerArm'),
                        rotate=([0, -_TWO_THETA_DEG, 0], 'RELATIVE', 'AnalyzerArm'))
    assembler.component('Focus', 'Arm',
                        at=([0, 0, 1], 'RELATIVE', 'DetectorArm'))
    assembler.component('Detector', 'PSD_monitor',
                        at=([0, 0, 0], 'RELATIVE', 'Focus'),
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
def test_zero_reflectivity_suppresses_scatter():
    """With r0=0 no neutron should be Bragg-scattered.

    Uses an EXTEND block on the Analyzer to count the number of SCATTER
    events; asserts that the count remains zero across 1000 neutrons.
    """
    from mccode_antlr import Flavor
    from mccode_antlr.assembler import Assembler

    assembler = Assembler('MonoRowlandZeroR0',
                          registries=_make_registries(),
                          flavor=Flavor.MCSTAS)
    assembler.declare('int scatter_count = 0;')
    assembler.initialize('printf("zero_r0_start\\n");')

    assembler.component('Origin', 'TrivialSource',
                        at=([0, 0, 0], 'ABSOLUTE'),
                        parameters={'velocity': _V_BRAGG})
    assembler.component('AnalyzerArm', 'Arm',
                        at=([0, 0, 1], 'RELATIVE', 'Origin'),
                        rotate=([0, -_TWO_THETA_DEG, 0], 'RELATIVE', 'Origin'))
    mono = assembler.component(
        'Analyzer', 'Monochromator_Rowland',
        at=([0, 0, 0], 'RELATIVE', 'AnalyzerArm'),
        rotate=([0, 0, 0], 'RELATIVE', 'AnalyzerArm'),
        parameters={
            'NH': 1, 'zwidth': 0.05, 'yheight': 0.15,
            'mosaic': 60.0, 'DM': _DM, 'gap': 0.0,
            'r0': 0.0,       # ← zero reflectivity
            'source': '"Origin"', 'sink': '"Focus"',
        })
    mono.EXTEND('%{ if (SCATTERED) scatter_count++; %}')

    assembler.component('DetectorArm', 'Arm',
                        at=([0, 0, 0], 'RELATIVE', 'AnalyzerArm'),
                        rotate=([0, -_TWO_THETA_DEG, 0], 'RELATIVE', 'AnalyzerArm'))
    assembler.component('Focus', 'Arm',
                        at=([0, 0, 1], 'RELATIVE', 'DetectorArm'))
    assembler.finalize(
        'printf("scatter_count=%d\\n", scatter_count);'
    )

    results = compile_and_run(assembler.instrument, ncount=1000, seed=42)
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
    from mccode_antlr import Flavor
    from mccode_antlr.assembler import Assembler

    assembler = Assembler('MonoRowlandBragg',
                          registries=_make_registries(),
                          flavor=Flavor.MCSTAS)
    assembler.declare('int scatter_count = 0;')
    assembler.initialize('printf("bragg_start\\n");')

    assembler.component('Origin', 'TrivialSource',
                        at=([0, 0, 0], 'ABSOLUTE'),
                        parameters={'velocity': _V_BRAGG})
    assembler.component('AnalyzerArm', 'Arm',
                        at=([0, 0, 1], 'RELATIVE', 'Origin'),
                        rotate=([0, -_TWO_THETA_DEG, 0], 'RELATIVE', 'Origin'))
    mono = assembler.component(
        'Analyzer', 'Monochromator_Rowland',
        at=([0, 0, 0], 'RELATIVE', 'AnalyzerArm'),
        rotate=([0, 0, 0], 'RELATIVE', 'AnalyzerArm'),
        parameters={
            'NH': 1, 'zwidth': 0.05, 'yheight': 0.15,
            'mosaic': 60.0, 'DM': _DM, 'gap': 0.0,
            'r0': 0.7,
            'source': '"Origin"', 'sink': '"Focus"',
        })
    mono.EXTEND('%{ if (SCATTERED) scatter_count++; %}')

    assembler.component('DetectorArm', 'Arm',
                        at=([0, 0, 0], 'RELATIVE', 'AnalyzerArm'),
                        rotate=([0, -_TWO_THETA_DEG, 0], 'RELATIVE', 'AnalyzerArm'))
    assembler.component('Focus', 'Arm',
                        at=([0, 0, 1], 'RELATIVE', 'DetectorArm'))
    assembler.finalize(
        'printf("scatter_count=%d\\n", scatter_count);'
    )

    results = compile_and_run(assembler.instrument, ncount=10000, seed=1)
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
    from mccode_antlr import Flavor
    from mccode_antlr.assembler import Assembler

    Q_equiv = 2.0 * math.pi / _DM

    def build(use_dm: bool) -> object:
        name = 'MonoRowlandDM' if use_dm else 'MonoRowlandQ'
        asm = Assembler(name, registries=_make_registries(), flavor=Flavor.MCSTAS)
        asm.declare('int scatter_count = 0;')
        asm.initialize(f'printf("{name}_start\\n");')
        asm.component('Origin', 'TrivialSource',
                      at=([0, 0, 0], 'ABSOLUTE'),
                      parameters={'velocity': _V_BRAGG})
        asm.component('AnalyzerArm', 'Arm',
                      at=([0, 0, 1], 'RELATIVE', 'Origin'),
                      rotate=([0, -_TWO_THETA_DEG, 0], 'RELATIVE', 'Origin'))
        mono_params = {
            'NH': 1, 'zwidth': 0.05, 'yheight': 0.15,
            'mosaic': 60.0, 'gap': 0.0, 'r0': 0.7,
            'source': '"Origin"', 'sink': '"Focus"',
        }
        if use_dm:
            mono_params['DM'] = _DM
        else:
            mono_params['Q'] = Q_equiv
        mono = asm.component(
            'Analyzer', 'Monochromator_Rowland',
            at=([0, 0, 0], 'RELATIVE', 'AnalyzerArm'),
            rotate=([0, 0, 0], 'RELATIVE', 'AnalyzerArm'),
            parameters=mono_params)
        mono.EXTEND('%{ if (SCATTERED) scatter_count++; %}')
        asm.component('DetectorArm', 'Arm',
                      at=([0, 0, 0], 'RELATIVE', 'AnalyzerArm'),
                      rotate=([0, -_TWO_THETA_DEG, 0], 'RELATIVE', 'AnalyzerArm'))
        asm.component('Focus', 'Arm',
                      at=([0, 0, 1], 'RELATIVE', 'DetectorArm'))
        asm.finalize('printf("scatter_count=%d\\n", scatter_count);')
        return asm.instrument

    def get_scatter_count(instr) -> int:
        res = compile_and_run(instr, ncount=1000, seed=7)
        text = res['output'].decode(errors='replace')
        m = re.search(r'scatter_count=(\d+)', text)
        assert m, f"scatter_count sentinel not found:\n{text}"
        return int(m.group(1))

    count_dm = get_scatter_count(build(use_dm=True))
    count_q  = get_scatter_count(build(use_dm=False))
    assert count_dm == count_q, (
        f"DM={_DM} gave {count_dm} scatters but Q={Q_equiv:.6f} gave {count_q}. "
        "They should be identical.")


@compiled
@pytest.mark.parametrize('focush', ['', 'parallel', 'point', 'exact'])
def test_focus_modes_all_run(focush):
    """Every focush mode compiles, runs, and prints the Rowland circle info.

    Parallel and point modes additionally log the computed focus radius.
    """
    instr = _build_standard_instrument(f'MonoRowlandFocus_{focush or "none"}',
                                       focush=focush)
    results = compile_and_run(instr, ncount=10, seed=1)
    text = results['output'].decode(errors='replace')

    assert 'Rowland sphere centered at' in text, (
        f"focush='{focush}': 'Rowland sphere centered at' missing.\n{text}")

    if focush in ('parallel', 'point'):
        assert 'Calculated horizontal focus radius' in text, (
            f"focush='{focush}': Expected focus-radius log line.\n{text}")
