from pathlib import Path
from collections.abc import Callable


def compiled(method):
    """Decorator: skip test if no working C compiler is available."""
    from pytest import mark
    from mccode_antlr.compiler.check import simple_instr_compiles
    if simple_instr_compiles('cc'):
        return method

    @mark.skip(reason=f"A working C compiler is required for {method.__name__}")
    def skipped(*args, **kwargs):
        pass

    return skipped


def repo_registry():
    """Return a LocalRegistry pointing at the root of this git repository."""
    from git import Repo, InvalidGitRepositoryError
    from mccode_antlr.reader.registry import LocalRegistry
    try:
        repo = Repo('.', search_parent_directories=True)
        return LocalRegistry('monochromator_rowland', repo.working_tree_dir)
    except InvalidGitRepositoryError as exc:
        raise RuntimeError(f"Could not locate repository root: {exc}") from exc


def acceptable_total_counts(dat, expected, tolerance=0.1):
    """Return True if the total intensity in *dat* is within *tolerance* of *expected*."""
    return abs(dat['I'].sum() - expected) <= tolerance * expected


def _time(method):
    """Decorator: return (elapsed_seconds, result)."""
    import time as _t

    def timed(*args, **kwargs):
        start = _t.perf_counter()
        try:
            result = method(*args, **kwargs)
        finally:
            elapsed = _t.perf_counter() - start
            print(f"{method.__name__} took {elapsed:.2f} s")
        return elapsed, result

    return timed


@_time
def _timed_compile(sim, directory=None):
    if directory is None:
        return sim.compile()
    return sim.compile(directory=directory)


@_time
def _timed_run(sim, params, ncount=1000, seed=1):
    return sim.run(params, ncount=ncount, seed=seed)


@_time
def _timed_scan(sim, params, ncount=1000, seed=1):
    return sim.scan(params, ncount=ncount, seed=seed)


def compile_and_run(instr, ncount: int, parameters: dict | None = None,
                    seed: int = 1, use_temp_dir: bool = True) -> dict:
    """Compile *instr* and run once.  Returns dict with 'output', 'data',
    'compile', and 'run' keys.  Raises RuntimeError on failure."""
    from mccode_antlr.run import McStas
    sim = McStas(instr)
    compile_time, _ = _timed_compile(sim, directory=None if use_temp_dir else Path('.'))
    run_time, returned = _timed_run(sim, parameters or {}, ncount=ncount, seed=seed)
    return {'compile': compile_time, 'run': run_time, 'output': returned}


def compile_and_scan(instr, parameters: dict, ncount: int, seed: int = 1,
                     use_temp_dir: bool = True) -> dict:
    """Compile *instr* and run a parameter scan.  Returns dict with
    'scan_result' (list of results dicts), 'compile', and 'run' keys."""
    from mccode_antlr.run import McStas
    sim = McStas(instr)
    compile_time, _ = _timed_compile(sim, directory=None if use_temp_dir else Path('.'))
    run_time, output_results_list = _timed_scan(sim, parameters, ncount=ncount, seed=seed)
    scan_result = [results for _output, results in output_results_list]
    return {'compile': compile_time, 'run': run_time, 'scan_result': scan_result}
