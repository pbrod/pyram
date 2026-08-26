"""Tests for PyRAMmp."""

import xml.etree.ElementTree as et
from copy import deepcopy
from pathlib import Path
from time import time

import numpy as np

from pyram.PyRAM import PyRAM, PyRAMResults
from pyram.PyRAMmp import FloatArray, PyRAMArgs, PyRAMKwargs, PyRAMmp, PyRAMRun


def test_pyrammp() -> None:
    """Verify PyRAMmp produces results consistent with sequential PyRAM runs."""

    config = et.parse(Path(__file__).parent / "TestPyRAMmp_Config.xml").getroot()

    range_step = config.find("RangeStep")
    depth_step = config.find("DepthStep")
    num_rep = config.find("NumberOfRepetitions")

    assert range_step is not None
    assert depth_step is not None
    assert num_rep is not None

    assert range_step.text is not None
    assert depth_step.text is not None
    assert num_rep.text is not None

    dr = float(range_step.text)
    dz = float(depth_step.text)
    nrep = int(num_rep.text)

    pyram_args: PyRAMArgs = {
        "freq": 0.0,  # overwritten in test loops
        "zs": 50.0,
        "zr": 50.0,
        "z_ss": np.array([0.0, 100.0, 400.0]),
        "rp_ss": np.array([0.0, 25000.0]),
        "cw": np.array(
            [
                [1480.0, 1530.0],
                [1520.0, 1530.0],
                [1530.0, 1530.0],
            ]
        ),
        "z_sb": np.array([0.0]),
        "rp_sb": np.array([0.0]),
        "cb": np.array([[1700.0]]),
        "rhob": np.array([[1.5]]),
        "attn": np.array([[0.5]]),
        "rbzb": np.array(
            [
                [0.0, 200.0],
                [40000.0, 400.0],
            ]
        ),
    }

    pyram_kwargs: PyRAMKwargs = {
        "rmax": 50000.0,
        "dr": dr,
        "dz": dz,
        "zmplt": 500.0,
        "c0": 1600.0,
    }

    freqs = [30.0, 40.0, 50.0, 60.0, 70.0]

    ref_r: list[FloatArray] = []
    ref_z: list[FloatArray] = []
    ref_tl: list[FloatArray] = []

    for freq in freqs:
        args = deepcopy(pyram_args)
        args["freq"] = float(freq)
        res = PyRAM(
            **args,
            **pyram_kwargs,
        ).run()

        ref_r.append(res.ranges)
        ref_z.append(res.depths)
        ref_tl.append(res.loss_grid)

    freqs_rep = np.tile(freqs, nrep)
    num_runs = len(freqs_rep)

    runs: list[PyRAMRun] = []

    for run_id, freq in enumerate(freqs_rep):
        args = deepcopy(pyram_args)
        args["freq"] = float(freq)

        kwargs = deepcopy(pyram_kwargs)
        kwargs["id"] = run_id

        runs.append((args, kwargs))

    pyram_mp = PyRAMmp()

    try:
        nproc = pyram_mp.num_processes

        t0 = time()

        midpoint = num_runs // 2
        pyram_mp.submit_runs(runs[:midpoint])
        pyram_mp.submit_runs(runs[midpoint:])

        elapsed_time = time() - t0

        assert len(pyram_mp.results) == num_runs

        results_tmp: list[PyRAMResults | None] = [None] * num_runs

        proc_time = 0.0
        for output in pyram_mp.results:
            results_tmp[output.id] = output
            proc_time += output.proc_time

        assert all(r is not None for r in results_tmp)

        results: list[PyRAMResults] = [r for r in results_tmp if r is not None]

        for n, res in enumerate(results):
            assert res is not None

            run_args, _ = runs[n]
            freq = run_args["freq"]

            ref_idx = freqs.index(freq)

            np.testing.assert_array_equal(
                ref_r[ref_idx],
                res.ranges,
            )

            np.testing.assert_array_equal(
                ref_z[ref_idx],
                res.depths,
            )

            np.testing.assert_allclose(
                ref_tl[ref_idx],
                res.loss_grid,
            )

        speed_fact = 100.0 * (proc_time / nproc) / elapsed_time
        print(f"\n{speed_fact:.1f}% of expected speed-up achieved")

    finally:
        pyram_mp.close()
