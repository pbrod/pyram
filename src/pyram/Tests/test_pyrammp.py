"""Tests for PyRAMmp."""

import xml.etree.ElementTree as et
from copy import deepcopy
from pathlib import Path
from time import time

import numpy as np

from pyram.PyRAM import PyRAM
from pyram.PyRAMmp import PyRAMmp


def test_pyrammp():
    """Verify PyRAMmp produces results consistent with sequential PyRAM runs."""

    config = et.parse(Path(__file__).parent / "TestPyRAMmp_Config.xml").getroot()

    dr = float(config.find("RangeStep").text)
    dz = float(config.find("DepthStep").text)
    nrep = int(config.find("NumberOfRepetitions").text)

    pyram_args = {
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

    pyram_kwargs = {
        "rmax": 50000.0,
        "dr": dr,
        "dz": dz,
        "zmplt": 500.0,
        "c0": 1600.0,
    }

    freqs = [30.0, 40.0, 50.0, 60.0, 70.0]

    ref_r = []
    ref_z = []
    ref_tl = []

    for freq in freqs:
        args = deepcopy(pyram_args)
        kwargs = deepcopy(pyram_kwargs)
        result = PyRAM(
            freq=freq,
            **args,
            **kwargs,
        ).run()

        ref_r.append(result["Ranges"])
        ref_z.append(result["Depths"])
        ref_tl.append(result["TL Grid"])

    freqs_rep = np.tile(freqs, nrep)
    num_runs = len(freqs_rep)

    runs = []

    for run_id, freq in enumerate(freqs_rep):
        args = deepcopy(pyram_args)
        args["freq"] = float(freq)

        kwargs = deepcopy(pyram_kwargs)
        kwargs["id"] = run_id

        runs.append((args, kwargs))

    pyram_mp = PyRAMmp()

    try:
        nproc = pyram_mp.pool._processes

        t0 = time()

        midpoint = num_runs // 2
        pyram_mp.submit_runs(runs[:midpoint])
        pyram_mp.submit_runs(runs[midpoint:])

        elapsed_time = time() - t0

        results = [None] * num_runs
        proc_time = 0.0

        for result in pyram_mp.results:
            results[result["ID"]] = result
            proc_time += result["Proc Time"]

        for n, result in enumerate(results):
            freq = runs[n][0]["freq"]
            ref_idx = freqs.index(freq)

            np.testing.assert_array_equal(
                ref_r[ref_idx],
                result["Ranges"],
            )

            np.testing.assert_array_equal(
                ref_z[ref_idx],
                result["Depths"],
            )

            np.testing.assert_allclose(
                ref_tl[ref_idx],
                result["TL Grid"],
            )

        speed_fact = 100.0 * (proc_time / nproc) / elapsed_time
        print(f"\n{speed_fact:.1f}% of expected speed-up achieved")

    finally:
        pyram_mp.close()
