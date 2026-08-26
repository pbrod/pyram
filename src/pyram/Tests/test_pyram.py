"""Tests for PyRAM."""

from pathlib import Path

import numpy as np

from pyram.PyRAM import PyRAM


def test_pyram() -> None:
    """Verify PyRAM reproduces the reference RAM solution."""

    dat = np.fromfile(
        Path(__file__).parent / "tl_ref.line",
        sep="\t",
    ).reshape(100, 2)

    ref_r = dat[:, 0]
    ref_tl = dat[:, 1]

    pyram = PyRAM(
        freq=50,
        zs=50,
        zr=50,
        z_ss=np.array([0, 100, 400]),
        rp_ss=np.array([0, 25000]),
        cw=np.array(
            [
                [1480, 1530],
                [1520, 1530],
                [1530, 1530],
            ]
        ),
        z_sb=np.array([0]),
        rp_sb=np.array([0]),
        cb=np.array([[1700]]),
        rhob=np.array([[1.5]]),
        attn=np.array([[0.5]]),
        rbzb=np.array(
            [
                [0, 200],
                [40000, 400],
            ]
        ),
        rmax=50000,
        dr=500,
        dz=2,
        zmplt=500,
        c0=1600,
    )

    pyram.run()

    np.testing.assert_array_equal(ref_r, pyram.vr)

    mean_diff = np.mean(np.abs(pyram.tll - ref_tl))
    assert mean_diff <= 1e-2, f"Mean TL difference ({mean_diff:.6f} dB) exceeds tolerance (0.01 dB)"
