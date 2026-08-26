"""
outpt function definition
"""

import numpy as np
from numba import complex128, float64, int64, jit
from numpy.typing import NDArray


@jit(
    int64[:](
        float64,
        int64,
        int64,
        int64,
        int64,
        float64[:],
        complex128[:],
        float64,
        int64,
        float64[:],
        float64[:, :],
        complex128[:],
        complex128[:, :],
    ),
    nopython=True,
)
def outpt(
    r: float,
    mdr: int,
    ndr: int,
    ndz: int,
    tlc: int,
    f3: NDArray[np.float64],
    u: NDArray[np.complex128],
    _dir: float,
    ir: int,
    tll: NDArray[np.float64],
    tlg: NDArray[np.float64],
    cpl: NDArray[np.complex128],
    cpg: NDArray[np.complex128],
) -> NDArray[np.int64]:
    """
    Output transmission loss and complex pressure.

    Complex pressure does not include the cylindrical spreading
    term 1/sqrt(r) or the phase term exp(-j*k0*r).
    """

    eps = 1e-20

    mdr += 1
    if mdr == ndr:
        mdr = 0
        tlc += 1
        cpl[tlc] = (1 - _dir) * f3[ir] * u[ir] + _dir * f3[ir + 1] * u[ir + 1]
        temp = 10 * np.log10(r + eps)
        tll[tlc] = -20 * np.log10(np.abs(cpl[tlc]) + eps) + temp

        for i in range(tlg.shape[0]):
            j = (i + 1) * ndz
            cpg[i, tlc] = u[j] * f3[j]
            tlg[i, tlc] = -20 * np.log10(np.abs(cpg[i, tlc]) + eps) + temp

    return np.array([mdr, tlc], dtype=np.int64)
