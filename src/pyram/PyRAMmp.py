"""
PyRAMmp class definition
"""

from multiprocessing.pool import Pool
from time import sleep
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from pyram.PyRAM import PyRAM, PyRAMResults

FloatArray = NDArray[np.float64]


class PyRAMArgs(TypedDict):
    freq: float
    zs: float
    zr: float
    z_ss: FloatArray
    rp_ss: FloatArray
    cw: FloatArray
    z_sb: FloatArray
    rp_sb: FloatArray
    cb: FloatArray
    rhob: FloatArray
    attn: FloatArray
    rbzb: FloatArray


PyRAMKwargs = dict[str, Any]
PyRAMRun = tuple[PyRAMArgs, PyRAMKwargs]


def run_pyram(run: PyRAMRun) -> PyRAMResults:
    """
    Add a new PyRAM run (needs to be a function rather than a class method)
    """

    args, kwargs = run
    return PyRAM(
        freq=args["freq"],
        zs=args["zs"],
        zr=args["zr"],
        z_ss=args["z_ss"],
        rp_ss=args["rp_ss"],
        cw=args["cw"],
        z_sb=args["z_sb"],
        rp_sb=args["rp_sb"],
        cb=args["cb"],
        rhob=args["rhob"],
        attn=args["attn"],
        rbzb=args["rbzb"],
        **kwargs,
    ).run()


class PyRAMmp:
    """
    The PyRAMmp class sets up and runs a multiprocessing pool to enable
    parallel PyRAM model runs
    """

    def __init__(
        self,
        processes: int | None = None,
        maxtasksperchild: int | None = None,
    ) -> None:
        """
        Initialise the pool and variable lists
        processes and maxtasksperchild are passed to the pool
        """

        self.pool: Pool = Pool(processes=processes, maxtasksperchild=maxtasksperchild)
        self.results: list[PyRAMResults] = []  # Results from PyRAM.run()
        self._outputs: list[PyRAMResults] = []  # New outputs from PyRAM.run() for transfer to self.results
        self._waiting: list[PyRAMRun] = []  # Waiting runs
        self._num_processes = (
            processes if processes is not None else self.pool._processes  # type: ignore[attr-defined]
        )
        self._num_waiting: int = 0  # Number of waiting runs
        self._num_active: int = 0  # Number of active runs
        self._sleep_time: float = 1e-2  # Minimum sleep time between adding runs to pool
        self._new: bool = True  # Flag to indicate ready for new set of runs

    @property
    def num_processes(self) -> int:
        return self._num_processes

    def submit_runs(self, runs: list[PyRAMRun]) -> None:
        """
        Submit new runs to the pool as resources become available
        runs is a list of PyRAM input tuples (args, kwargs)
        """

        # Add to waiting list
        for run in runs:
            self._waiting.append(run)
        self._num_waiting = len(self._waiting)

        # Check how many active runs have finished
        for _ in range(len(self._outputs)):
            output = self._outputs.pop(0)
            self.results.append(output)
            self._num_active -= 1

        num_start = self._num_processes - self._num_active
        num_start = min(num_start, self._num_waiting)

        # Start new runs if processes are free
        for _ in range(num_start):
            run = self._waiting.pop(0)
            self.pool.apply_async(run_pyram, args=(run,), callback=self._get_output)
            self._num_active += 1

        if self._new:
            self._new = False
            self._wait()

    def _wait(self) -> None:
        """
        Wait for all submitted runs to complete
        """

        while self._num_active > 0:
            self.submit_runs([])
            sleep(self._sleep_time)

        self._new = True

    def close(self) -> None:
        """
        Close the pool and wait for all processes to finish
        """

        self.pool.close()
        self.pool.join()

    def _get_output(self, output: PyRAMResults) -> None:
        """
        Get a PyRAM output
        """

        self._outputs.append(output)

    def __del__(self) -> None:

        self.close()
