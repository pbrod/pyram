"""
PyRAM: Python adaptation of the Range-dependent Acoustic Model (RAM).
RAM was created by Michael D Collins at the US Naval Research Laboratory.
This adaptation is of RAM v1.5, available from the Ocean Acoustics Library at
https://oalib-acoustics.org/models-and-software/parabolic-equation

The purpose of PyRAM is to provide a version of RAM which can be used within a
Python interpreter environment (e.g. Spyder or the Jupyter notebook) and is
easier to understand, extend and integrate into other applications than the
Fortran version. It is written in pure Python and achieves speeds comparable to
native code by using the Numba library for JIT compilation.

The PyRAM class contains methods which largely correspond to the original
Fortran subroutines and functions (including retaining the same names). The
variable names are also mostly the same. However some of the original code
(e.g. subroutine zread) is unnecessary when the same purpose can be achieved
using available Python library functions (e.g. from NumPy or SciPy) and has
therefore been replaced.

A difference in functionality is that sound speed profile updates with range
are decoupled from seabed parameter updates, which provides more flexibility
in specifying the environment (e.g. if the data comes from different sources).

PyRAM also provides various conveniences, e.g. automatic calculation of range
and depth steps (though these can be overridden using keyword arguments).
"""

import warnings
from time import process_time
from typing import NamedTuple

import numpy as np

from pyram.matrc import matrc
from pyram.outpt import outpt
from pyram.solve import solve

__all__ = (
    "arctic_profile",
    "munk_profile",
    "PyRAM",
    "PyRAMResults",
)


class PyRAMResults(NamedTuple):
    """
    Results returned by PyRAM.run().

    Fields
    ------
    ranges : ndarray
        Calculation ranges [m].
    depths : ndarray
        Calculation depths [m].
    loss_grid : ndarray
        Transmission loss [dB] over depth and range.
    loss_line : ndarray
        Transmission loss [dB] at receiver depth.
    pressure_grid : ndarray
        Complex pressure field.
    pressure_line : ndarray
        Complex pressure at receiver depth.
    c0 : float
        Reference sound speed [m/s].
    proc_time : float
        Processing time [s].
    id : int
        Run identifier.
    """

    ranges: np.ndarray
    depths: np.ndarray
    loss_grid: np.ndarray
    loss_line: np.ndarray
    pressure_grid: np.ndarray
    pressure_line: np.ndarray
    c0: float
    proc_time: float
    id: int


def arctic_profile(
    z,
    c0=1440.0,
    gradient=0.02,
):
    """
    Simple Arctic sound-speed profile.

    Parameters
    ----------
    z : array_like
        Depth [m], positive downward.
    c0 : float
        Surface sound speed [m/s].
    gradient : float
        Vertical sound-speed gradient [m/s/m].
        gradient = 0.02 corresponds to an increase of
        approximately 20 m/s per km depth.

    Returns
    -------
    ndarray
        Sound speed [m/s].

    Notes
    -----
    This is not a standard published sound-speed profile.
    It is a simple first-order approximation motivated by the
    approximately monotonic increase of sound speed with depth
    observed in many Arctic and Antarctic water columns.

    References
    ----------
    Munk, W., Worcester, P., and Wunsch, C.
    Ocean Acoustic Tomography.
    Cambridge University Press, 1995.
    """
    if c0 <= 0:
        raise ValueError("c0 must be positive")
    if gradient < 0:
        warnings.warn(
            "Negative gradient produces decreasing sound speed with depth.",
            stacklevel=2,
        )
    z = np.asarray(z, dtype=float)

    return c0 + gradient * z


def munk_profile(
    z,
    c0=1500.0,
    epsilon=0.00737,
    z_axis=1300.0,
    scale_depth=1300.0,
):
    """
    Canonical Munk sound-speed profile.

    Parameters
    ----------
    z : array_like
        Depth [m], positive downward.
    c0 : float, optional
        Reference sound speed [m/s].
    epsilon : float, optional
        Dimensionless profile-strength parameter.
    z_axis : float, optional
        Sound-channel axis depth [m].
    scale_depth : float, optional
        Scale depth [m].

    Returns
    -------
    ndarray
        Sound speed [m/s].

    Notes
    -----
    The Munk profile is defined as

        c(z) = c0 * [1 + ε * (η + exp(-η) - 1)]

    where

        η = 2 * (z - z_axis) / scale_depth.

    References
    ----------
    Munk, W.
    "Sound channel in an exponentially stratified ocean,
    with application to SOFAR."
    Journal of the Acoustical Society of America,
    55(2), 1974.

    Munk, W., Worcester, P., and Wunsch, C.
    Ocean Acoustic Tomography.
    Cambridge University Press, 1995.
    """
    if scale_depth <= 0:
        raise ValueError("scale_depth must be positive")
    if c0 <= 0:
        raise ValueError("c0 must be positive")
    if z_axis < 0:
        raise ValueError("z_axis must be non-negative")
    if epsilon < 0:
        warnings.warn(
            "Negative epsilon produces an inverted Munk profile.",
            stacklevel=2,
        )

    z = np.asarray(z, dtype=float)

    eta = 2.0 * (z - z_axis) / scale_depth

    return c0 * (1.0 + epsilon * (eta + np.expm1(-eta)))


class PyRAM:
    """
    Range-dependent Acoustic Model (RAM)

    Parameters
    ----------
    freq : float
        Acoustic frequency [Hz].
    zs : float
        Source depth [m].
    zr : float
        Receiver depth [m].
    z_ss : ndarray
        Depths [m] corresponding to water sound-speed values.
    rp_ss : ndarray
        Ranges [m] corresponding to water sound-speed values.
    cw : ndarray
        Water sound-speed values [m/s], with shape
        ``(z_ss.size, rp_ss.size)``.
    z_sb : ndarray
        Depths [m] corresponding to seabed property values.
    rp_sb : ndarray
        Ranges [m] corresponding to seabed property values.
    cb : ndarray
        Seabed sound-speed values [m/s], with shape
        ``(z_sb.size, rp_sb.size)``.
    rhob : ndarray
        Seabed density values [g/cm³], with the same shape as
        ``cb``.
    attn : ndarray
        Seabed attenuation values [dB/wavelength], with the same
        shape as ``cb``.
    rbzb : ndarray
        Bathymetry array [m], with columns containing range and
        depth pairs.

    Other Parameters
    ----------------
    np : int, optional
        Number of Padé approximation terms. Defaults to ``_np_default`` (8).
    c0 : float, optional
        Reference sound speed [m/s]. Defaults to the mean of the
        first water sound-speed profile.
    dr : float, optional
        Calculation range step [m]. Defaults to ``0.5`` times the
        acoustic wavelength.
    dz : float, optional
        Calculation depth step [m]. Defaults to
        ``0.05`` times the acoustic wavelength.
    ndr : int, optional
        Number of range steps between outputs. Defaults to
        ``_ndr_default`` (1).
    ndz : int, optional
        Number of depth steps between outputs. Defaults to
        ``_ndz_default`` (1).
    zmplt : float, optional
        Maximum output depth [m]. Defaults to the maximum depth in
        ``rbzb``.
    rmax : float, optional
        Maximum calculation range [m]. Defaults to the maximum
        range in ``rp_ss``, ``rp_sb``, or ``rbzb``.
    ns : int, optional
        Number of stability constraints. Defaults to
        ``_ns_default`` (1).
    rs : float, optional
        Maximum range [m] over which stability constraints are
        applied. Defaults to ``rmax``.
    lyrw : float, optional
        Width of the absorbing layer [wavelengths]. Defaults to
        ``_lyrw_default`` (20).
    id : int, optional
        Integer identifier for the model instance. Defaults to
        ``_id_default`` (0).

    Notes
    -----
    RAM is intended primarily for low-frequency acoustic propagation
    (typically below approximately 500 Hz) in range-dependent environments
    consisting of fluid layers and neglecting seabed shear waves.

    The numerical accuracy is primarily controlled by ``np``, ``dr``, and ``dz``.
    Increasing the number of Padé terms generally improves accuracy at
    the expense of increased computational cost. The number of stability
    constraints (``ns``) can be increased if the solution exhibits
    numerical instability for any combination of ``np``, ``dr`` and ``dz``.

    General rules for using RAM:

    As a rule of thumb use:

       ``dr`` <= 0.66 * wavelength           (1)
       ``dz`` <= 0.066 * wavelength          (2)

    The default values of ``dr`` and ``dz`` satisfy these recommendations.
    The allowable range-step size is limited by the degree of range
    dependence in the environment. The RAM PE solver permits arbitrarily
    large range steps of many wavelengths for range-independent regions
    and dense range sampling. For an environment with small range variations,
    a solution computed using the range-step specifications defined by
    Eq. (1) and sampled on a fine scale, is barely distinguishable from a
    solution computed with large range steps (``dr = 50 * wavelength``),
    although sampled more sparsely. Calculations with a range step size that
    satisfies Eq. (1) are computationally inefficient for environments with
    small variations, but will handle a greater range of environments without
    the hassle of having to evaluate the rate of range dependency and can
    provide a solution that is sampled on a fine scale.

    Occasionally you will see output that looks like the computation stopped
    halfway or the data will be blank. This likely means there was an
    instability in the calculation. The usual remedy is to vary ``dr`` until
    a stable solution is obtained. If varying ``dr`` does not eliminate
    the instability, try increasing ``ns`` above 1.

    It is good to refine your grid (make ``dr`` and ``dz`` smaller) and increase
    the number of Padé terms to ensure that the solution has converged.
    Experience will guide you in deciding numerical parameters given a frequency
    and geoacoustic environment.

    The RAM model approximates a semi-infinite bottom half-space by
    appending an artificial high-attenuation layer, also called a sponge layer,
    at the lower boundary of the physical domain in order to prevent spurious
    reflections from the lower computational boundary. The thickness of this
    layer is controlled by ``lyrw``, which defaults to 20 wavelengths.

    References
    ----------
    [1] M. D. Collins (1993), "A split-step Padé solution for the parabolic equation method",
        J. Acoust. Soc. Am., vol. 93, pp. 1736-1742
    [2] M. D. Collins (2015), "User's Guide for RAM Version 1.0 and 1.0p",
        Naval Research Laboratory.
        [Online]. Available:
        http://staff.washington.edu/dushaw/AcousticsCode/ram.pdf
    [3] D. Calvo (2006), "Quick introduction to using the Naval Research Laboratory RAM parabolic
        equation (PE) code that includes bottom loss",
        Naval Research Laboratory.
        [Online]. Available:
        http://oalib.hlsresearch.com/PE/ramsurf/readme.orig
    [4] M. D. Collins (1994), "Generalization of the split-step Padé solution",
        J. Acoust. Soc. Am., vol. 96, pp. 382-385
    """

    _np_default = 8
    _ndr_default = 1
    _ndz_default = 1
    _ns_default = 1
    _lyrw_default = 20
    _id_default = 0

    def __init__(self, freq, zs, zr, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb, **kwargs):
        self._freq, self._zs, self._zr = freq, zs, zr
        self.check_inputs(z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb)
        self.set_params(**kwargs)

    def run(self):
        """
        Run the acoustic propagation model.

        Returns
        -------
        PyRAMResults
            Model output containing ranges, depths, transmission loss,
            complex pressure, reference sound speed, processing time,
            and run identifier.
        """

        t0 = process_time()

        self.setup()

        nr = int(np.round(self._rmax / self._dr)) - 1

        for rn in range(nr):
            self.updat()

            solve(
                self.u,
                self.v,
                self.s1,
                self.s2,
                self.s3,
                self.r1,
                self.r2,
                self.r3,
                self.iz,
                self.nz,
                self._np,
            )

            self.r = (rn + 2) * self._dr

            self.mdr, self.tlc = outpt(
                self.r,
                self.mdr,
                self._ndr,
                self._ndz,
                self.tlc,
                self.f3,
                self.u,
                self.dir,
                self.ir,
                self.tll,
                self.tlg,
                self.cpl,
                self.cpg,
            )[:]

        self.proc_time = process_time() - t0

        return PyRAMResults(
            ranges=self.vr,
            depths=self.vz,
            loss_grid=self.tlg,
            loss_line=self.tll,
            pressure_grid=self.cpg,
            pressure_line=self.cpl,
            c0=self._c0,
            proc_time=self.proc_time,
            id=self._id,
        )

    def check_inputs(self, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb):
        """Validate and store model inputs."""

        # Source and receiver depths
        if not z_ss[0] <= self._zs <= z_ss[-1]:
            raise ValueError("Source depth outside sound speed depths")

        if not z_ss[0] <= self._zr <= z_ss[-1]:
            raise ValueError("Receiver depth outside sound speed depths")

        # Water sound-speed profiles
        if cw.shape != (z_ss.size, rp_ss.size):
            raise ValueError("Dimensions of z_ss, rp_ss, and cw must be consistent.")

        # Seabed profiles
        expected_shape = (z_sb.size, rp_sb.size)

        for _name, profile in (
            ("cb", cb),
            ("rhob", rhob),
            ("attn", attn),
        ):
            if profile.shape != expected_shape:
                raise ValueError("Dimensions of z_sb, rp_sb, cb, rhob, and attn must be consistent.")

        # Bathymetry
        if rbzb[:, 1].max() > z_ss[-1]:
            raise ValueError("Deepest sound speed point must be at or below deepest bathymetry point.")

        # Store copies to avoid modifying caller-owned arrays
        self._z_ss = np.array(z_ss, copy=True)
        self._rp_ss = np.array(rp_ss, copy=True)
        self._cw = np.array(cw, copy=True)

        self._z_sb = np.array(z_sb, copy=True)
        self._rp_sb = np.array(rp_sb, copy=True)
        self._cb = np.array(cb, copy=True)
        self._rhob = np.array(rhob, copy=True)
        self._attn = np.array(attn, copy=True)

        self._rbzb = np.array(rbzb, copy=True)

        # Range-dependence flags
        self.rd_ss = self._rp_ss.size > 1
        self.rd_sb = self._rp_sb.size > 1
        self.rd_bt = self._rbzb.shape[0] > 1

    def set_params(self, **kwargs):
        """Set the parameters from the keyword arguments"""

        self._np = kwargs.get("np", PyRAM._np_default)

        c0 = np.mean(self._cw[:, 0]) if len(self._cw.shape) > 1 else np.mean(self._cw)
        self._c0 = kwargs.get("c0", c0)

        self._lambda = lambda0 = self._c0 / self._freq

        # dr and dz are based on c0 to get sensible output steps
        self._dr = kwargs.get("dr", 0.5 * lambda0)
        self._dz = kwargs.get("dz", 0.05 * lambda0)

        self._ndr = kwargs.get("ndr", PyRAM._ndr_default)
        self._ndz = kwargs.get("ndz", PyRAM._ndz_default)

        self._zmplt = kwargs.get("zmplt", self._rbzb[:, 1].max())

        self._rmax = kwargs.get(
            "rmax", np.max([self._rp_ss.max(), self._rp_sb.max(), self._rbzb[:, 0].max()])
        )

        self._ns = kwargs.get("ns", PyRAM._ns_default)
        self._rs = kwargs.get("rs", self._rmax + self._dr)

        self._lyrw = kwargs.get("lyrw", PyRAM._lyrw_default)

        self._id = kwargs.get("id", PyRAM._id_default)

        self.proc_time = None

    def setup(self):
        """Initialise the parameters, acoustic field, and matrices"""

        if self._rbzb[-1, 0] < self._rmax:
            self._rbzb = np.append(self._rbzb, np.array([[self._rmax, self._rbzb[-1, 1]]]), axis=0)

        self.eta = 1 / (40 * np.pi * np.log10(np.exp(1)))
        self.ib = 0  # Bathymetry pair index
        self.mdr = 0  # Output range counter
        self.r = self._dr
        self.omega = 2 * np.pi * self._freq
        ri = self._zr / self._dz
        self.ir = int(np.floor(ri))  # Receiver depth index
        self.dir = ri - self.ir  # Offset
        self.k0 = self.omega / self._c0
        self._z_sb += self._z_ss[-1]  # Make seabed profiles relative to deepest water profile point
        self._zmax = self._z_sb.max() + self._lyrw * self._lambda
        self.nz = int(np.floor(self._zmax / self._dz)) - 1  # Number of depth grid points - 2
        self.nzplt = int(np.floor(self._zmplt / self._dz))  # Deepest output grid point
        self.iz = int(np.floor(self._rbzb[0, 1] / self._dz))  # First index below seabed
        self.iz = max(1, self.iz)
        self.iz = min(self.nz - 1, self.iz)

        self.u = np.zeros(self.nz + 2, dtype=np.complex128)
        self.v = np.zeros(self.nz + 2, dtype=np.complex128)
        self.ksq = np.zeros(self.nz + 2, dtype=np.complex128)
        self.ksqb = np.zeros(self.nz + 2, dtype=np.complex128)
        self.r1 = np.zeros([self.nz + 2, self._np], dtype=np.complex128)
        self.r2 = np.zeros([self.nz + 2, self._np], dtype=np.complex128)
        self.r3 = np.zeros([self.nz + 2, self._np], dtype=np.complex128)
        self.s1 = np.zeros([self.nz + 2, self._np], dtype=np.complex128)
        self.s2 = np.zeros([self.nz + 2, self._np], dtype=np.complex128)
        self.s3 = np.zeros([self.nz + 2, self._np], dtype=np.complex128)
        self.pd1 = np.zeros(self._np, dtype=np.complex128)
        self.pd2 = np.zeros(self._np, dtype=np.complex128)

        self.alpw = np.zeros(self.nz + 2)
        self.alpb = np.zeros(self.nz + 2)
        self.f1 = np.zeros(self.nz + 2)
        self.f2 = np.zeros(self.nz + 2)
        self.f3 = np.zeros(self.nz + 2)
        self.ksqw = np.zeros(self.nz + 2)
        nvr = int(np.floor(self._rmax / (self._dr * self._ndr)))
        self._rmax = nvr * self._dr * self._ndr
        nvz = int(np.floor(self.nzplt / self._ndz))
        self.vr = np.arange(1, nvr + 1) * self._dr * self._ndr
        self.vz = np.arange(1, nvz + 1) * self._dz * self._ndz
        self.tll = np.zeros(nvr)
        self.tlg = np.zeros([nvz, nvr])
        self.cpl = np.zeros(nvr) * 1j
        self.cpg = np.zeros([nvz, nvr]) * 1j
        self.tlc = -1  # TL output range counter

        self.ss_ind = 0  # Sound speed profile range index
        self.sb_ind = 0  # Seabed parameters range index
        self.bt_ind = 0  # Bathymetry range index

        # The initial profiles and starting field
        self.profl()
        self.selfs()
        self.mdr, self.tlc = outpt(
            self.r,
            self.mdr,
            self._ndr,
            self._ndz,
            self.tlc,
            self.f3,
            self.u,
            self.dir,
            self.ir,
            self.tll,
            self.tlg,
            self.cpl,
            self.cpg,
        )[:]

        # The propagation matrices
        self.epade()
        matrc(
            self.k0,
            self._dz,
            self.iz,
            self.iz,
            self.nz,
            self._np,
            self.f1,
            self.f2,
            self.f3,
            self.ksq,
            self.alpw,
            self.alpb,
            self.ksqw,
            self.ksqb,
            self.rhob,
            self.r1,
            self.r2,
            self.r3,
            self.s1,
            self.s2,
            self.s3,
            self.pd1,
            self.pd2,
        )

    def profl(self):
        """Set up the profiles"""

        attnf = 10  # 10dB/wavelength at floor

        z = np.linspace(0, self._zmax, self.nz + 2)
        self.cw = np.interp(
            z,
            self._z_ss,
            self._cw[:, self.ss_ind],
            left=self._cw[0, self.ss_ind],
            right=self._cw[-1, self.ss_ind],
        )
        self.cb = np.interp(
            z,
            self._z_sb,
            self._cb[:, self.sb_ind],
            left=self._cb[0, self.sb_ind],
            right=self._cb[-1, self.sb_ind],
        )
        self.rhob = np.interp(
            z,
            self._z_sb,
            self._rhob[:, self.sb_ind],
            left=self._rhob[0, self.sb_ind],
            right=self._rhob[-1, self.sb_ind],
        )
        attnlyr = np.concatenate((self._attn[:, self.sb_ind], [self._attn[-1, self.sb_ind], attnf]))
        zlyr = np.concatenate(
            (
                self._z_sb,
                [
                    self._z_sb[-1] + 0.75 * self._lyrw * self._lambda,
                    self._z_sb[-1] + self._lyrw * self._lambda,
                ],
            )
        )
        self.attn = np.interp(z, zlyr, attnlyr, left=self._attn[0, self.sb_ind], right=attnf)

        self.ksqw = (self.omega / self.cw) ** 2 - self.k0**2
        self.ksqb = ((self.omega / self.cb) * (1 + 1j * self.eta * self.attn)) ** 2 - self.k0**2
        self.alpw = np.sqrt(self.cw / self._c0)
        self.alpb = np.sqrt(self.rhob * self.cb / self._c0)

    def updat(self):
        """Matrix updates"""

        # Varying bathymetry
        if self.rd_bt:
            npt = self._rbzb.shape[0]
            while (self.bt_ind < npt - 1) and (self.r >= self._rbzb[self.bt_ind + 1, 0]):
                self.bt_ind += 1
            jz = self.iz
            z = self._rbzb[self.bt_ind, 1] + (self.r + 0.5 * self._dr - self._rbzb[self.bt_ind, 0]) * (
                self._rbzb[self.bt_ind + 1, 1] - self._rbzb[self.bt_ind, 1]
            ) / (self._rbzb[self.bt_ind + 1, 0] - self._rbzb[self.bt_ind, 0])
            self.iz = int(np.floor(z / self._dz))  # First index below seabed
            self.iz = max(1, self.iz)
            self.iz = min(self.nz - 1, self.iz)
            if self.iz != jz:
                matrc(
                    self.k0,
                    self._dz,
                    self.iz,
                    jz,
                    self.nz,
                    self._np,
                    self.f1,
                    self.f2,
                    self.f3,
                    self.ksq,
                    self.alpw,
                    self.alpb,
                    self.ksqw,
                    self.ksqb,
                    self.rhob,
                    self.r1,
                    self.r2,
                    self.r3,
                    self.s1,
                    self.s2,
                    self.s3,
                    self.pd1,
                    self.pd2,
                )

        # Varying sound speed profile
        if self.rd_ss:
            npt = self._rp_ss.size
            ss_ind_o = self.ss_ind
            while (self.ss_ind < npt - 1) and (self.r >= self._rp_ss[self.ss_ind + 1]):
                self.ss_ind += 1
            if self.ss_ind != ss_ind_o:
                self.profl()
                matrc(
                    self.k0,
                    self._dz,
                    self.iz,
                    self.iz,
                    self.nz,
                    self._np,
                    self.f1,
                    self.f2,
                    self.f3,
                    self.ksq,
                    self.alpw,
                    self.alpb,
                    self.ksqw,
                    self.ksqb,
                    self.rhob,
                    self.r1,
                    self.r2,
                    self.r3,
                    self.s1,
                    self.s2,
                    self.s3,
                    self.pd1,
                    self.pd2,
                )

        # Varying seabed profile
        if self.rd_sb:
            npt = self._rp_sb.size
            sb_ind_o = self.sb_ind
            while (self.sb_ind < npt - 1) and (self.r >= self._rp_sb[self.sb_ind + 1]):
                self.sb_ind += 1
            if self.sb_ind != sb_ind_o:
                self.profl()
                matrc(
                    self.k0,
                    self._dz,
                    self.iz,
                    self.iz,
                    self.nz,
                    self._np,
                    self.f1,
                    self.f2,
                    self.f3,
                    self.ksq,
                    self.alpw,
                    self.alpb,
                    self.ksqw,
                    self.ksqb,
                    self.rhob,
                    self.r1,
                    self.r2,
                    self.r3,
                    self.s1,
                    self.s2,
                    self.s3,
                    self.pd1,
                    self.pd2,
                )

        # Turn off the stability constraints
        if self.r >= self._rs:
            self._ns = 0
            self._rs = self._rmax + self._dr
            self.epade()
            matrc(
                self.k0,
                self._dz,
                self.iz,
                self.iz,
                self.nz,
                self._np,
                self.f1,
                self.f2,
                self.f3,
                self.ksq,
                self.alpw,
                self.alpb,
                self.ksqw,
                self.ksqb,
                self.rhob,
                self.r1,
                self.r2,
                self.r3,
                self.s1,
                self.s2,
                self.s3,
                self.pd1,
                self.pd2,
            )

    def selfs(self):
        """Set up the initial field. The self-starter"""

        # Conditions for the delta function

        si = self._zs / self._dz
        _is = int(np.floor(si))  # Source depth index
        dis = si - _is  # Offset

        self.u[_is] = (1 - dis) * np.sqrt(2 * np.pi / self.k0) / (self._dz * self.alpw[_is])
        self.u[_is + 1] = dis * np.sqrt(2 * np.pi / self.k0) / (self._dz * self.alpw[_is])

        # Divide the delta function by (1-X)**2 to get a smooth rhs

        self.pd1[0] = 0
        self.pd2[0] = -1

        matrc(
            self.k0,
            self._dz,
            self.iz,
            self.iz,
            self.nz,
            1,
            self.f1,
            self.f2,
            self.f3,
            self.ksq,
            self.alpw,
            self.alpb,
            self.ksqw,
            self.ksqb,
            self.rhob,
            self.r1,
            self.r2,
            self.r3,
            self.s1,
            self.s2,
            self.s3,
            self.pd1,
            self.pd2,
        )
        for _ in range(2):
            solve(self.u, self.v, self.s1, self.s2, self.s3, self.r1, self.r2, self.r3, self.iz, self.nz, 1)

        # Apply the operator (1-X)**2*(1+X)**(-1/4)*exp(ci*k0*r*sqrt(1+X))

        self.epade(ip=2)
        matrc(
            self.k0,
            self._dz,
            self.iz,
            self.iz,
            self.nz,
            self._np,
            self.f1,
            self.f2,
            self.f3,
            self.ksq,
            self.alpw,
            self.alpb,
            self.ksqw,
            self.ksqb,
            self.rhob,
            self.r1,
            self.r2,
            self.r3,
            self.s1,
            self.s2,
            self.s3,
            self.pd1,
            self.pd2,
        )
        solve(
            self.u, self.v, self.s1, self.s2, self.s3, self.r1, self.r2, self.r3, self.iz, self.nz, self._np
        )

    def epade(self, ip=1):
        """Set the coefficients of the rational approximation"""

        n = 2 * self._np
        _bin = np.zeros([n + 1, n + 1])
        a = np.zeros([n + 1, n + 1], dtype=np.complex128)
        b = np.zeros(n, dtype=np.complex128)
        dg = np.zeros(n + 1, dtype=np.complex128)
        dh1 = np.zeros(n, dtype=np.complex128)
        dh2 = np.zeros(n, dtype=np.complex128)
        dh3 = np.zeros(n, dtype=np.complex128)
        fact = np.zeros(n + 1)
        sig = self.k0 * self._dr

        if ip == 1:
            nu, alp = 0, 0
        else:
            nu, alp = 1, -0.25

        # The factorials
        fact[0] = 1
        for i in range(1, n):
            fact[i] = (i + 1) * fact[i - 1]

        # The binomial coefficients
        for i in range(n + 1):
            _bin[i, 0] = 1
            _bin[i, i] = 1
        for i in range(2, n + 1):
            for j in range(1, i):
                _bin[i, j] = _bin[i - 1, j - 1] + _bin[i - 1, j]

        # The accuracy constraints
        dg, dh1, dh2, dh3 = self.deriv(n, sig, alp, dg, dh1, dh2, dh3, _bin, nu)
        for i in range(n):
            b[i] = dg[i + 1]
        for i in range(n):
            if 2 * i <= n - 1:
                a[i, 2 * i] = fact[i]
            for j in range(i + 1):
                if 2 * j + 1 <= n - 1:
                    a[i, 2 * j + 1] = -_bin[i + 1, j + 1] * fact[j] * dg[i - j]

        # The stability constraints

        if self._ns >= 1:
            z1 = -3 + 0j
            b[n - 1] = -1
            for j in range(self._np):
                a[n - 1, 2 * j] = z1 ** (j + 1)
                a[n - 1, 2 * j + 1] = 0

        if self._ns >= 2:
            z1 = -1.5 + 0j
            b[n - 2] = -1
            for j in range(self._np):
                a[n - 2, 2 * j] = z1 ** (j + 1)
                a[n - 2, 2 * j + 1] = 0

        a, b = self.gauss(n, a, b, self.pivot)

        dh1[0] = 1
        for j in range(self._np):
            dh1[j + 1] = b[2 * j]
        dh1, dh2 = self.fndrt(dh1, self._np, dh2, self.guerre)
        for j in range(self._np):
            self.pd1[j] = -1 / dh2[j]

        dh1[0] = 1
        for j in range(self._np):
            dh1[j + 1] = b[2 * j + 1]
        dh1, dh2 = self.fndrt(dh1, self._np, dh2, self.guerre)
        for j in range(self._np):
            self.pd2[j] = -1 / dh2[j]

    @staticmethod
    def deriv(n, sig, alp, dg, dh1, dh2, dh3, _bin, nu):
        """Return the derivatives of the operator function at x=0"""

        dh1[0] = 0.5 * 1j * sig
        exp1 = -0.5
        dh2[0] = alp
        exp2 = -1
        dh3[0] = -2 * nu
        exp3 = -1
        for i in range(1, n):
            dh1[i] = dh1[i - 1] * exp1
            exp1 -= 1
            dh2[i] = dh2[i - 1] * exp2
            exp2 -= 1
            dh3[i] = -nu * dh3[i - 1] * exp3
            exp3 -= 1

        dg[0] = 1
        dg[1] = dh1[0] + dh2[0] + dh3[0]
        for i in range(1, n):
            dg[i + 1] = dh1[i] + dh2[i] + dh3[i]
            for j in range(i):
                dg[i + 1] += _bin[i, j] * (dh1[j] + dh2[j] + dh3[j]) * dg[i - j]

        return dg, dh1, dh2, dh3

    @staticmethod
    def gauss(n, a, b, pivot):
        """
        Gaussian elimination
        """

        # Downward elimination
        for i in range(n):
            if i < n - 1:
                a, b = pivot(n, i, a, b)
            a[i, i] = 1 / a[i, i]
            b[i] *= a[i, i]
            if i < n - 1:
                for j in range(i + 1, n + 1):
                    a[i, j] *= a[i, i]
                for k in range(i + 1, n):
                    b[k] -= a[k, i] * b[i]
                    for j in range(i + 1, n):
                        a[k, j] -= a[k, i] * a[i, j]

        # Back substitution
        for i in range(n - 2, -1, -1):
            for j in range(i + 1, n):
                b[i] -= a[i, j] * b[j]

        return a, b

    @staticmethod
    def pivot(n, i, a, b):
        """
        Rows are interchanged for stability
        """

        i0 = i
        amp0 = np.abs(a[i, i])
        for j in range(i + 1, n):
            amp = np.abs(a[j, i])
            if amp > amp0:
                i0 = j
                amp0 = amp

        if i0 != i:
            b[i0], b[i] = b[i], b[i0]
            for j in range(i, n + 1):
                a[i0, j], a[i, j] = a[i, j], a[i0, j]

        return a, b

    @staticmethod
    def fndrt(a, n, z, guerre):
        """Find the roots of polynomial a"""

        if n == 1:
            z[0] = -a[0] / a[1]
            return a, z

        if n != 2:
            for k in range(n - 1, 1, -1):
                # Obtain an approximate root
                root = 0
                err = 1e-12
                a, root, err = guerre(a, k + 1, root, err, 1000)
                # Refine the root by iterating five more times
                err = 0
                a, root, err = guerre(a, k + 1, root, err, 5)
                z[k] = root
                # Divide out the factor (z-root).
                for i in range(k, -1, -1):
                    a[i] += root * a[i + 1]
                for i in range(k + 1):
                    a[i] = a[i + 1]

        z[1] = 0.5 * (-a[1] + np.sqrt(a[1] ** 2 - 4 * a[0] * a[2])) / a[2]
        z[0] = 0.5 * (-a[1] - np.sqrt(a[1] ** 2 - 4 * a[0] * a[2])) / a[2]

        return a, z

    @staticmethod
    def guerre(a, n, z, err, nter):
        """
        Return the root of a polynomial of degree n > 2 by Laguerre's method
        """

        az = np.zeros(n, dtype=np.complex128)
        azz = np.zeros(n - 1, dtype=np.complex128)

        eps = 1e-20
        # The coefficients of p'(z) and p''(z)
        for i in range(n):
            az[i] = (i + 1) * a[i + 1]
        for i in range(n - 1):
            azz[i] = (i + 1) * az[i + 1]

        _iter = 0
        jter = 0  # Missing from original code - assume this is correct
        dz = np.inf

        while (np.abs(dz) > err) and (_iter < nter - 1):
            p = a[n - 1] + a[n] * z
            for i in range(n - 2, -1, -1):
                p = a[i] + z * p
            if np.abs(p) < eps:
                return a, z, err

            pz = az[n - 2] + az[n - 1] * z
            for i in range(n - 3, -1, -1):
                pz = az[i] + z * pz

            pzz = azz[n - 3] + azz[n - 2] * z
            for i in range(n - 4, -1, -1):
                pzz = azz[i] + z * pzz

            # The Laguerre perturbation
            f = pz / p
            g = f**2 - pzz / p
            h = np.sqrt((n - 1) * (n * g - f**2))
            amp1 = np.abs(f + h)
            amp2 = np.abs(f - h)
            if amp1 > amp2:
                dz = -n / (f + h)
            else:
                dz = -n / (f - h)

            _iter += 1

            # Rotate by 90 degrees to avoid limit cycles

            jter += 1
            if jter == 9:
                jter = 0
                dz *= 1j
            z += dz

            if _iter == 100:
                raise ValueError("Laguerre method not converging. Try a different combination of DR and NP.")

        return a, z, err
