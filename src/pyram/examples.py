import numpy as np
import matplotlib.pyplot as plt

from pyram.PyRAM import PYRAM


def example1():
    pyram = PyRAM(
        freq=50,
        zs=50,
        zr=50,
        z_ss=np.array([0, 100, 400]),
        rp_ss=np.array([0, 25000]),
        cw=np.array([[1480, 1530], [1520, 1530], [1530, 1530]]),
        z_sb=np.array([0]),
        rp_sb=np.array([0]),
        cb=np.array([[1700]]),
        rhob=np.array([[1.5]]),
        attn=np.array([[0.5]]),
        rbzb=np.array([[0, 200], [40000, 400]]),
        # rmax=50000,
        # dr=500,
        # dz=2,
        # zmplt=500,
        # c0=1600,
    )
    pyram.run()
    r = pyram.vr / 1000
    plt.plot(r, -pyram.tll)

    plt.plot(r, -15 * np.log10(r * 1000))
    plt.xlabel("Range [km]")
    plt.ylabel("Tloss [dB re 1m]")
    plt.show()


def example2():

    pyram = PyRAM(
        freq=50,
        zs=4,
        zr=45,
        z_ss=np.array([0, 100, 400]),
        rp_ss=np.array([0, 25000]),
        cw=np.array([[1480, 1530], [1520, 1530], [1530, 1530]]),
        z_sb=np.array([0]),
        rp_sb=np.array([0]),
        cb=np.array([[1700]]),
        rhob=np.array([[1.5]]),
        attn=np.array([[0.5]]),
        rbzb=np.array([[0, 200], [40000, 400]]),
        rmax=500,
        dr=5,
        dz=2,
        zmplt=500,
        c0=1600,
    )
    pyram.run()
    r = pyram.vr
    if True:
        fig2, ax2 = plt.subplots(layout="constrained")
        z = pyram.vz
        tlg = pyram.tlg
        CS3 = ax2.contourf(r, z, tlg)
        ax2.invert_yaxis()
        fig2.colorbar(CS3)
    else:
        plt.plot(r, -pyram.tll)
        plt.plot(r, -20 * np.log10(r))
        plt.xlabel("Range [m]")
        plt.ylabel("Tloss [dB re 1m]")
    plt.show()


def example3():
    """
    Attributes
    ----------
    freq: Frequency (Hz).
    zs: Source depth (m).
    zr: Receiver depth (m).
    z_ss: Depths (m) for water sound speed values, NumPy 1D array.
    rp_ss: Ranges (m) for water sound speed values, NumPy 1D array.
    cw: Water sound speed values (m/s),
        Numpy 2D array, dimensions z_ss.size by rp_ss.size.
    z_sb: Depths for seabed parameter values, NumPy 1D array.
    rp_sb: Ranges (m) for seabed parameter, NumPy 1D array.
    cb: Seabed sound speed values (m/s),
        NumPy 2D array, dimensions z_sb.size by rp_sb.size.
    rhob: Seabed density values (g/cm3), same dimensions as cb
    attn: Seabed attenuation values (dB/wavelength), same dimensions as cb
    rbzb: Bathymetry (m), Numpy 2D array with columns of ranges and depths
    ---------
    kwargs...
    ---------
    np: Number of Pade terms. Defaults to _np_default.
    c0: Reference sound speed (m/s). Defaults to mean of 1st profile.
    dr: Calculation range step (m). Defaults to np times the wavelength.
    dz: Calculation depth step (m). Defaults to _dzf*wavelength.
    ndr: Number of range steps between outputs. Defaults to _ndr_default.
    ndz: Number of depth steps between outputs. Defaults to _ndz_default.
    zmplt: Maximum output depth (m). Defaults to maximum depth in rbzb.
    rmax: Maximum calculation range (m). Defaults to max in rp_ss or rp_sb.
    ns: Number of stability constraints. Defaults to _ns_default.
    rs: Maximum range of the stability constraints (m). Defaults to rmax.
    lyrw: Absorbing layer width (wavelengths). Defaults to _lyrw_default.
    NB: original zmax input not needed due to lyrw.
    id: Integer identifier for this instance.

    _np_default = 8
    _dzf = 0.1
    _ndr_default = 1
    _ndz_default = 1
    _ns_default = 1
    _lyrw_default = 20
    _id_default = 0
    """

    freqs = np.arange(10, 500, 5)
    tl = []
    zs = 4.0
    zr = 45.0
    rh = 100.0
    c0 = 1490.0
    for f in freqs:
        npdefault = 8 if f < 100 else 14
        pyram = PyRAM(
            freq=f,  # frequency
            zs=zs,  # Source depth (m)
            zr=zr,  # Receiver depth (m).
            z_ss=np.array([0.0, 2.2, 4.0, 6.3, 1000.0]),  # depths (m) for water sound speed values
            rp_ss=np.array([0.0]),  # ranges (m) for water sound speed values
            cw=np.array(
                [
                    [1466.0],  # Water sound speed values size z_ss.size X rp_ss
                    [1486.0],
                    [1490.0],
                    [1500.0],
                    [1500.0],
                ]
            ),
            z_sb=np.array([0.0, 20.0]),  # Depths for seabed parameter values
            rp_sb=np.array([0.0]),  # Ranges for seabed parameter values
            cb=np.array([[1700.0], [5200.0]]),  # Seabed sound speed values size z_sb.size X rp_sb
            rhob=np.array([[1.6], [2.6]]),  # Seabed density values size z_sb.size X rp_sb.size
            attn=np.array([[0.5], [0.1]]),  # Seabed attenuation values size z_sb.size X rp_sb.size
            rbzb=np.array(
                [
                    [0.0, 380.0],  # Bathymetry (m), Numpy 2D array with columns of ranges and depths
                    [2000.0, 380.0],
                ]
            ),
            rmax=500.0,
            dr=rh,
            # dz=dz,
            zmplt=380.0,
            c0=c0,
            np=npdefault,
        )
        pyram.run()
        tl.append(-pyram.tll[0])
        print(pyram.vr[0], pyram.vr[-1], pyram.vz[0], pyram.vz[-1])

    mu = 1

    r = np.hypot(rh, zr - zs)
    theta = np.abs(np.arctan2(zr, rh))
    f0 = c0 / (4 * zs * np.sin(theta))
    plt.semilogx(freqs, tl, label="Pyram")
    plt.semilogx(freqs, -20 * np.log10(r) * np.ones(len(freqs)), label="20log(r)")
    plt.semilogx(
        freqs, -20 * np.log10(r) + 10 * np.log10(1 + mu**2 - 2 * mu * np.cos(np.pi * freqs / f0)), label="LM"
    )
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Tloss [dB re 1m]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    example3()
