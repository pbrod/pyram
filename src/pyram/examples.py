import matplotlib.pyplot as plt
import numpy as np

from pyram.PyRAM import PyRAM, arctic_profile, munk_profile


def example1():
    """Plot long range transimission loss vs 15log(r). Same testcase as used in test_pyram"""
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
        rmax=50000,
        dr=500,
        dz=2,
        zmplt=500,
        c0=1600,
    )
    res = pyram.run()

    print(res.proc_time)

    r = res.ranges / 1000
    plt.plot(r, -res.loss_line)
    plt.plot(r, -15 * np.log10(r * 1000))
    plt.xlabel("Range [km]")
    plt.ylabel("Tloss [dB re 1m]")
    plt.show()


def example2():
    """Transmission loss contour-plto for f=50Hz"""
    pyram2 = PyRAM(
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
        dr=10,
        dz=0.2,
        zmplt=500,
        c0=1500,
    )
    res = pyram2.run()
    r = res.ranges
    print(res.proc_time)
    if True:
        fig2, ax2 = plt.subplots(layout="constrained")
        z = res.depths
        tlg = res.loss_grid
        CS3 = ax2.contourf(r, z, tlg, levels=list(range(0, 121, 15)))
        ax2.invert_yaxis()
        fig2.colorbar(CS3)
    else:
        tll = res.loss_line
        plt.plot(r, -tll)
        plt.plot(r, -20 * np.log10(r))
        plt.xlabel("Range [m]")
        plt.ylabel("Tloss [dB re 1m]")
    plt.show()


def example3():
    """
    Plot RAM vs Lloyd mirror transmisson loss for all frequencies from 10 to 1kHz
    """

    freqs = np.arange(10, 500, 5)
    freqs = np.logspace(1, 3, 50)
    tl = []
    zs = 4.0
    zr = 45.0
    rh = 20.0
    dr = 1
    dz = 0.2
    c0 = 1490.0
    water_depth = 46
    for f in freqs:
        npdefault = 10 if f < 50 else 4
        lambda0 = c0 / f
        dz = lambda0 / 20
        pyram3 = PyRAM(
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
                    [0.0, water_depth],  # Bathymetry (m), Numpy 2D array with columns of ranges and depths
                    [2000.0, water_depth],
                ]
            ),
            rmax=1500.0,
            dr=dr,
            dz=dz,
            zmplt=380.0,
            c0=c0,
            np=npdefault,
        )
        res = pyram3.run()
        r = res.ranges
        z = res.depths
        i0 = np.argmin(np.abs(r - rh))
        tl.append(-res.loss_line[i0])
        print(r[0], r[-1], z[0], z[-1], dr, dz, i0)

    mu = 1

    r = np.hypot(rh, zr - zs)
    theta = np.abs(np.arctan2(zr, rh))
    f0 = c0 / (4 * zs * np.sin(theta))
    llm = -20 * np.log10(r) + 10 * np.log10(1 + mu**2 - 2 * mu * np.cos(np.pi * freqs / f0))
    plt.semilogx(freqs, tl, label="Pyram")
    plt.semilogx(freqs, -20 * np.log10(r) * np.ones(len(freqs)), label="20log(r)")
    plt.semilogx(freqs, llm, label="LM")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Tloss [dB re 1m]")
    plt.legend()
    plt.title(f"Water depth {water_depth} m")
    plt.show()


def example4():
    """
    Plot munk speed profile vs arctic speed profile
    """

    z = np.arange(2000)

    plt.plot(munk_profile(z), -z, label="Munk")
    plt.plot(arctic_profile(z), -z, label="Arctic")

    plt.xlabel("Sound speed [m/s]")
    plt.ylabel("Depth [m]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    example1()
    example2()
    example3()
    example4()
