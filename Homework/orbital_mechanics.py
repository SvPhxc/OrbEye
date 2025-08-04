import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import os

s_cfg = {
    'solver': 'DOP853', 'rtol': 1e-13, 'atol': 1e-13,
    'j_harm': True, 'tb': True,
    'drag': True, 'srp': True,
    'gmat_path': 'Satellite_PVT_GMAT.csv',
    'reentry_alt': 100.0
}

cnst = {
    'mu_e': 398600.4418, 'r_eq': 6378.137,
    'mu_s': 1.32712440018e11, 'mu_m': 4902.800076,
    'au': 149597870.7, 'ecl': np.radians(23.4392911),
    'p_sol': 4.56e-6
}

t_prms = {
    'j2': 1.08263e-3,
    'j3': -2.532669123e-6,
    'mass': 1000.0,
    'cd': 2.2,
    'a_d': 15.0,
    'cr': 1.8,
    'a_srp': 15.0
}


def get_atm_dns(alt):
    s_data = {100: (5.1e-7, 5.5), 150: (2.2e-9, 8.0), 200: (5.5e-11, 25.0),
              300: (2.5e-12, 45.0), 400: (8.0e-13, 60.0), 500: (2.5e-13, 75.0),
              600: (1.2e-13, 90.0), 800: (1.0e-14, 150.0), 1000: (3.0e-15, 200.0)}
    s_alts = sorted(s_data.keys())
    h0, (rho0, H) = s_alts[-1], s_data[s_alts[-1]]
    for ak in s_alts:
        if alt < ak: h0, (rho0, H) = ak, s_data[ak]; break
    if alt > 1000: return 0.0
    return rho0 * np.exp(-(alt - h0) / H)


def get_tb_pos(t, c):
    td = t / 86400.0
    eps = c['ecl']
    rot_m = np.array([[1, 0, 0], [0, np.cos(eps), -np.sin(eps)], [0, np.sin(eps), np.cos(eps)]])
    ma_s = np.radians(357.528 + 0.9856003 * td)
    r_s_ecl = np.array([c['au'] * np.cos(ma_s), c['au'] * np.sin(ma_s), 0])
    r_es = rot_m @ r_s_ecl
    ma_m = np.radians(297.85 + 13.176396 * td)
    d_m = 384400
    r_m_ecl = np.array([d_m * np.cos(ma_m), d_m * np.sin(ma_m), 0])
    r_em = rot_m @ r_m_ecl
    return r_es, r_em


def hf_mm(t, y, cfg, c, p):
    r, v = y[:3], y[3:]
    r_mag = np.linalg.norm(r)
    if r_mag == 0: return np.zeros(6)

    alt = r_mag - c['r_eq']
    if alt < cfg['reentry_alt']: return np.zeros(6)

    a_tot = -c['mu_e'] / r_mag ** 3 * r
    r_es, r_em = get_tb_pos(t, c)

    if cfg['j_harm']:
        x, y, z = r;
        r_eq = c['r_eq'];
        r_sq, z_sq = r_mag ** 2, z ** 2
        j2 = p['j2'];
        j3 = p['j3']
        j2f = -1.5 * j2 * c['mu_e'] * r_eq ** 2 / r_mag ** 5
        ax2 = j2f * x * (1 - 5 * z_sq / r_sq);
        ay2 = j2f * y * (1 - 5 * z_sq / r_sq);
        az2 = j2f * z * (3 - 5 * z_sq / r_sq)
        a_tot += np.array([ax2, ay2, az2])
        j3f = -2.5 * j3 * c['mu_e'] * r_eq ** 3 / r_mag ** 7
        ax3 = j3f * x * (3 * z - 7 * z ** 3 / r_sq);
        ay3 = j3f * y * (3 * z - 7 * z ** 3 / r_sq)
        az3 = j3f * (3 * z_sq - 0.6 * r_sq - 7 * z ** 4 / r_sq)
        a_tot += np.array([ax3, ay3, az3])

    if cfg['tb']:
        r_ss = r - r_es;
        r_sm = r - r_em
        a_sun = c['mu_s'] * ((r_ss / np.linalg.norm(r_ss) ** 3) - (r_es / np.linalg.norm(r_es) ** 3))
        a_moon = c['mu_m'] * ((r_sm / np.linalg.norm(r_sm) ** 3) - (r_em / np.linalg.norm(r_em) ** 3))
        a_tot += a_sun + a_moon

    if cfg['drag']:
        if alt < 1000:
            rho = get_atm_dns(alt);
            v_ms = v * 1000;
            v_mag_ms = np.linalg.norm(v_ms)
            if v_mag_ms > 0:
                force = -0.5 * rho * v_mag_ms ** 2 * p['cd'] * p['a_d'] * (v_ms / v_mag_ms)
                a_tot += (force / p['mass']) / 1000

    if cfg['srp']:
        r_s2s = r_es - r
        ratio = np.clip(c['r_eq'] / r_mag, -1.0, 1.0)
        th_l = np.arccos(ratio)
        th_s = np.arccos(np.dot(r, r_s2s) / (r_mag * np.linalg.norm(r_s2s)))
        if not (np.dot(r, r_s2s) > 0 and th_s < th_l):
            u_srp = r_s2s / np.linalg.norm(r_s2s)
            force = -c['p_sol'] * p['cr'] * p['a_srp'] * u_srp
            a_tot += (force / p['mass']) / 1000

    return np.concatenate((v, a_tot))


def read_gmat(fp):
    if not os.path.exists(fp): return None
    try:
        df = pd.read_csv(fp, skipinitialspace=True)
        remap = {df.columns[i]: ['time', 'x', 'y', 'z', 'vx', 'vy', 'vz'][i] for i in range(7)}
        df.rename(columns=remap, inplace=True);
        for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True);
        return df
    except Exception as e:
        return None


def sim_comp(y0, gdf, cfg, c, p):
    if gdf is None or gdf.empty: return

    t_sp = [0, gdf['time'].iloc[-1]];
    t_ev = gdf['time'].values
    print("Executing...")
    res = solve_ivp(
        hf_mm, t_sp, y0, method=cfg['solver'],
        t_eval=t_ev, rtol=cfg['rtol'], atol=cfg['atol'], args=(cfg, c, p)
    )
    print(f"Status: {res.message}")

    st, ss = res.t, res.y.T

    n_pts = len(st)
    if n_pts < len(gdf):
        print(f"Terminated early at t={st[-1] / 3600:.2f}h.")
        gdf = gdf.iloc[:n_pts]

    gp, gv = gdf[['x', 'y', 'z']].values, gdf[['vx', 'vy', 'vz']].values
    e_vec = ss[:, :3] - gp

    e_ric = np.zeros_like(gp)
    for i in range(len(gp)):
        r, v = gp[i], gv[i];
        r_mag = np.linalg.norm(r)
        if r_mag == 0: continue
        c_mag = np.linalg.norm(np.cross(r, v))
        if c_mag == 0: continue
        R_hat = r / r_mag;
        C_hat = np.cross(r, v) / c_mag;
        I_hat = np.cross(C_hat, R_hat)
        e_ric[i] = [np.dot(e_vec[i], R_hat), np.dot(e_vec[i], I_hat), np.dot(e_vec[i], C_hat)]

    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111, projection='3d')

    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    x_e = c['r_eq'] * np.cos(u) * np.sin(v)
    y_e = c['r_eq'] * np.sin(u) * np.sin(v)
    z_e = c['r_eq'] * np.cos(v)
    ax1.plot_surface(x_e, y_e, z_e, color="blue", alpha=0.2)

    ax1.plot(ss[:, 0], ss[:, 1], ss[:, 2], label='Simulation', color='cyan', linewidth=2)
    ax1.plot(gdf['x'], gdf['y'], gdf['z'], label='GMAT', color='red', linestyle='--', linewidth=2)

    max_range = np.max(np.abs(ss[:, :3])) * 1.1
    ax1.set_xlim([-max_range, max_range]);
    ax1.set_ylim([-max_range, max_range]);
    ax1.set_zlim([-max_range, max_range])
    ax1.set_title("3D Orbit Comparison")
    ax1.set_xlabel("X (km)");
    ax1.set_ylabel("Y (km)");
    ax1.set_zlabel("Z (km)")
    ax1.legend()

    fig2, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    t_hr = st / 3600
    axs[0].plot(t_hr, e_ric[:, 0] * 1000);
    axs[0].set_ylabel("Radial Error (m)");
    axs[0].grid(True)
    axs[1].plot(t_hr, e_ric[:, 1] * 1000);
    axs[1].set_ylabel("In-Track Error (m)");
    axs[1].grid(True)
    axs[2].plot(t_hr, e_ric[:, 2] * 1000);
    axs[2].set_ylabel("Cross-Track Error (m)");
    axs[2].grid(True)
    axs[2].set_xlabel("Time (hours)");
    fig2.suptitle("Propagator Error Components (Simulation vs. GMAT)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
    plt.show()


if __name__ == '__main__':
    y0 = np.array([7000.0, 0.000001, -0.001608, 0.000002, 1.310359, 7.431412])
    gmat_data = read_gmat(s_cfg['gmat_path'])
    sim_comp(y0, gmat_data, s_cfg, cnst, t_prms)
