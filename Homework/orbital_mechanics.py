import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_truth(fpath):
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath, skipinitialspace=True)
        col_map = {df.columns[i]: name for i, name in enumerate(['t', 'x', 'y', 'z', 'vx', 'vy', 'vz'])}
        df.rename(columns=col_map, inplace=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

def get_keplerian(state, mu, r_e):
    r_vec, v_vec = state[:3], state[3:]
    r_norm = np.linalg.norm(r_vec)

    alt = r_norm - r_e

    h_vec = np.cross(r_vec, v_vec)
    inc = np.degrees(np.arccos(h_vec[2] / np.linalg.norm(h_vec)))

    e_vec = ((np.linalg.norm(v_vec)**2 - mu / r_norm) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e_norm = np.linalg.norm(e_vec)

    if e_norm < 1e-6:
        n_vec = np.cross([0, 0, 1], h_vec)
        arg_lat = np.arccos(np.dot(n_vec, r_vec) / (np.linalg.norm(n_vec) * r_norm))
        if r_vec[2] < 0:
            arg_lat = 2 * np.pi - arg_lat
        return alt, inc, arg_lat
    else:
        ta = np.arccos(np.dot(e_vec, r_vec) / (e_norm * r_norm))
        if np.dot(r_vec, v_vec) < 0:
            ta = 2 * np.pi - ta
        return alt, inc, ta

def ode2b(t, y, mu):
    r, v = y[:3], y[3:]
    accel = -mu / np.linalg.norm(r)**3 * r
    return np.concatenate((v, accel))

def rk4(f, t, y, dt, mu):
    k1 = dt * f(t, y, mu)
    k2 = dt * f(t + 0.5 * dt, y + 0.5 * k1, mu)
    k3 = dt * f(t + 0.5 * dt, y + 0.5 * k2, mu)
    k4 = dt * f(t + dt, y + k3, mu)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

def run_sim(y0, truth_df):
    if truth_df is None or truth_df.empty:
        return

    mu = 398600.4418
    r_e = 6378.137
    dt = 60.0
    t_end = truth_df['t'].iloc[-1]

    ts = np.arange(0, t_end, dt)
    states = np.zeros((len(ts), 6))
    states[0, :] = y0

    for i in range(len(ts) - 1):
        states[i+1, :] = rk4(ode2b, ts[i], states[i], dt, mu)

    alt_err, inc_err, track_err = [], [], []

    for i, t_now in enumerate(ts):
        sim_state = states[i]
        truth_state = np.array([np.interp(t_now, truth_df['t'], truth_df[col]) for col in ['x', 'y', 'z', 'vx', 'vy', 'vz']])

        sim_alt, sim_inc, sim_ta = get_keplerian(sim_state, mu, r_e)
        truth_alt, truth_inc, truth_ta = get_keplerian(truth_state, mu, r_e)

        alt_err.append(sim_alt - truth_alt)
        inc_err.append(sim_inc - truth_inc)

        d_ta = sim_ta - truth_ta
        if d_ta > np.pi: d_ta -= 2*np.pi
        if d_ta < -np.pi: d_ta += 2*np.pi

        track_err.append(d_ta * np.linalg.norm(truth_state[:3]))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    time_hr = ts / 3600

    ax1.plot(time_hr, alt_err)
    ax1.set_ylabel("Altitude Error (km)")
    ax1.set_title("Orbital Element Errors (Simple Model vs. GMAT)")
    ax1.grid(True)

    ax2.plot(time_hr, track_err)
    ax2.set_ylabel("In-Track Error (km)")
    ax2.grid(True)

    ax3.plot(time_hr, inc_err)
    ax3.set_ylabel("Inclination Error (deg)")
    ax3.set_xlabel("Time (hours)")
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

if __name__ == '__main__':
    y0 = np.array([7000., 1e-6, -0.001608, 2e-6, 1.310359, 7.431412])
    gmat_file = 'Satellite_PVT_GMAT.csv'

    gmat_dat = load_truth(gmat_file)
    run_sim(y0, gmat_dat)
