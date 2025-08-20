from __future__ import annotations
from datetime import timedelta, timezone, datetime
import os
import requests
import numpy as np
import math

# SGP4
from sgp4.api import Satrec, jday
from sgp4.conveniences import sat_epoch_datetime

# ---------------------------
# Public API
# ---------------------------

def parse_tle_file(file_path: str) -> list[tuple[str, str, str]]:
    """
    Read a .tle file and return a list of (name, line1, line2) triples.
    The file is expected in 3-line blocks: name, L1, L2.
    """
    triples: list[tuple[str, str, str]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # find 3-line groups that look like TLEs
    i = 0
    while i + 2 < len(lines):
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if _looks_like_tle(l1, l2):
            triples.append((name, l1, l2))
            i += 3
        else:
            # if not a proper block, advance by 1 and keep scanning
            i += 1
    return triples


def fetch_tle_by_name(satellite_name: str, timeout: float = 10.0) -> tuple[str, str, str]:
    """
    Fetch (name, line1, line2) for a satellite from Celestrak's ACTIVE set.
    Name match is case-insensitive and substring-based.
    Raises ValueError if not found.
    """
    if not satellite_name or not satellite_name.strip():
        raise ValueError("Satellite name must be a non-empty string.")

    # common typo fix
    if satellite_name.upper() in {"ISS (ZAYRA)", "ISS(ZAYRA)"}:
        satellite_name = "ISS (ZARYA)"

    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip()]
    except Exception as e:
        raise RuntimeError(f"Failed to fetch TLEs from Celestrak: {e}") from e

    target = satellite_name.strip().lower()
    for i in range(0, len(lines) - 2, 3):
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if target in name.lower() and _looks_like_tle(l1, l2):
            return (name, l1, l2)

    raise ValueError(f"Satellite '{satellite_name}' not found.")


def normalize_tle_input(query: str | None,
                        default_path: str = "example.tle") -> tuple[str, str, str]:
    """
    Accepts:
      - a satellite name (e.g., "ISS (ZARYA)")
      - "TLE" or empty/None -> load the first TLE from default_path
      - a .tle file path (endswith .tle or existing file)
    Returns (name, line1, line2).
    """
    q = (query or "").strip()

    # Shortcut to local file: empty, "TLE", or explicit file
    is_file = False
    if not q or q.upper() == "TLE":
        q = default_path
        is_file = True
    elif q.lower().endswith(".tle"):
        is_file = True
    elif os.path.isfile(q):
        is_file = True

    if is_file:
        triples = parse_tle_file(q)
        if not triples:
            raise ValueError(f"No valid TLE triples found in '{q}'.")
        return triples[0]  # first triple
    else:
        # treat as a satellite name
        return fetch_tle_by_name(q)


def save_tle_to_example(name: str, line1: str, line2: str, path: str = "example.tle") -> None:
    """
    Overwrite/Save a TLE triple to a local file for quick plotting via 'TLE'.
    """
    if not (name and line1 and line2):
        raise ValueError("All of name, line1, and line2 must be provided.")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{name.rstrip()}\n{line1.rstrip()}\n{line2.rstrip()}\n")


def generate_orbit_xyz(*,
                       tle_lines: tuple[str, str] | tuple[str, str, str] | None = None,
                       tle_filename: str | None = None,
                       duration_minutes: int = 90,
                       step_seconds: int = 60,
                       start_time_utc = None) -> np.ndarray:
    """
    Propagate a TLE with SGP4 and return Nx3 TEME/ECI positions in **kilometers**.
    - Provide either `tle_lines=(L1,L2)` or `(name,L1,L2)`, or `tle_filename="file.tle"`.
    - No coordinate transforms; output is geocentric (Earth center), perfect for your GL view.
    """
    line1, line2 = _resolve_tle_lines(tle_lines=tle_lines, tle_filename=tle_filename)

    # Build SGP4 satellite record
    sat = Satrec.twoline2rv(line1, line2)
    t0 = start_time_utc or sat_epoch_datetime(sat)

    total_secs = max(int(duration_minutes) * 60, step_seconds * 2)
    n_steps = max(int(total_secs // max(1, step_seconds)) + 1, 2)

    xyz_km = np.empty((n_steps, 3), dtype=float)
    idx = 0

    for i in range(n_steps):
        t = t0 + timedelta(seconds=i * step_seconds)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        err, r_km, _ = sat.sgp4(jd, fr)
        if err == 0 and r_km is not None:
            xyz_km[idx] = np.asarray(r_km, dtype=float)
        else:
            # If propagation fails at this step, repeat last valid point or use NaNs.
            xyz_km[idx] = xyz_km[idx - 1] if idx > 0 else np.array([np.nan, np.nan, np.nan], float)
        idx += 1

    return xyz_km


def get_orbit_xyz_for_query(query: str | None,
                            duration_minutes: int = 90,
                            step_seconds: int = 60,
                            default_path: str = "example.tle"):
    """
    Convenience wrapper:
      Input: satellite name, 'TLE'/empty for example.tle, or a .tle filename.
      Output: (name, Nx3 TEME/ECI positions in km)
    """
    name, l1, l2 = normalize_tle_input(query, default_path=default_path)
    pts = generate_orbit_xyz(tle_lines=(l1, l2), duration_minutes=duration_minutes, step_seconds=step_seconds)
    return name, pts

# ---------------------------
# Internal helpers
# ---------------------------

def _looks_like_tle(l1: str, l2: str) -> bool:
    return l1.startswith("1 ") and l2.startswith("2 ") and len(l1) >= 60 and len(l2) >= 60


def _resolve_tle_lines(*, tle_lines, tle_filename) -> tuple[str, str]:
    """
    Normalize any accepted TLE input to (line1, line2).
    """
    if tle_lines:
        if isinstance(tle_lines, (list, tuple)):
            if len(tle_lines) == 2:
                return str(tle_lines[0]), str(tle_lines[1])
            if len(tle_lines) == 3:
                # (name, l1, l2)
                return str(tle_lines[1]), str(tle_lines[2])
    if tle_filename:
        triples = parse_tle_file(tle_filename)
        if not triples:
            raise ValueError(f"No valid TLE triples found in '{tle_filename}'.")
        # First triple
        _, l1, l2 = triples[0]
        return l1, l2
    raise ValueError("Provide either tle_lines=(l1,l2) or tle_filename='file.tle'.")












MU_EARTH_KM3_S2 = 398600.4418  # Earth's GM (km^3/s^2)



# =========================
#  TLE fitting from samples
# =========================
def fit_tle_from_satellite_points(
    sat_points,
    *,
    unit: str = "cm",
    name: str = "FITTED-SAT",
    epoch_hint: datetime | None = None
):
    """
    Fit a rough TLE from a sequence of satellite_points samples.

    Input
    -----
    sat_points : array-like (N x 5 or N x 4)
        Rows: [az_deg, el_deg, dist_cm_or_unit, (strength optional), timestamp_seconds]
        - az, el in degrees (your GUI convention)
        - dist in `unit` (default "cm")
        - timestamp in seconds (preferably Unix epoch seconds; otherwise relative seconds)
    unit : {"cm","m","km"}  distance unit for the 3rd column
    name : str               TLE name to embed
    epoch_hint : datetime    if given, use as the TLE epoch; otherwise inferred from timestamps

    Returns
    -------
    (name, line1, line2) : tuple[str, str, str]
        Two TLE lines ready to use with sgp4.

    Notes
    -----
    - Assumes geocentric observer (as per your project simplification).
    - Uses central-difference velocity at the median time sample.
    - Produces *approximate* SGP4-compatible elements; good for visualization.
    """
    pts = _np_as_array(sat_points)
    if pts.shape[1] < 4:
        raise ValueError("sat_points must have columns [az_deg, el_deg, dist, (strength?), ts].")

    # Columns
    az_deg = pts[:, 0].astype(float)
    el_deg = pts[:, 1].astype(float)
    dist   = pts[:, 2].astype(float)
    ts     = pts[:, -1].astype(float)  # last col = timestamp

    if len(pts) < 3:
        raise ValueError("Need at least 3 samples to estimate velocity.")

    # Convert distance to kilometers
    if unit == "cm":
        dist_km = dist / 1e5
    elif unit == "m":
        dist_km = dist / 1e3
    elif unit == "km":
        dist_km = dist.copy()
    else:
        raise ValueError(f"Unsupported unit '{unit}'. Use 'cm', 'm', or 'km'.")

    # Build position vectors in km, GUI convention:
    # x = r * cos(el) * cos(az)
    # y = r * cos(el) * sin(az)
    # z = r * sin(el)
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    cos_el = np.cos(el)
    x = dist_km * cos_el * np.cos(az)
    y = dist_km * cos_el * np.sin(az)
    z = dist_km * np.sin(el)
    r_km = np.column_stack((x, y, z))

    # Time array (s) and epoch
    t0 = _infer_epoch_from_ts(ts, epoch_hint)
    t_sec = ts - ts[0] if _looks_like_relative_time(ts) else (ts - ts[0])  # centered at first sample
    # Better: use actual deltas for gradient
    dt = np.gradient(ts)

    # Velocity via central differences (km/s) – robust over nonuniform ts
    v_km_s = np.empty_like(r_km)
    for j in range(3):
        v_km_s[:, j] = np.gradient(r_km[:, j], ts)

    # Choose the mid sample for element computation
    mid = len(pts) // 2
    r0 = r_km[mid]
    v0 = v_km_s[mid]

    # Convert (r,v) -> classical orbital elements
    coe = _rv_to_coe(r0, v0, MU_EARTH_KM3_S2)
    # Unpack
    a_km = coe["a_km"]
    e    = coe["e"]
    i_deg = np.degrees(coe["i"])
    raan_deg = _wrap_deg(np.degrees(coe["raan"]))
    argp_deg = _wrap_deg(np.degrees(coe["argp"]))
    M_deg    = _wrap_deg(np.degrees(coe["M"]))

    # Mean motion (rev/day)
    if not np.isfinite(a_km) or a_km <= 0.0:
        raise ValueError("Fitted semi-major axis not positive; cannot form TLE.")
    n_rad_s = math.sqrt(MU_EARTH_KM3_S2 / (a_km ** 3))
    n_rev_day = n_rad_s / (2.0 * math.pi) * 86400.0

    # TLE lines
    epoch_dt = epoch_hint or _epoch_from_ts_for_tle(ts, t0)
    line1, line2 = _build_tle_lines(
        name=name,
        epoch=epoch_dt,
        inc_deg=i_deg,
        raan_deg=raan_deg,
        ecc=e,
        argp_deg=argp_deg,
        M_deg=M_deg,
        n_rev_day=n_rev_day
    )
    return name, line1, line2


# ------------------------
# Helpers for TLE fitting
# ------------------------

def _np_as_array(x):
    return x if isinstance(x, np.ndarray) else np.asarray(x, dtype=float)

def _looks_like_relative_time(ts):
    # crude: Unix times are ~1e9..1.9e9 right now; relative times usually < 1e7
    return float(np.max(ts)) < 1e8

def _infer_epoch_from_ts(ts, epoch_hint):
    if epoch_hint:
        return epoch_hint.replace(tzinfo=timezone.utc)
    if not _looks_like_relative_time(ts):
        # treat as Unix epoch seconds; use first timestamp as epoch
        return datetime.fromtimestamp(float(ts[0]), tz=timezone.utc)
    # fallback: now in UTC, minus first timestamp (assumed relative)
    return datetime.now(tz=timezone.utc)

def _epoch_from_ts_for_tle(ts, epoch0):
    """
    Choose an epoch near the *middle* of the dataset to reduce mean anomaly bias.
    If ts are absolute, use mid timestamp; else use epoch0 + mid offset.
    """
    mid_idx = len(ts) // 2
    if _looks_like_relative_time(ts):
        return epoch0 + (ts[mid_idx] - ts[0]) * _SEC
    return datetime.fromtimestamp(float(ts[mid_idx]), tz=timezone.utc)

_SEC = timedelta(seconds=1)


def _rv_to_coe(r, v, mu):
    """
    r [km], v [km/s] -> dict of classical orbital elements in radians & km.
    """
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rn = np.linalg.norm(r)
    vn = np.linalg.norm(v)

    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    k = np.array([0.0, 0.0, 1.0])
    n = np.cross(k, h)
    nn = np.linalg.norm(n)

    e_vec = (np.cross(v, h) / mu) - (r / rn)
    e = np.linalg.norm(e_vec)

    # Specific mechanical energy
    eps = vn*vn/2.0 - mu / rn
    a = -mu / (2.0 * eps) if abs(eps) > 0 else np.inf

    # Angles
    i = math.acos(np.clip(h[2] / hn, -1.0, 1.0))

    raan = 0.0
    if nn > 1e-12:
        raan = math.atan2(n[1], n[0])  # 0..2pi

    # Argument of perigee
    if e > 1e-10 and nn > 1e-12:
        # Use atan2 with quadrant via cross products
        cos_omega = np.dot(n, e_vec) / (nn * e)
        sin_omega = np.dot(np.cross(n, e_vec), h) / (nn * e * hn)
        argp = math.atan2(sin_omega, np.clip(cos_omega, -1.0, 1.0))
    else:
        argp = 0.0

    # True anomaly (or argument of latitude for circular)
    if e > 1e-10:
        cos_nu = np.dot(e_vec, r) / (e * rn)
        sin_nu = np.dot(np.cross(e_vec, r), h) / (e * rn * hn)
        nu = math.atan2(sin_nu, np.clip(cos_nu, -1.0, 1.0))
        # Eccentric anomaly -> Mean anomaly
        E = math.atan2(math.sin(nu) * math.sqrt(1 - e*e), e + math.cos(nu))
        M = E - e * math.sin(E)
    else:
        # circular: use argument of latitude u, set ω=0, M ~ u
        if nn > 1e-12:
            cos_u = np.dot(n, r) / (nn * rn)
            sin_u = np.dot(np.cross(n, r), h) / (nn * rn * hn)
            u = math.atan2(sin_u, np.clip(cos_u, -1.0, 1.0))
        else:
            # equatorial circular fallback
            u = math.atan2(r[1], r[0])
        argp = 0.0
        M = u

    # Normalize angles to [0, 2π)
    raan = _wrap_rad(raan)
    argp = _wrap_rad(argp)
    M    = _wrap_rad(M)

    return {
        "a_km": a,
        "e": e,
        "i": i,
        "raan": raan,
        "argp": argp,
        "M": M,
    }

def _wrap_rad(x):
    return (x + 2.0 * math.pi) % (2.0 * math.pi)

def _wrap_deg(x):
    x = x % 360.0
    return x + 360.0 if x < 0.0 else x


# ------------------------
# TLE building & checksum
# ------------------------
def _build_tle_lines(
    *,
    name: str,
    epoch: datetime,
    inc_deg: float,
    raan_deg: float,
    ecc: float,
    argp_deg: float,
    M_deg: float,
    n_rev_day: float,
    norad_id: int = 99999,
    set_number: int = 1,
    bstar: float = 0.0,
    ndot: float = 0.0,
    nddot: float = 0.0,
    rev_number: int = 0,
):
    # Epoch YYDDD.DDDDDDDD
    epoch = epoch.astimezone(timezone.utc)
    yy = epoch.year % 100
    doy = (epoch - datetime(epoch.year, 1, 1, tzinfo=timezone.utc)).total_seconds() / 86400.0 + 1.0
    epoch_field = f"{yy:02d}{doy:012.8f}"

    # Eccentricity (7 digits, no decimal)
    ecc_field = f"{int(round(abs(ecc) * 1e7)):07d}"

    # B*, nddot in TLE "exponent" format
    bstar_field  = _tle_exp_field(bstar)
    nddot_field  = _tle_exp_field(nddot)

    # First derivative of mean motion in rev/day^2 (decimal fixed field, signed)
    ndot_field = f"{ndot: .8f}".replace("+", " ")

    # Mean motion field (11 chars, 8 decimals)
    n_field = f"{n_rev_day:11.8f}"

    # Angles fields (8 chars, 4 decimals)
    inc_field  = f"{inc_deg:8.4f}"
    raan_field = f"{raan_deg:8.4f}"
    argp_field = f"{argp_deg:8.4f}"
    M_field    = f"{M_deg:8.4f}"

    # Build line 1 & 2 without checksums
    line1 = (
        f"1 {norad_id:05d}U 00000A   {epoch_field} {ndot_field} {nddot_field} {bstar_field} 0 {set_number:4d}"
    )
    line2 = (
        f"2 {norad_id:05d} {inc_field} {raan_field} {ecc_field} {argp_field} {M_field} {n_field} {rev_number:05d}"
    )

    # Checksums
    line1 = line1 + str(_tle_checksum(line1))
    line2 = line2 + str(_tle_checksum(line2))

    # Prepend the name as the 0th line (optional; many parsers accept name separately)
    # Return only 2-line per your request
    return line1, line2

def _tle_exp_field(val: float) -> str:
    """
    Convert float to TLE's mantissa/exponent field (8 chars):
      " 00000-0" for zero,
      or "±ddddd±e" where the decimal point is omitted and mantissa is 0.ddddd.
    """
    if val == 0.0:
        return " 00000-0"

    sgn = "-" if val < 0 else " "
    v = abs(val)
    exp10 = int(math.floor(math.log10(v)))
    # Make mantissa 0.ddddd by shifting one decade
    mant = v / (10 ** (exp10 + 1)) * 10.0 if exp10 != -999 else 0.0
    # Equivalent simpler form:
    mant = v / (10 ** (exp10 + 1)) * 10.0  # -> between 0.1 and 1.0
    # Round to 5 digits
    digits = int(round(mant * 1e5))
    if digits == 100000:
        digits = 99999  # clamp

    # TLE exponent is exp10+1 (because of the mantissa shift)
    e = exp10 + 1
    es = "+" if e >= 0 else "-"
    return f"{sgn}{digits:05d}{es}{abs(e):02d}"

def _tle_checksum(line: str) -> int:
    """
    TLE checksum: sum of all digits + count of '-' characters, modulo 10.
    """
    s = 0
    for ch in line[:68]:  # checksum excludes the checksum itself
        if ch.isdigit():
            s += int(ch)
        elif ch == "-":
            s += 1
    return s % 10



def get_acquisition_pan_deg(*, tle_lines=None, tle_filename=None) -> float:
    """
    Acquisition pan ≈ RAAN (deg). Assumes Earth-center observer & GUI azimuth:
    +X at 0°, +Y at 90°, increasing CCW in the XY plane.
    """
    line1, line2 = _resolve_tle_lines(tle_lines=tle_lines, tle_filename=tle_filename)
    sat = Satrec.twoline2rv(line1, line2)
    raan_deg = (math.degrees(sat.nodeo)) % 360.0
    return raan_deg

def get_ascending_node_unit_vector(*, tle_lines=None, tle_filename=None) -> np.ndarray:
    """
    Unit vector in ECI/TEME toward the ascending node (lies in the equatorial plane).
    Useful to draw a line in your 3D view.
    """
    pan_deg = get_acquisition_pan_deg(tle_lines=tle_lines, tle_filename=tle_filename)
    az = math.radians(pan_deg)
    return np.array([math.cos(az), math.sin(az), 0.0], dtype=float)