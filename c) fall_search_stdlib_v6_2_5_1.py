"""
fall_search_stdlib_v6.py

Monte‑Carlo landing dispersion for a falling/tumbling object with wind and gusts.
Outputs CSV of landings and JSON/GeoJSON/KML search areas + distance-first radii + spiral path. Stdlib‑only.

Coordinate frames & conventions:
- Position frame: x=east (m), y=north (m), z=up (m).
- Angles (code frame): 0° = east, counter‑clockwise positive.
- Wind direction inputs: MET FROM (degrees).
- Heading inputs (pre‑release): MET TO (degrees).
- Units: meters, seconds, kilograms. Winds in km/h externally; converted to m/s internally.
"""
# [PATCH-SMOKE] unified-diff test marker (safe to keep or remove)

import math, random, csv, json, time, os, unicodedata
from datetime import datetime

EARTH_METERS_PER_DEG_LAT = 111320.0

# ---- JSON configuration loader (compatible with v5 schema) ----------------------------

def _deep_merge(base: dict, overrides: dict) -> dict:
    out = json.loads(json.dumps(base))  # deep copy via JSON
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config_json(path: str, base: dict = None, *, strict: bool = False) -> dict:
    if base is None:
        base = CONFIG
    with open(path, "r", encoding="utf-8") as f:
        external = json.load(f)
    merged = _deep_merge(base, external)
    errs = validate_config(merged)
    if errs:
        msg = "CONFIG validation issues:\n" + "\n".join(f" - {e}" for e in errs)
        if strict:
            raise ValueError(msg)
        else:
            print(msg)
    return merged

# ---- Geodesy helpers -------------------------------------------------------

def meters_to_latlon(lat0, lon0, east_m, north_m):
    lat = lat0 + north_m / EARTH_METERS_PER_DEG_LAT
    lon = lon0 + east_m / (EARTH_METERS_PER_DEG_LAT * math.cos(math.radians(lat0)))
    return lat, lon

def met_from_to_code_toward_rad(met_from_deg: float) -> float:
    return math.radians((90.0 - float(met_from_deg)) % 360.0)

def met_toward_to_code_toward_rad(met_toward_deg: float) -> float:
    return math.radians((90.0 - float(met_toward_deg)) % 360.0)

# ---- Physics models --------------------------------------------------------

def powerlaw_speed_at_height(vref_ms, z, zref=10.0, alpha=0.143):
    z_eff = max(0.1, float(z))
    return vref_ms * (z_eff / zref) ** alpha

def air_density_exponential(z, rho0=1.225, H=8400.0):
    if z < 0: z = 0.0
    return rho0 * math.exp(-z / H)

def ou_gust_step(x_prev, dt, sigma, tau):
    if sigma <= 0.0:
        return 0.0 if not math.isfinite(x_prev) else x_prev * max(0.0, 1.0 - dt / max(1e-6, tau))
    theta = 1.0 / max(1e-6, tau)
    sigma_w = sigma * math.sqrt(2.0 * theta)
    dw = random.gauss(0.0, math.sqrt(max(1e-9, dt)))
    return x_prev + theta * (0.0 - x_prev) * dt + sigma_w * dw

def drag_horizontal_accel(v_rel_x, v_rel_y, rho, Cd, A, mass):
    vrel_h = math.hypot(v_rel_x, v_rel_y)
    if vrel_h == 0.0:
        return 0.0, 0.0
    F_scalar = -0.5 * rho * Cd * A * vrel_h
    Fx = F_scalar * v_rel_x
    Fy = F_scalar * v_rel_y
    return Fx / mass, Fy / mass

# ---- Integrator ------------------------------------------------------------

def simulate_one_2d(run_params):
    m = run_params['mass']
    A_base = run_params['A_base']
    Cd_base = run_params['Cd_base']
    A_frac = run_params.get('A_tumble_frac', 0.0)
    Cd_frac = run_params.get('Cd_tumble_frac', 0.0)
    omega = run_params.get('tumble_omega', 0.0)
    phase = run_params.get('tumble_phase', 0.0)

    vref_ms = run_params['vref_ms']
    wind_dir_met_deg = run_params['wind_dir_met_deg']
    alpha = run_params.get('wind_alpha', 0.143)
    gust_sigma = run_params.get('gust_sigma', 0.0)
    gust_tau = run_params.get('gust_tau', 5.0)
    air_vz_mean = run_params.get('w_mean', 0.0)

    x = run_params.get('x0', 0.0)
    y = run_params.get('y0', 0.0)
    z = run_params['release_alt_m']
    vx = run_params.get('initial_vx', 0.0)
    vy = run_params.get('initial_vy', 0.0)
    vz = run_params.get('initial_vz', 0.0)

    dt_slow = run_params.get('dt_slow', 0.02)
    dt_fast = run_params.get('dt_fast', 0.005)
    v_switch = run_params.get('v_switch', 30.0)
    z_switch = run_params.get('z_switch', 100.0)
    max_time = run_params.get('max_time', 4000.0)
    g_acc = 9.80665
    base_theta = met_from_to_code_toward_rad(wind_dir_met_deg)

    gust_u = 0.0
    gust_v = 0.0
    t = 0.0
    last_state = None
    while t < max_time:
        if z <= 0.0:
            return {'x_e': x, 'y_n': y, 't_land': t, 'landed': True, 'interp': False}

        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        dt = dt_fast if (speed > v_switch or z > z_switch) else dt_slow

        if omega and (A_frac != 0.0 or Cd_frac != 0.0):
            ang = omega * t + phase
            A_eff = max(1e-6, A_base * (1.0 + A_frac * math.sin(ang)))
            Cd_eff = max(0.01, Cd_base * (1.0 + Cd_frac * math.cos(ang)))
        else:
            A_eff, Cd_eff = A_base, Cd_base

        v_profile = powerlaw_speed_at_height(vref_ms, z, zref=10.0, alpha=alpha)
        wind_e = v_profile * math.cos(base_theta)
        wind_n = v_profile * math.sin(base_theta)
        if gust_sigma > 0.0:
            gust_u = ou_gust_step(gust_u, dt, gust_sigma, gust_tau)
            gust_v = ou_gust_step(gust_v, dt, gust_sigma, gust_tau)
        air_e = wind_e + gust_u
        air_n = wind_n + gust_v
        air_vz = air_vz_mean

        vrel_e = vx - air_e
        vrel_n = vy - air_n
        vrel_v = vz - air_vz

        rho_local = air_density_exponential(z)

        ax_d, ay_d = drag_horizontal_accel(vrel_e, vrel_n, rho_local, Cd_eff, A_eff, m)
        vrel_mag = math.sqrt(vrel_e * vrel_e + vrel_n * vrel_n + vrel_v * vrel_v)
        az_d = 0.0 if vrel_mag == 0.0 else (-0.5 * rho_local * Cd_eff * A_eff * vrel_mag * vrel_v) / m

        ax = ax_d
        ay = ay_d
        az = az_d - g_acc

        last_state = (t, x, y, z)

        vx += ax * dt
        vy += ay * dt
        vz += az * dt
        x += vx * dt
        y += vy * dt
        z += vz * dt
        t += dt

        if last_state and z <= 0.0:
            t0, x0, y0, z0 = last_state
            denom = (z0 - z)
            s = 0.0 if denom == 0.0 else max(0.0, min(1.0, z0 / denom))
            x_land = x0 + s * (x - x0)
            y_land = y0 + s * (y - y0)
            t_land = t0 + s * (t - t0)
            return {'x_e': x_land, 'y_n': y_land, 't_land': t_land, 'landed': True, 'interp': True}
    return {'x_e': x, 'y_n': y, 't_land': t, 'landed': False, 'interp': False}

# ---- Monte Carlo driver ----------------------------------------------------

def run_monte_carlo(N, base_inputs, sample_specs):
    results = []
    t0 = time.time()
    seed = base_inputs.get('seed', int(time.time() * 1000) % 2_000_000_000)
    random.seed(seed)

    A_mean = base_inputs['A_mean']
    Cd_mean = base_inputs['Cd_mean']
    sA = sample_specs.get('A_std', 0.0)
    sCd = sample_specs.get('Cd_std', 0.0)
    rhoACd = sample_specs.get('rho_A_Cd', 0.0)

    cell_size_m = sample_specs.get('cell_size_m', 50.0)
    containment_probs = sample_specs.get('containment_probs', (0.5, 0.9, 0.95))
    use_kde = sample_specs.get('use_kde_for_mode', True)
    kde_bw_m = sample_specs.get('kde_bandwidth_m', 75.0)
    dt_slow = sample_specs.get('dt_slow', 0.02)
    dt_fast = sample_specs.get('dt_fast', 0.005)
    v_switch = sample_specs.get('v_switch', 30.0)
    z_switch = sample_specs.get('z_switch', 100.0)
    max_time = sample_specs.get('max_time', 4000.0)
    dir_conv = sample_specs.get('wind_dir_convention', 'met_from')

    for i in range(N):
        if (i + 1) % max(1, N // 10) == 0 or (i + 1) == N:
            print("Progress: {}/{} runs".format(i + 1, N))

        z1 = random.gauss(0.0, 1.0)
        z2 = random.gauss(0.0, 1.0)
        A_s = max(1e-6, A_mean + sA * z1)
        Cd_s = max(0.01, Cd_mean + rhoACd * sCd * z1 + math.sqrt(max(0.0, 1.0 - rhoACd * rhoACd)) * sCd * z2)

        v_kph = max(0.0, random.gauss(sample_specs.get('wind_speed_kph_mean', 10.0),
                                      sample_specs.get('wind_speed_kph_std', 2.0)))
        v_ms = v_kph * (1000.0 / 3600.0)

        if dir_conv == 'met_from':
            wind_dir_met_deg = (random.gauss(sample_specs.get('wind_dir_met_from_mean', 90.0),
                                             sample_specs.get('wind_dir_met_from_std', 20.0))) % 360.0
        elif dir_conv == 'code_from':
            wind_dir_code_from = (random.gauss(sample_specs.get('wind_dir_deg_mean', 90.0),
                                               sample_specs.get('wind_dir_deg_std', 20.0))) % 360.0
            code_toward = (wind_dir_code_from + 180.0) % 360.0
            wind_dir_met_deg = (90.0 - code_toward) % 360.0
        else:
            raise ValueError("wind_dir_convention must be 'met_from' or 'code_from'")
        wind_alpha = sample_specs.get('wind_alpha', 0.143)
        gust_sigma = sample_specs.get('gust_sigma', 0.0)
        gust_tau = sample_specs.get('gust_tau', 5.0)

        omega = random.gauss(sample_specs.get('tumble_omega_mean', 0.0),
                             sample_specs.get('tumble_omega_std', 0.0))
        phase = random.random() * 2.0 * math.pi

        x0 = y0 = 0.0
        vx0 = vy0 = 0.0
        pre = sample_specs.get('pre_release', {'enabled': False})
        if pre.get('enabled', False):
            hconv = pre.get('heading_convention', 'met_toward')
            if hconv == 'met_toward':
                heading_met = (random.gauss(pre.get('heading_met_toward_mean', 0.0),
                                            pre.get('heading_met_toward_std', 0.0))) % 360.0
                theta = met_toward_to_code_toward_rad(heading_met)
            elif hconv == 'code_toward':
                heading_code_toward = (random.gauss(pre.get('heading_deg_mean', 0.0),
                                                    pre.get('heading_deg_std', 0.0))) % 360.0
                theta = math.radians(heading_code_toward)
            else:
                raise ValueError("pre_release.heading_convention must be 'met_toward' or 'code_toward'")
            mode = pre.get('mode', 'time')
            if mode == 'time':
                t_move = max(0.0, random.gauss(pre.get('time_s_mean', 0.0), pre.get('time_s_std', 0.0)))
                v_gs_ms = max(0.0, random.gauss(pre.get('groundspeed_kph_mean', 0.0),
                                                pre.get('groundspeed_kph_std', 0.0))) * (1000.0 / 3600.0)
                dist = v_gs_ms * t_move
                x0 = dist * math.cos(theta)
                y0 = dist * math.sin(theta)
                vx0 = v_gs_ms * math.cos(theta)
                vy0 = v_gs_ms * math.sin(theta)
            elif mode == 'glide_ratio':
                glide_ratio = max(0.0, random.gauss(pre.get('glide_ratio_mean', 0.0), pre.get('glide_ratio_std', 0.0)))
                dist = glide_ratio * base_inputs['release_alt_m']
                x0 = dist * math.cos(theta)
                y0 = dist * math.sin(theta)
                v_ms_gr = max(0.0, random.gauss(pre.get('release_airspeed_kph_mean', 0.0),
                                                pre.get('release_airspeed_kph_std', 0.0))) * (1000.0 / 3600.0)
                vx0 = v_ms_gr * math.cos(theta)
                vy0 = v_ms_gr * math.sin(theta)

        rec = simulate_one_2d({'mass': base_inputs['mass'], 'A_base': A_s, 'Cd_base': Cd_s,
                               'A_tumble_frac': sample_specs.get('A_tumble_frac', 0.0),
                               'Cd_tumble_frac': sample_specs.get('Cd_tumble_frac', 0.0),
                               'tumble_omega': omega, 'tumble_phase': phase,
                               'vref_ms': v_ms, 'wind_alpha': wind_alpha, 'wind_dir_met_deg': wind_dir_met_deg,
                               'gust_sigma': gust_sigma, 'gust_tau': gust_tau,
                               'release_alt_m': base_inputs['release_alt_m'],
                               'x0': x0, 'y0': y0, 'initial_vx': vx0, 'initial_vy': vy0, 'initial_vz': 0.0,
                               'w_mean': sample_specs.get('w_mean', 0.0),
                               'dt_slow': sample_specs.get('dt_slow', 0.02),
                               'dt_fast': sample_specs.get('dt_fast', 0.005),
                               'v_switch': sample_specs.get('v_switch', 30.0),
                               'z_switch': sample_specs.get('z_switch', 100.0),
                               'max_time': sample_specs.get('max_time', 4000.0)})

        lat, lon = meters_to_latlon(base_inputs['release_lat'], base_inputs['release_lon'], rec['x_e'], rec['y_n'])
        results.append({'run': i + 1, 'x_e': rec['x_e'], 'y_n': rec['y_n'], 't_land': rec['t_land'],
                        'lat': lat, 'lon': lon, 'A': A_s, 'Cd': Cd_s, 'wind_kph': v_kph,
                        'wind_dir_met_from_deg': wind_dir_met_deg, 'interp': int(rec.get('interp', False)),
                        'vref_ms_10m': v_ms, 'wind_alpha': wind_alpha, 'gust_sigma': gust_sigma, 'gust_tau': gust_tau})
    elapsed = time.time() - t0

    stats = {'n': N, 'elapsed_s': elapsed, 'seed': seed, 'cell_size_m': cell_size_m,
             'containment_probs': containment_probs,
             'use_kde_for_mode': use_kde, 'kde_bandwidth_m': kde_bw_m,
             'dt_slow': dt_slow, 'dt_fast': dt_fast, 'v_switch': v_switch, 'z_switch': z_switch,
             'density_model': 'exponential_rho(z)', 'integrator': 'semi-implicit Euler + interpolation',
             'model_version': 'v6', 'schema_version': '1.2'}
    return results, stats

# ---- Histogram + KDE summaries --------------------------------------------

def grid_and_contours(landings, cell_size_m=50.0, containment_probs=(0.5, 0.9, 0.95)):
    xs = [r['x_e'] for r in landings]
    ys = [r['y_n'] for r in landings]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = cell_size_m * 2
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad
    nx = max(1, int(math.ceil((xmax - xmin) / cell_size_m)))
    ny = max(1, int(math.ceil((ymax - ymin) / cell_size_m)))
    counts = {}
    for r in landings:
        ix = int((r['x_e'] - xmin) // cell_size_m)
        iy = int((r['y_n'] - ymin) // cell_size_m)
        counts[(ix, iy)] = counts.get((ix, iy), 0) + 1
    total = sum(counts.values())
    prob_grid = {k: v / total for (k, v) in counts.items()}
    mode_cell = max(prob_grid.items(), key=lambda kv: kv[1])[0]
    mode_center_x = xmin + (mode_cell[0] + 0.5) * cell_size_m
    mode_center_y = ymin + (mode_cell[1] + 0.5) * cell_size_m
    centroid_x = sum(xs) / len(xs)
    centroid_y = sum(ys) / len(ys)
    cells_sorted = sorted(prob_grid.items(), key=lambda kv: kv[1], reverse=True)
    contours = {}
    for target in sorted(containment_probs):
        accum = 0.0
        sel = []
        for (cell, p) in cells_sorted:
            sel.append(cell); accum += p
            if accum >= target: break
        centers = []
        for (ix, iy) in sel:
            cx = xmin + (ix + 0.5) * cell_size_m
            cy = ymin + (iy + 0.5) * cell_size_m
            centers.append((cx, cy))
        contours[target] = {'cells': sel, 'polygon_centers': centers, 'prob': accum}
    return {'xmin': xmin, 'ymin': ymin, 'nx': nx, 'ny': ny, 'cell_size': cell_size_m, 'prob_grid': prob_grid,
            'mode_center_hist': (mode_center_x, mode_center_y), 'centroid': (centroid_x, centroid_y),
            'contours': contours}

def kde_mode_estimate(landings, bandwidth_m=75.0, grid_cell_m=50.0):
    xs = [r['x_e'] for r in landings]
    ys = [r['y_n'] for r in landings]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = 3.0 * bandwidth_m
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad
    nx = max(1, int(math.ceil((xmax - xmin) / grid_cell_m)))
    ny = max(1, int(math.ceil((ymax - ymin) / grid_cell_m)))
    max_density = -1.0
    mode_xy = (0.0, 0.0)
    bw2 = bandwidth_m * bandwidth_m
    two_pi_bw2 = 2.0 * math.pi * bw2
    inv_two_bw2 = 1.0 / (2.0 * bw2)
    for ix in range(nx):
        cx = xmin + (ix + 0.5) * grid_cell_m
        for iy in range(ny):
            cy = ymin + (iy + 0.5) * grid_cell_m
            s = 0.0
            for (x, y) in zip(xs, ys):
                dx = x - cx; dy = y - cy
                s += math.exp(-(dx * dx + dy * dy) * inv_two_bw2)
            dens = s / (len(xs) * two_pi_bw2)
            if dens > max_density:
                max_density = dens; mode_xy = (cx, cy)
    return {'mode_center_kde': mode_xy, 'bandwidth_m': bandwidth_m, 'grid_cell_m': grid_cell_m}

def kde_hdr_contours(landings, bandwidth_m=75.0, grid_cell_m=50.0, probs=(0.5, 0.9, 0.95)):
    """
    Build KDE-based High-Density Regions (HDR) at given probabilities.
    Returns:
      {
        prob_value: {
          'cells': [(ix,iy), ...],                 # selected grid cells (density-descending)
          'polygon_centers': [(cx,cy), ...],       # cell-center XY in meters (east,north)
          'prob': accumulated_probability_estimate
        }, ...
      }
    Notes:
      - Small-area approximation: probability ≈ sum(density * cell_area).
      - Grid bounds padded by ~3σ to capture tails (same padding strategy as kde_mode_estimate).
    """
    if not landings:
        return {p: {'cells': [], 'polygon_centers': [], 'prob': 0.0} for p in probs}

    xs = [r['x_e'] for r in landings]
    ys = [r['y_n'] for r in landings]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    pad = 3.0 * float(bandwidth_m)
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad
    nx = max(1, int(math.ceil((xmax - xmin) / grid_cell_m)))
    ny = max(1, int(math.ceil((ymax - ymin) / grid_cell_m)))

    # Gaussian KDE constants
    bw2 = float(bandwidth_m) * float(bandwidth_m)
    two_pi_bw2 = 2.0 * math.pi * bw2
    inv_two_bw2 = 1.0 / (2.0 * bw2)
    n = float(len(xs))

    # Evaluate density per grid cell center
    densities = []  # list of (dens, ix, iy, cx, cy)
    for ix in range(nx):
        cx = xmin + (ix + 0.5) * grid_cell_m
        for iy in range(ny):
            cy = ymin + (iy + 0.5) * grid_cell_m
            s = 0.0
            # Tight loop; no numpy by design (stdlib-only)
            for (x, y) in zip(xs, ys):
                dx = x - cx; dy = y - cy
                s += math.exp(-(dx*dx + dy*dy) * inv_two_bw2)
            dens = s / (n * two_pi_bw2)
            densities.append((dens, ix, iy, cx, cy))

    # Sort by density (desc)
    densities.sort(key=lambda t: t[0], reverse=True)

    # Approximate probability mass per cell
    cell_area = float(grid_cell_m) * float(grid_cell_m)

    # Build HDR selections for each target prob
    targets = sorted(set(float(p) for p in probs))
    out = {p: {'cells': [], 'polygon_centers': [], 'prob': 0.0} for p in targets}

    cum = 0.0
    ti = 0  # index into targets
    # Walk once; when cum crosses a target, finalize for that target
    for dens, ix, iy, cx, cy in densities:
        pmass = dens * cell_area
        cum += pmass
        # Add to all not-yet-reached targets (they share the same top set)
        while ti < len(targets) and cum >= targets[ti] - 1e-12:
            # Fill up to this point for target targets[ti]
            # Selection is all entries up to current index
            # Collect centers quickly without re-scanning: slice densities[0:k]
            k = densities.index((dens, ix, iy, cx, cy)) + 1
            sel_cells = []
            sel_centers = []
            for j in range(k):
                _, jx, jy, jcx, jcy = densities[j]
                sel_cells.append((jx, jy))
                sel_centers.append((jcx, jcy))
            out[targets[ti]] = {
                'cells': sel_cells,
                'polygon_centers': sel_centers,
                'prob': cum
            }
            ti += 1
        if ti >= len(targets):
            break

    # Ensure all targets exist (edge cases with tiny grids)
    for p in targets:
        out.setdefault(p, {'cells': [], 'polygon_centers': [], 'prob': cum})
    return out


# ---- CSV + JSON + GeoJSON + KML writers ----------------------------------
def _kde_density_at_points(pts_xy, bandwidth_m=75.0):
    """
    Evaluate an isotropic Gaussian KDE at each point location using the sample itself
    (leave-in estimate; fine for ranking). Returns a list of densities aligned to pts_xy.
    Stdlib-only implementation.
    """
    n = len(pts_xy)
    if n == 0:
        return []
    bw2 = float(bandwidth_m) * float(bandwidth_m)
    two_pi_bw2 = 2.0 * math.pi * bw2
    inv_two_bw2 = 1.0 / (2.0 * bw2)
    dens = []
    for (xi, yi) in pts_xy:
        s = 0.0
        for (xj, yj) in pts_xy:
            dx = xi - xj
            dy = yi - yj
            s += math.exp(-(dx * dx + dy * dy) * inv_two_bw2)
        dens.append(s / (n * two_pi_bw2))
    return dens

def kdr_hdr_contours(landings, probs=(0.5, 0.9, 0.95), bandwidth_m=75.0):
    """
    KDE Highest-Density Regions via Kernel Density Rank (KDR):
      1) Compute KDE density at each landing sample.
      2) Sort samples by density (desc).
      3) For each probability p, take top p*N samples and form a convex hull.
    Returns: {p: {'polygon_centers': [(x,y), ...], 'method': 'KDE_KDR'}}  (meters, x=east,y=north)
    """
    pts = [(r['x_e'], r['y_n']) for r in landings]
    n = len(pts)
    out = {}
    if n == 0:
        for p in probs:
            out[p] = {'polygon_centers': [], 'method': 'KDE_KDR'}
        return out
    dens = _kde_density_at_points(pts, bandwidth_m=bandwidth_m)
    order = sorted(range(n), key=lambda i: dens[i], reverse=True)
    for p in sorted(probs):
        k = max(1, min(n, int(round(p * n))))
        sel = [pts[i] for i in order[:k]]
        hull = _convex_hull(sel)
        if len(hull) == 2:
            hull = [hull[0], hull[1], hull[1], hull[0]]
        out[p] = {'polygon_centers': hull, 'method': 'KDE_KDR'}
    return out




def save_landings_csv(landings, filename):
    keys = ['run', 'x_e', 'y_n', 't_land', 'lat', 'lon', 'A', 'Cd', 'wind_kph', 'wind_dir_met_from_deg', 'interp',
            'vref_ms_10m', 'wind_alpha', 'gust_sigma', 'gust_tau']
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(keys)
        for r in landings: w.writerow([r.get(k, '') for k in keys])

def save_summary_json(filename, base_inputs, stats, grid_info, mode_latlon_hist,
                      centroid_latlon, sample_specs, kde_info=None, radii_mode=None,
                      radii_centroid=None, radial_from_release=None, spiral_info=None,
                      radii_kde=None, contours_kde_hdr=None, contours_kde_kdr=None):

    release_latlon = (base_inputs['release_lat'], base_inputs['release_lon'])
    out = {'mode_latlon_hist': mode_latlon_hist, 'centroid_latlon': centroid_latlon,
           'radii_mode_hist_m': radii_mode if radii_mode else {},
           'release_latlon': release_latlon,
           'radii_centroid_m': radii_centroid if radii_centroid else {},
           'radial_percentiles_from_release_m': radial_from_release if radial_from_release else {},
           'radii_mode_kde_m': radii_kde if radii_kde else {},
           'spiral_info': spiral_info if spiral_info else {},
           'grid_cell_m': grid_info['cell_size'], 'containment_probs': list(grid_info['contours'].keys()),
           'contours': {str(k): v for k, v in grid_info['contours'].items()},
           'metadata': {'base_inputs': base_inputs, 'stats': stats,
                        'conventions': {'wind_dir': sample_specs.get('wind_dir_convention', 'met_from'),
                                        'heading': sample_specs.get('pre_release', {}).get('heading_convention', 'met_toward'),
                                        'angle_frame': 'code: 0°=east, CCW positive; inputs: MET FROM (wind), MET TO (heading)'},
                        'open_meteo': {'timezone': sample_specs.get('openmeteo_tz', None), 'timezone_abbr': sample_specs.get('openmeteo_tz_abbr', None)}}}
    if kde_info:
        mode_x, mode_y = kde_info['mode_center_kde']
        lat, lon = meters_to_latlon(base_inputs['release_lat'], base_inputs['release_lon'], mode_x, mode_y)
        out['mode_latlon_kde'] = (lat, lon)
        out['kde_settings'] = {'bandwidth_m': kde_info['bandwidth_m'], 'grid_cell_m': kde_info['grid_cell_m']}

    if contours_kde_hdr:
        out['contours_kde_hdr'] = {str(k): v for k, v in contours_kde_hdr.items()}
    if contours_kde_kdr:
        out['contours_kde_kdr'] = {str(k): v for k, v in contours_kde_kdr.items()}

    with open(filename, 'w') as f:
        json.dump(out, f)

def _convex_hull(points):
    pts = [tuple(p) for p in points]
    pts = sorted(set(pts))
    if len(pts) <= 1: return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def _centers_to_polygon_latlon(centers_m, rlat, rlon):
    if len(centers_m) == 0:
        return []
    hull = _convex_hull(centers_m)
    # Require at least 3 distinct points for a valid polygon ring
    if len(hull) < 3:
        return []
    ring = []
    for (xe, yn) in hull:
        lat, lon = meters_to_latlon(rlat, rlon, xe, yn)
        ring.append([lon, lat])
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring

    
def _circle_ring_lonlat(center_lat, center_lon, radius_m, n_vertices=96):
    """
    Build a closed lon/lat ring approximating a circle of radius_m (meters)
    around (center_lat, center_lon) using the small-area meters->lat/lon conversion.
    """
    if radius_m is None or not math.isfinite(radius_m) or radius_m <= 0:
        return []
    ring = []
    for i in range(n_vertices):
        ang = (2.0 * math.pi * i) / n_vertices
        xe = radius_m * math.cos(ang)
        yn = radius_m * math.sin(ang)
        lat, lon = meters_to_latlon(center_lat, center_lon, xe, yn)
        ring.append([lon, lat])
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring



def export_geojson_kml(summary_json_path, out_geojson_path, out_kml_path):
    # --- helpers ---
    def _desc(text: str) -> str:
        return f"<description><![CDATA[{text}]]></description>"

    DESC_MODE_HIST = ("Most-populated grid cell of binned landings; the empirical "
                      "histogram peak (no smoothing).")
    DESC_MODE_KDE = ("Highest-density point on the Gaussian KDE surface of landings "
                     "(bandwidth controls smoothing).")
    DESC_CENTROID = ("Arithmetic mean of all landing coordinates (center of mass).")

    def _desc_cov_hist(pct: int) -> str:
        return _desc(f"Coverage — Grid Histogram ({pct}%): "
                     "Sort grid cells by empirical probability and accumulate until the "
                     f"total ≥ {pct}%. A single convex hull is drawn for a flyable boundary.")

    def _desc_hdr_kde(pct: int) -> str:
        return _desc(f"High-Density Region — KDE Surface ({pct}%): "
                     "Rank grid cells by KDE value (density), accumulate to target probability, "
                     "then draw one convex hull around selected centers.")

    def _desc_hdr_kdr(pct: int) -> str:
        return _desc(f"High-Density Region — KDE Rank Hull ({pct}%): "
                     "Compute KDE at each sample, keep the top {pct}% most-dense samples, "
                     "and draw their convex hull.")

    def _desc_radius(center_label: str, pct: int) -> str:
        return _desc(f"Inclusion Radius — {center_label} ({pct}%): "
                     "Minimum radius about the stated center that contains the given "
                     "fraction of simulated landings.")
    with open(summary_json_path, "r") as f:
        data = json.load(f)

    mode_hist_lat, mode_hist_lon = data.get("mode_latlon_hist", (0.0, 0.0))
    cent_lat, cent_lon = data.get("centroid_latlon", (0.0, 0.0))

    rlat, rlon = data.get("release_latlon", (0.0, 0.0))
    if not (math.isfinite(rlat) and math.isfinite(rlon) and (abs(rlat) + abs(rlon) > 0.0)):
        base = data.get("metadata", {}).get("base_inputs", {})
        rlat = float(base.get("release_lat", 0.0) or 0.0)
        rlon = float(base.get("release_lon", 0.0) or 0.0)
    if not (math.isfinite(rlat) and math.isfinite(rlon) and (abs(rlat) + abs(rlon) > 0.0)):
        if (abs(cent_lat) + abs(cent_lon)) > 0.0:
            rlat, rlon = cent_lat, cent_lon
        else:
            rlat, rlon = mode_hist_lat, mode_hist_lon

    try:
        contours_raw_dbg = data.get("contours", {})
        first_centers = next(iter(contours_raw_dbg.values())).get("polygon_centers", [])[:3]
        print(f"[export] anchor used: ({rlat:.6f}, {rlon:.6f}); sample centers: {first_centers}")
    except Exception:
        print(f"[export] anchor used: ({rlat:.6f}, {rlon:.6f})")

    if not (abs(rlat) + abs(rlon)) and not (abs(mode_hist_lat) + abs(mode_hist_lon)) and not (abs(cent_lat) + abs(cent_lon)):
        raise ValueError("No valid lat/lon anchor found in summary JSON (release/centroid/mode all 0.0). Re-run the simulation with a non-zero release location.")
    
    print(f"[export] containment anchor: lat={rlat:.6f}, lon={rlon:.6f}")
    features = []

    #mode_hist_lat, mode_hist_lon = data["mode_latlon_hist"]
    #cent_lat, cent_lon = data["centroid_latlon"]
    features.append({"type": "Feature", "properties": {"name": "Mode — Grid Histogram"},
                     "geometry": {"type": "Point", "coordinates": [mode_hist_lon, mode_hist_lat]}})
    features.append({"type": "Feature", "properties": {"name": "Mean of Landings (Centroid)"},
                     "geometry": {"type": "Point", "coordinates": [cent_lon, cent_lat]}})
    if "mode_latlon_kde" in data:
        kde_lat, kde_lon = data["mode_latlon_kde"]
        features.append({"type": "Feature", "properties": {"name": "Mode — KDE Surface"},
                         "geometry": {"type": "Point", "coordinates": [kde_lon, kde_lat]}})
                         
    contours_raw = data.get("contours", {})
    # Keep this float conversion; some runs save keys as strings in JSON.
    contours = {float(k): v for k, v in contours_raw.items()}

    for lvl in sorted(contours.keys()):
        centers_m = contours[lvl].get("polygon_centers", [])
        ring = _centers_to_polygon_latlon(centers_m, rlat, rlon)

        # Guard: skip invalid/degenerate rings (prevents ghost polygons at 0,0)
        if not ring or len(ring) < 4:
            print(f"[export] skip GeoJSON containment {lvl*100:.0f}% (degenerate hull)")
            continue

        features.append({
            "type": "Feature",
            "properties": {
                "name": f"{int(round(lvl * 100))}% Containment",
                "containment_prob": float(lvl)
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]}
        })
    # --- KDE containment polygons (GeoJSON; optional) ---
    if "contours_kde_hdr" in data:
        contours_kde_hdr = data["contours_kde_hdr"]
        levels_sorted_hdr = sorted(contours_kde_hdr.keys(), key=lambda s: float(s))
        for lvl in levels_sorted_hdr:
            ring_hdr = _centers_to_polygon_latlon(contours_kde_hdr[lvl].get("polygon_centers", []), rlat, rlon)
            if ring_hdr and len(ring_hdr) >= 4:
                features.append({
                    "type": "Feature",
                    "properties": {"name": f"{float(lvl) * 100:.0f}% HDR (KDE)", "containment_prob": float(lvl),
                                   "kde": "hdr"},
                    "geometry": {"type": "Polygon", "coordinates": [ring_hdr]}
                })

    if "contours_kde_kdr" in data:
        contours_kde_kdr = data["contours_kde_kdr"]
        levels_sorted_kdr = sorted(contours_kde_kdr.keys(), key=lambda s: float(s))
        for lvl in levels_sorted_kdr:
            ring_kdr = _centers_to_polygon_latlon(contours_kde_kdr[lvl].get("polygon_centers", []), rlat, rlon)
            if ring_kdr and len(ring_kdr) >= 4:
                features.append({
                    "type": "Feature",
                    "properties": {"name": f"{float(lvl) * 100:.0f}% HDR (KDE/KDR)", "containment_prob": float(lvl),
                                   "kde": "kdr"},
                    "geometry": {"type": "Polygon", "coordinates": [ring_kdr]}
                })

    # --- NEW: add percentile radius circles as polygons (GeoJSON) ---
    def _add_circle_features(name_prefix, center_lat, center_lon, radii_dict):
        if not radii_dict:
            return
        for key in sorted(radii_dict.keys(), key=lambda k: float(k)):
            p = float(key)
            rad = float(radii_dict[key])
            ring = _circle_ring_lonlat(center_lat, center_lon, rad, n_vertices=96)
            if not ring:
                continue
            features.append({
                "type": "Feature",
                "properties": {"name": f"{int(round(p*100))}% Radius ({name_prefix})",
                               "center": name_prefix, "percent": p, "radius_m": rad},
                "geometry": {"type": "Polygon", "coordinates": [ring]}
            })

    # Optional: KDE KDR/HDR polygons
    if "contours_kde_kdr" in data:
        contours_kdr = data["contours_kde_kdr"]
        kdr_levels = sorted(contours_kdr.keys(), key=lambda s: float(s))
        for lvl in kdr_levels:
            centers = contours_kdr[lvl].get("polygon_centers", [])
            ring = _centers_to_polygon_latlon(centers, rlat, rlon)
            features.append({"type": "Feature",
                             "properties": {"name": f"{float(lvl)*100:.0f}% HDR (KDE)", "containment_prob": float(lvl), "method": "KDE_KDR"},
                             "geometry": {"type": "Polygon", "coordinates": [ring]}})





    _add_circle_features("Mode (hist)", mode_hist_lat, mode_hist_lon, data.get("radii_mode_hist_m", {}))
    if "mode_latlon_kde" in data:
        kde_lat, kde_lon = data["mode_latlon_kde"]
        _add_circle_features("Mode (KDE)", kde_lat, kde_lon, data.get("radii_mode_kde_m", {}))
        
    _add_circle_features("Centroid", cent_lat, cent_lon, data.get("radii_centroid_m", {}))
    _add_circle_features("Release", rlat, rlon, data.get("radial_percentiles_from_release_m", {}))

                         
    geojson = {"type": "FeatureCollection", "name": "search_area_v6", "features": features}
    with open(out_geojson_path, "w") as f:
        json.dump(geojson, f)
    kml = [
        '<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document>\n  <name>search_area_v6</name>\n'
    ]
    # DEBUG: write the anchor as a placemark so we can visually confirm the file is current
    kml.append(
        f"  <Placemark><name>Mode — Grid Histogram</name>"
        f"{_desc(DESC_MODE_HIST)}"
        f"<Point><coordinates>{mode_hist_lon},{mode_hist_lat},0</coordinates></Point></Placemark>\n"
    )
    kml.append(
        f"  <Placemark><name>Mean of Landings (Centroid)</name>"
        f"{_desc(DESC_CENTROID)}"
        f"<Point><coordinates>{cent_lon},{cent_lat},0</coordinates></Point></Placemark>\n"
    )
    if "mode_latlon_kde" in data:
        kml.append(
            f"  <Placemark><name>Mode — KDE Surface</name>"
            f"{_desc(DESC_MODE_KDE)}"
            f"<Point><coordinates>{kde_lon},{kde_lat},0</coordinates></Point></Placemark>\n"
        )
    # --- KDE containment polygons (KML; optional) ---
    # Draw grid‑HDR first if present
    if "contours_kde_hdr" in data:
        contours_kde_hdr = data["contours_kde_hdr"]
        levels_sorted_hdr = sorted(contours_kde_hdr.keys(), key=lambda s: float(s))
        for lvl in levels_sorted_hdr:
            ring_hdr = _centers_to_polygon_latlon(contours_kde_hdr[lvl].get("polygon_centers", []), rlat, rlon)
            # Skip degenerate rings (<4 points after closure)
            if not ring_hdr or len(ring_hdr) < 4:
                continue
            coords_hdr = " ".join([f"{lon},{lat},0" for lon, lat in ring_hdr])
            kml.append(
                "  <Placemark>"
                f"<name>{float(lvl) * 100:.0f}% HDR (KDE)</name>"
                "<Polygon><outerBoundaryIs><LinearRing>"
                f"<coordinates>{coords_hdr}</coordinates>"
                "</LinearRing></outerBoundaryIs></Polygon>"
                "</Placemark>\n"
            )

    # Then draw sample‑rank KDR (if present)
    if "contours_kde_kdr" in data:
        contours_kde_kdr = data["contours_kde_kdr"]
        levels_sorted_kdr = sorted(contours_kde_kdr.keys(), key=lambda s: float(s))
        for lvl in levels_sorted_kdr:
            ring_kdr = _centers_to_polygon_latlon(contours_kde_kdr[lvl].get("polygon_centers", []), rlat, rlon)
            if not ring_kdr or len(ring_kdr) < 4:
                continue
            coords_kdr = " ".join([f"{lon},{lat},0" for lon, lat in ring_kdr])
            kml.append(
                "  <Placemark>"
                f"<name>{float(lvl) * 100:.0f}% High-Density Region — KDE Surface</name>"
                f"{_desc_hdr_kde(int(float(lvl) * 100))}"
                "<Polygon><outerBoundaryIs><LinearRing>"
                f"<coordinates>{coords_hdr}</coordinates>"
                "</LinearRing></outerBoundaryIs></Polygon>"
                "</Placemark>\n"
            )


    for lvl in sorted(contours.keys()):
        ring = _centers_to_polygon_latlon(contours[lvl].get("polygon_centers", []), rlat, rlon)
        coords = " ".join([f"{lon},{lat},0" for lon, lat in ring])
        kml.append(
            "  <Placemark>"
            f"<name>{int(round(lvl * 100))}% Coverage — Grid Histogram</name>"
            f"{_desc_cov_hist(int(round(lvl * 100)))}"
            "<Polygon><outerBoundaryIs><LinearRing>"
            f"<coordinates>{coords}</coordinates>"
            "</LinearRing></outerBoundaryIs></Polygon>"
            "</Placemark>\n"
        )
    # --- NEW: add percentile radius circles as polygons (KML) ---
    def _kml_circle(name_prefix, center_lat, center_lon, radii_dict):
        if not radii_dict:
            return
        for key in sorted(radii_dict.keys(), key=lambda k: float(k)):
            p = float(key)
            rad = float(radii_dict[key])
            ring = _circle_ring_lonlat(center_lat, center_lon, rad, n_vertices=96)
            if not ring:
                continue
            coords = " ".join([f"{lon},{lat},0" for lon, lat in ring])
            pct = int(round(p * 100))  # convert 0.5 -> 50
            kml.append(
                "  <Placemark>"
                f"<name>{pct}% Radius ({name_prefix})</name>"
                f"{_desc_radius(name_prefix, pct)}"
                "<Polygon><outerBoundaryIs><LinearRing>"
                f"<coordinates>{coords}</coordinates>"
                "</LinearRing></outerBoundaryIs></Polygon>"
                "</Placemark>\n"
            )

    _kml_circle("Mode (Grid)", mode_hist_lat, mode_hist_lon, data.get("radii_mode_hist_m", {}))
    if "mode_latlon_kde" in data:
        _kml_circle("Mode (KDE)", kde_lat, kde_lon, data.get("radii_mode_kde_m", {}))
    _kml_circle("Centroid", cent_lat, cent_lon, data.get("radii_centroid_m", {}))
    _kml_circle("Release", rlat, rlon, data.get("radial_percentiles_from_release_m", {}))

    kml.append("</Document></kml>\n")
    with open(out_kml_path, "w") as f:
        f.write("".join(kml))

# ---------- NEW IN V6: distance-first radii + spiral search helpers ----------

def _radii_from_center(landings, center_xy, probs=(0.5, 0.9, 0.95)):
    """
    Compute minimal radii (in meters) around a given center (x0,y0) so that
    the fraction of samples within radius r meets each target probability.
    Returns a dict {prob: radius_m}.
    """
    x0, y0 = center_xy
    ds = sorted((((r['x_e'] - x0)**2 + (r['y_n'] - y0)**2) ** 0.5) for r in landings)
    out = {}
    for p in sorted(probs):
        if not ds:
            out[p] = float('nan'); continue
        k = max(1, int(round(p * len(ds))))
        k = min(k, len(ds))
        out[p] = ds[k - 1]
    return out

def _spiral_xy_points(max_radius_m, spacing_m=50.0, step_m=10.0):
    """
    Generate Archimedean spiral points (x=east,y=north) from 0 to max_radius_m.
    spacing_m sets the distance between successive turns (≈ 2πb).
    step_m controls linear distance between emitted points along the spiral.
    Returns list of (x,y) tuples.
    """
    if max_radius_m <= 0:
        return [(0.0, 0.0)]
    b = spacing_m / (2.0 * math.pi)  # r = b * theta
    pts = []
    theta = 0.0
    r = 0.0
    pts.append((0.0, 0.0))
    while r < max_radius_m:
        dtheta = step_m / max(1e-6, math.hypot(b, r))
        theta += dtheta
        r = b * theta
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        pts.append((x, y))
    return pts

def save_spiral_geojson(out_path, pts_xy, rlat, rlon):
    coords = []
    for (xe, yn) in pts_xy:
        lat, lon = meters_to_latlon(rlat, rlon, xe, yn)
        coords.append([lon, lat])
    gj = {"type": "FeatureCollection", "name": "spiral_search_v6",
          "features": [{"type": "Feature", "properties": {"name": "Spiral Search Path"},
                        "geometry": {"type": "LineString", "coordinates": coords}}]}
    with open(out_path, "w") as f:
        json.dump(gj, f)

# ---- Util ------------------------------------------------------------------

def _slugify(text):
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
    text = ''.join(ch for ch in text if ch.isalnum() or ch in (' ', '-', '_'))
    text = text.strip().replace(' ', '_')
    return text[:40] if text else 'incident'

def make_run_dir(incident_name="incident"):
    os.makedirs("data/runs", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    slug = _slugify(incident_name)
    run_dir = os.path.join("data/runs", f"{ts}_{slug}_v6")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def validate_config(CONFIG):
    errors = []
    if CONFIG['object']['mass'] <= 0: errors.append("mass must be > 0 kg")
    if CONFIG['object']['A_mean'] <= 0: errors.append("A_mean must be > 0 m^2")
    if CONFIG['object']['Cd_mean'] <= 0: errors.append("Cd_mean must be > 0")
    if CONFIG['winds']['wind_speed_kph_mean'] < 0: errors.append("wind_speed_kph_mean cannot be negative")
    return errors

# ---- (Optional) Open-Meteo winds (same as v5, trimmed) ---------------------

import urllib.request, urllib.parse

def _http_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _to_params(base):
    return urllib.parse.urlencode(base, doseq=True)

def fetch_wind_current(lat, lon, tz="auto"):
    p = {"latitude": f"{lat:.6f}", "longitude": f"{lon:.6f}",
         "current": "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
         "wind_speed_unit": "kmh", "timezone": tz}
    url = "https://api.open-meteo.com/v1/forecast?" + _to_params(p)
    j = _http_json(url)
    cur = j.get("current", {})
    spd = float(cur.get("wind_speed_10m", float("nan")))
    ddd = float(cur.get("wind_direction_10m", float("nan")))
    gust = float(cur.get("wind_gusts_10m", 0.0))
    tz_name = j.get("timezone", tz)
    tz_abbr = j.get("timezone_abbreviation", "")
    return {"speed": spd, "dir": ddd, "gust": gust, "tz": tz_name, "tz_abbr": tz_abbr}

def apply_open_meteo_to_config_at(CONFIG, mode="current", hours=-1, when_iso=None, tz="UTC"):
    lat = CONFIG["location"]["release_lat"]
    lon = CONFIG["location"]["release_lon"]
    try:
        if mode == "current":
            w = fetch_wind_current(lat, lon, tz=tz)
        else:
            w = fetch_wind_current(lat, lon, tz=tz)
        ws_mean = w["speed"] if math.isfinite(w["speed"]) else 10.0
        ws_std = max(0.0, ws_mean * 0.2)
        wd_mean = w["dir"] if math.isfinite(w["dir"]) else 90.0
        wd_std = 15.0
        CONFIG.setdefault("winds", {}).update({
            "wind_speed_kph_mean": ws_mean,
            "wind_speed_kph_std": ws_std,
            "wind_dir_met_from_mean": wd_mean % 360.0,
            "wind_dir_met_from_std": wd_std,
            "wind_alpha": CONFIG["winds"].get("wind_alpha", 0.143),
            "gust_sigma": CONFIG["winds"].get("gust_sigma", max(0.0, ws_std / 3.0)),
            "gust_tau": CONFIG["winds"].get("gust_tau", 5.0),
            "w_mean": CONFIG["winds"].get("w_mean", 0.0)
        })
        return True, {"when": mode, "wind": w}
    except Exception as e:
        print("Wind fetch failed:", e)
        return False, None

# ---- End-to-end runner -----------------------------------------------------

def run_from_config(CONFIG):
    errs = validate_config(CONFIG)
    if errs:
        print("CONFIG errors:"); [print(' -', e) for e in errs]; return None

    base_inputs = {'release_lat': CONFIG['location']['release_lat'],
                   'release_lon': CONFIG['location']['release_lon'],
                   'release_alt_m': CONFIG['location']['release_alt_m_agl'],
                   'mass': CONFIG['object']['mass'],
                   'A_mean': CONFIG['object']['A_mean'],
                   'Cd_mean': CONFIG['object']['Cd_mean'],
                   'seed': CONFIG.get('seed', 1337)}

    sample_specs = {'wind_dir_convention': 'met_from',
                    'wind_speed_kph_mean': CONFIG['winds']['wind_speed_kph_mean'],
                    'wind_speed_kph_std': CONFIG['winds']['wind_speed_kph_std'],
                    'wind_dir_met_from_mean': CONFIG['winds']['wind_dir_met_from_mean'],
                    'wind_dir_met_from_std': CONFIG['winds']['wind_dir_met_from_std'],
                    'wind_alpha': CONFIG['winds'].get('wind_alpha', 0.143),
                    'gust_sigma': CONFIG['winds'].get('gust_sigma', 1.0),
                    'gust_tau': CONFIG['winds'].get('gust_tau', 5.0),
                    'A_std': CONFIG['object'].get('A_std', 0.0),
                    'Cd_std': CONFIG['object'].get('Cd_std', 0.0),
                    'rho_A_Cd': CONFIG['object'].get('rho_A_Cd', 0.0),
                    'A_tumble_frac': CONFIG['object'].get('A_tumble_frac', 0.0),
                    'Cd_tumble_frac': CONFIG['object'].get('Cd_tumble_frac', 0.0),
                    'tumble_omega_mean': CONFIG['object'].get('tumble_omega_mean', 0.0),
                    'tumble_omega_std': CONFIG['object'].get('tumble_omega_std', 0.0),
                    'pre_release': {'enabled': CONFIG['pre_release']['enabled'],
                                    'heading_convention': 'met_toward',
                                    'mode': CONFIG['pre_release']['mode'],
                                    'time_s_mean': CONFIG['pre_release'].get('time_s_mean', 0.0),
                                    'time_s_std': CONFIG['pre_release'].get('time_s_std', 0.0),
                                    'groundspeed_kph_mean': CONFIG['pre_release'].get('groundspeed_kph_mean', 0.0),
                                    'groundspeed_kph_std': CONFIG['pre_release'].get('groundspeed_kph_std', 0.0),
                                    'heading_met_toward_mean': CONFIG['pre_release'].get('heading_met_toward_mean', 0.0),
                                    'heading_met_toward_std': CONFIG['pre_release'].get('heading_met_toward_std', 0.0),
                                    'glide_ratio_mean': CONFIG['pre_release'].get('glide_ratio_mean', 0.0),
                                    'glide_ratio_std': CONFIG['pre_release'].get('glide_ratio_std', 0.0),
                                    'release_airspeed_kph_mean': CONFIG['pre_release'].get('release_airspeed_kph_mean', 0.0),
                                    'release_airspeed_kph_std': CONFIG['pre_release'].get('release_airspeed_kph_std', 0.0)},
                    'cell_size_m': CONFIG['numerics'].get('cell_size_m', 50.0),
                    'containment_probs': tuple(CONFIG['outputs'].get('containment_probs', [0.5, 0.9, 0.95])),
                    'use_kde_for_mode': CONFIG['outputs'].get('use_kde_for_mode', True),
                    'kde_bandwidth_m': CONFIG['outputs'].get('kde_bandwidth_m', 75.0),
                    'dt_slow': CONFIG['numerics'].get('dt_slow', 0.02),
                    'dt_fast': CONFIG['numerics'].get('dt_fast', 0.005),
                    'v_switch': CONFIG['numerics'].get('v_switch', 30.0),
                    'z_switch': CONFIG['numerics'].get('z_switch', 100.0),
                    'max_time': CONFIG['numerics'].get('max_time', 4000.0),
                    'w_mean': CONFIG['winds'].get('w_mean', 0.0)}

    N = CONFIG.get('N_runs', 500)
    run_dir = make_run_dir(CONFIG['incident']['name'])
    csv_path = os.path.join(run_dir, "landings_v6.csv")
    json_path = os.path.join(run_dir, "search_area_v6.json")
    geo_path = os.path.join(run_dir, "search_area_v6.geojson")
    kml_path = os.path.join(run_dir, "search_area_v6.kml")

    landings, stats = run_monte_carlo(N, base_inputs, sample_specs)
    save_landings_csv(landings, csv_path)
    grid_info = grid_and_contours(landings, cell_size_m=sample_specs.get('cell_size_m', 50.0),
                                  containment_probs=sample_specs.get('containment_probs', (0.5, 0.9, 0.95)))
    mode_x, mode_y = grid_info['mode_center_hist']
    cent_x, cent_y = grid_info['centroid']
    mode_lat, mode_lon = meters_to_latlon(base_inputs['release_lat'], base_inputs['release_lon'], mode_x, mode_y)
    cent_lat, cent_lon = meters_to_latlon(base_inputs['release_lat'], base_inputs['release_lon'], cent_x, cent_y)

    # --- Sanity check: empirical coverage of containment polygons ---
    # def _point_in_ring(lon, lat, ring_lonlat):
    #     # Ray casting (odd-even) rule; ring_lonlat is [[lon,lat], ...] CLOSED
    #     inside = False
    #     n = len(ring_lonlat)
    #     if n < 4:  # must be closed with >= 3 distinct points
    #         return False
    #     for i in range(n - 1):
    #         x1, y1 = ring_lonlat[i]
    #         x2, y2 = ring_lonlat[i + 1]
    #         # Check if edge crosses horizontal ray at y=lat
    #         if ((y1 > lat) != (y2 > lat)):
    #             x_int = x1 + (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-15)
    #             if x_int > lon:
    #                 inside = not inside
    #     return inside
    #
    # def _ring_from_centers_m(centers_m, rlat, rlon):
    #     ring = _centers_to_polygon_latlon(centers_m, rlat, rlon)
    #     return ring
    #
    # rlat_chk = base_inputs['release_lat']
    # rlon_chk = base_inputs['release_lon']
    # contours = grid_info['contours']  # same structure used in exporter
    #
    # # Convert landing XY to lon/lat once for speed
    # land_ll = []
    # for r in landings:
    #     xe = r['x_e']
    #     yn = r['y_n']
    #     lat_i, lon_i = meters_to_latlon(rlat_chk, rlon_chk, xe, yn)
    #     land_ll.append((lon_i, lat_i))  # store as (lon, lat) for point-in-ring
    #
    # print("—" * 64)
    # print("[check] Empirical containment coverage:")
    # for p in sorted(contours.keys()):
    #     centers_m = contours[p]['polygon_centers']
    #     ring = _ring_from_centers_m(centers_m, rlat_chk, rlon_chk)
    #     if not ring or len(ring) < 4:
    #         print(f"  {int(p * 100)}%: SKIPPED (degenerate ring)")
    #         continue
    #     inside = sum(1 for lon_i, lat_i in land_ll if _point_in_ring(lon_i, lat_i, ring))
    #     frac = inside / len(land_ll) if land_ll else float('nan')
    #     print(f"  {int(p * 100)}%: observed={frac:.3f} vs target={p:.3f}  (Δ={frac - p:+.3f})")

# --- End Sanity Check ---

    kde_info = None
    if sample_specs.get('use_kde_for_mode', True):
        kde = kde_mode_estimate(landings, bandwidth_m=sample_specs.get('kde_bandwidth_m', 75.0),
                                grid_cell_m=sample_specs.get('cell_size_m', 50.0))
        kde_info = {'mode_center_kde': kde['mode_center_kde'], 'bandwidth_m': kde['bandwidth_m'],
                    'grid_cell_m': kde['grid_cell_m']}
        
        # Radii from KDE mode (when available)
        radii_kde = None
        if kde_info:
        	radii_kde = _radii_from_center(
        	landings, kde_info['mode_center_kde'],
        	probs=tuple(sample_specs.get('containment_probs', (0.5, 0.9, 0.95))))


    # --- V6: distance-first radii + spiral ---
    mode_xy = grid_info['mode_center_hist']
    centroid_xy = grid_info['centroid']
    radii_mode = _radii_from_center(landings, mode_xy, probs=tuple(sample_specs.get('containment_probs', (0.5, 0.9, 0.95))))
    radii_centroid = _radii_from_center(landings, centroid_xy, probs=tuple(sample_specs.get('containment_probs', (0.5, 0.9, 0.95))))
    radial_from_release = _radii_from_center(landings, (0.0, 0.0), probs=(0.5, 0.9, 0.95))

    contours_kde_hdr = None
    if sample_specs.get('use_kde_for_mode', True):
        # Build KDE HDR polygons at the same probabilities as histogram containment
        probs = tuple(sample_specs.get('containment_probs', (0.5, 0.9, 0.95)))
        contours_kde_hdr = kde_hdr_contours(
            landings,
            bandwidth_m=sample_specs.get('kde_bandwidth_m', 75.0),
            grid_cell_m=sample_specs.get('cell_size_m', 50.0),
            probs=probs
        )

    spiral_spacing = CONFIG.get('spiral', {}).get('spacing_m', 50.0)
    spiral_step = CONFIG.get('spiral', {}).get('step_m', 10.0)
    spiral_max = radial_from_release.get(0.95, 0.0)
    spiral_pts = _spiral_xy_points(spiral_max, spacing_m=spiral_spacing, step_m=spiral_step)
    spiral_geo_path = os.path.join(run_dir, 'spiral_search_v6.geojson')
    save_spiral_geojson(spiral_geo_path, spiral_pts, base_inputs['release_lat'], base_inputs['release_lon'])

    # --- KDE KDR/HDR polygons (added; keeps existing histogram polygons) ---
    if sample_specs.get('use_kde_for_mode', True):
        try:
            contours_kde_kdr = kdr_hdr_contours(landings, probs=tuple(sample_specs.get('containment_probs', (0.5, 0.9, 0.95))),
                                                bandwidth_m=sample_specs.get('kde_bandwidth_m', 75.0))
        except Exception as _e:
            contours_kde_kdr = None  # non-fatal


    save_summary_json(json_path, base_inputs, stats, grid_info, (mode_lat, mode_lon), (cent_lat, cent_lon),
                      sample_specs, kde_info=kde_info,
                      radii_mode=radii_mode, radii_centroid=radii_centroid,
                      radial_from_release=radial_from_release,
                      spiral_info={'spacing_m': spiral_spacing, 'step_m': spiral_step, 'max_radius_m': spiral_max},
                      radii_kde=radii_kde,
                      contours_kde_kdr=contours_kde_kdr,
                      contours_kde_hdr=contours_kde_hdr)

    export_geojson_kml(json_path, geo_path, kml_path)

    wind_line = f"Wind@10m: {sample_specs['wind_speed_kph_mean']:.1f}±{sample_specs['wind_speed_kph_std']:.1f} kph FROM {sample_specs['wind_dir_met_from_mean']:.0f}±{sample_specs['wind_dir_met_from_std']:.0f} (MET)"
    print("—" * 64)
    print(f"Incident: {CONFIG['incident']['name']}  |  Seed: {base_inputs['seed']}  |  N={N}")
    print(wind_line)
    print(f"Mode(hist):   {mode_lat:.6f}, {mode_lon:.6f}")
    if kde_info:
        mkx, mky = kde_info['mode_center_kde']
        mlat, mlon = meters_to_latlon(base_inputs['release_lat'], base_inputs['release_lon'], mkx, mky)
        print(f"Mode(KDE):    {mlat:.6f}, {mlon:.6f}")
    print(f"Centroid:     {cent_lat:.6f}, {cent_lon:.6f}")
    print(f"Radii (from Release, m): 50={radial_from_release.get(0.5, float('nan')):.1f}, 90={radial_from_release.get(0.9, float('nan')):.1f}, 95={radial_from_release.get(0.95, float('nan')):.1f}")
    if kde_info and 'mode_center_kde' in kde_info:
    	# radii_kde may be None if KDE is disabled
    	if radii_kde:
        	print(f"Radii (from Mode[KDE], m): 50={radii_kde.get(0.5, float('nan')):.1f}, 90={radii_kde.get(0.9, float('nan')):.1f}, 95={radii_kde.get(0.95, float('nan')):.1f}")

    print(f"Radii (from Mode, m): 50={radii_mode.get(0.5, float('nan')):.1f}, 90={radii_mode.get(0.9, float('nan')):.1f}, 95={radii_mode.get(0.95, float('nan')):.1f}")
    print(f"Radii (from Centroid, m): 50={radii_centroid.get(0.5, float('nan')):.1f}, 90={radii_centroid.get(0.9, float('nan')):.1f}, 95={radii_centroid.get(0.95, float('nan')):.1f}")

    # --- DEBUG: check distances between centers ---
    def _dist_m(a_xy, b_xy):
        (ax, ay), (bx, by) = a_xy, b_xy
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    mode_xy = grid_info['mode_center_hist']
    centroid_xy = grid_info['centroid']
    release_xy = (0.0, 0.0)

    print("[check] Center separations (m):")
    print(f"  |Mode(hist)-Centroid| = {_dist_m(mode_xy, centroid_xy):.2f}")
    print(f"  |Mode(hist)-Release| = {_dist_m(mode_xy, release_xy):.2f}")
    print(f"  |Centroid-Release|   = {_dist_m(centroid_xy, release_xy):.2f}")
    # --- DEBUG END: check distances between centers ---

    print(f"Spiral out-from-release to 95% radius: {spiral_max:.1f} m (spacing {spiral_spacing:.0f} m)")

    if kde_info:
        kde_xy = kde_info['mode_center_kde']
        print(f"  |Mode(hist)-Mode(KDE)| = {_dist_m(mode_xy, kde_xy):.2f}")
        print(f"  |Centroid-Mode(KDE)|   = {_dist_m(centroid_xy, kde_xy):.2f}")
        print(f"  |Release-Mode(KDE)|    = {_dist_m(release_xy, kde_xy):.2f}")

    print("Files:")
    print(" ", csv_path)
    print(" ", json_path)
    print(" ", geo_path)
    print(" ", kml_path)
    print(" ", spiral_geo_path)
    print("—" * 64)
    return {'run_dir': run_dir, 'csv': csv_path, 'json': json_path, 'geojson': geo_path, 'kml': kml_path, 'spiral_geojson': spiral_geo_path}

# ---- Default CONFIG & CLI --------------------------------------------------

if __name__ == "__main__":
    CONFIG = {'incident': {'name': 'Shahaed-136'},
              'seed': 1337,
              'N_runs': 1500,
              'location': {
                  'release_lat': 49.274565941333805,
                  'release_lon': 11.80609661491095,
                  'release_alt_m_agl': 61.0},
              'object': {
                  'mass': 0.95,
                  'A_mean': 0.06,
                  'Cd_mean': 0.35,
                  'A_std': 0.15,
                  'Cd_std': 1.8,
                  'rho_A_Cd': 0.3,
                  'A_tumble_frac': 0.4,
                  'Cd_tumble_frac': 0.5,
                  'tumble_omega_mean': 6.0,
                  'tumble_omega_std': 3.0},
              'winds': {
                  'wind_speed_kph_mean': 10.0,
                  'wind_speed_kph_std': 2.0,
                  'wind_dir_met_from_mean': 0.0,
                  'wind_dir_met_from_std': 0.0,
                  'wind_alpha': 0.143,
                  'gust_sigma': 1.5,
                  'gust_tau': 5.0,
                  'w_mean': 0.0},
              'pre_release': {
                  'enabled': True,
                  'mode': 'glide_ratio', # Either 'time' or 'glide_ratio'
                  'time_s_mean': 10.0,
                  'time_s_std': 0.7,
                  'groundspeed_kph_mean': 120.0,
                  'groundspeed_kph_std': 8.0,
                  'heading_met_toward_mean': 100.0,
                  'heading_met_toward_std': 1.0,
                  'glide_ratio_mean': 10.0,
                  'glide_ratio_std': 0.2,
                  'release_airspeed_kph_mean': 102.48,
                  'release_airspeed_kph_std': 2.0},
              'numerics': {
                  'dt_slow': 0.02,
                  'dt_fast': 0.005,
                  'v_switch': 35.0,
                  'z_switch': 150.0,
                  'max_time': 6000.0,
                  'cell_size_m': 50.0},
              'outputs': {
                  'containment_probs': [0.5, 0.9, 0.95],
                  'use_kde_for_mode': True,
                  'kde_bandwidth_m': 150.0},
              'spiral': {'spacing_m': 50.0, 'step_m': 10.0}}
    print('Starting simulation (v6)\n')
    # Optional external config path
    json_path = "data/preset_configs/shahed_136_config.json"
    def _apply_winds_or_log(cfg, mode="current", tz="auto"):
        ok, info = apply_open_meteo_to_config_at(cfg, mode=mode, tz=tz)
        if ok:
            w = info["wind"]
            print(f"[Open-Meteo] Applied winds ({info['when']}): "
                  f"{w['speed']:.1f} km/h FROM {w['dir']:.0f}° "
                  f"(gust {w.get('gust', 0):.1f} km/h) "
                  f"[tz={w.get('tz','?')} {w.get('tz_abbr','')}]")
        else:
            print("[Open-Meteo] Wind fetch failed; using existing CONFIG winds.")
        return ok
        
    if json_path:
        try:
            cfg = load_config_json(json_path, base=CONFIG, strict=False)
            print(f"Loaded external config: {json_path}")
            _apply_winds_or_log(cfg, mode="current", tz="auto")
            run_from_config(cfg)
        except Exception as e:
            print("Failed to load external config:", e)
            print("Falling back to in-file CONFIG.")
            _apply_winds_or_log(CONFIG, mode="current", tz="auto")
            run_from_config(CONFIG)
    else:
        _apply_winds_or_log(CONFIG, mode="current", tz="auto")
        run_from_config(CONFIG)
