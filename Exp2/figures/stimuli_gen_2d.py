# circle_range_smoothness.py
import os, math, itertools, pathlib, csv
import numpy as np
from numpy.random import default_rng
from PIL import Image, ImageDraw

# Random seed for reproducibility
RANDOM_SEED = 10

# ---------------- Utilities ----------------
def ensure_closed_unique(P):
    """Return open polyline (first != last)."""
    return P[:-1] if np.allclose(P[0], P[-1]) else P

def close_loop(P):
    """Return explicitly closed polyline (last == first)."""
    return P if np.allclose(P[0], P[-1]) else np.vstack([P, P[0]])

def roll_start(P, rng):
    """Rotate start index of an open closed-path polyline to avoid any fixed seam/start bias."""
    Q = ensure_closed_unique(P)
    k = int(rng.integers(0, len(Q)))
    return np.vstack([Q[k:], Q[:k]])

# ---------------- Base generator: circle + jitter ----------------
def circle_polygon(cx, cy, nvertices, rng, R=100, range_px=0, angle_jitter_deg=2.0):
    """
    Generate a circle-like polygon by sampling N angles and adding per-vertex radial noise.

    Parameters
    ----------
    R : base radius (gets normalized later by center+scale, but controls initial geometry)
    range_px : radial perturbation magnitude (uniform in [-range_px, +range_px])
    angle_jitter_deg : jitter angles a bit to avoid "too regular" vertex spacing

    Returns
    -------
    V : (N,2) array of vertices
    """
    base_angles = np.linspace(0, 2*np.pi, nvertices, endpoint=False)

    if angle_jitter_deg and angle_jitter_deg > 0:
        jitter = rng.uniform(-np.deg2rad(angle_jitter_deg),
                             +np.deg2rad(angle_jitter_deg),
                             size=nvertices)
        angles = base_angles + jitter
    else:
        angles = base_angles

    if range_px > 0:
        # proportional band width
        w = 0.2 * range_px

        # randomly choose inward or outward for each vertex but avoid
        # runs of the same sign longer than 3 (i.e., if 3 consecutive are same, force opposite)
        sign = np.empty(nvertices, dtype=float)
        if nvertices <= 3:
            sign[:] = rng.choice([-1.0, 1.0], size=nvertices)
        else:
            # seed the first three randomly
            for j in range(3):
                sign[j] = rng.choice([-1.0, 1.0])
            for i in range(3, nvertices):
                # if the previous 3 signs are identical, force the opposite
                if sign[i-3] == sign[i-2] == sign[i-1]:
                    sign[i] = -sign[i-1]
                else:
                    sign[i] = rng.choice([-1.0, 1.0])

        # sample magnitude near ±range_px
        mag = rng.uniform(range_px - w, range_px + w, size=nvertices)

        eps = sign * mag
    else:
        eps = np.zeros(nvertices)

    radii = R + eps
    x = cx + radii * np.cos(angles)
    y = cy + radii * np.sin(angles)

    V = np.c_[x, y].astype(float)

    V = V[::-1]
    return V

# ---------------- Resampling ----------------
def densify_polyline(P, n=3000):
    """
    Resample a closed loop to n points along arc-length.
    Returns an OPEN polyline (no duplicated endpoint).
    """
    P = ensure_closed_unique(P)     # open, length N
    Pc = close_loop(P)             # closed, length N+1

    seg = np.sqrt(((Pc[1:] - Pc[:-1]) ** 2).sum(1))  # length N
    t = np.r_[0.0, np.cumsum(seg)]                   # length N+1
    L = t[-1]

    if L == 0:
        return P.copy()

    t_new = np.linspace(0, L, n, endpoint=False)

    x = np.interp(t_new, t, Pc[:, 0])
    y = np.interp(t_new, t, Pc[:, 1])

    return np.c_[x, y]

# ---------------- Seam-safe smoothing ----------------
def moving_average_closed(P, win=50, passes=2):
    """
    Circular moving-average smoothing on an open polyline that represents a closed loop.
    Returns open polyline (no duplicated endpoint).
    """
    Q = ensure_closed_unique(P)
    if win % 2 == 0:
        win += 1
    h = win // 2
    k = np.ones(win, dtype=float) / win

    for _ in range(passes):
        xpad = np.r_[Q[-h:, 0], Q[:, 0], Q[:h, 0]]
        ypad = np.r_[Q[-h:, 1], Q[:, 1], Q[:h, 1]]
        x = np.convolve(xpad, k, mode="valid")
        y = np.convolve(ypad, k, mode="valid")
        Q = np.c_[x, y]

    return Q

# ---------------- Geometry metrics ----------------
def poly_perimeter(P):
    Pc = close_loop(P)
    return float(np.sqrt(((np.roll(Pc, -1, axis=0) - Pc) ** 2).sum(1)).sum())

def poly_area(P):
    Pc = close_loop(P)
    x, y = Pc[:, 0], Pc[:, 1]
    return float(0.5 * np.abs(np.sum(x*np.roll(y, -1) - y*np.roll(x, -1))))

def curvature_energy(P):
    """
    Discrete curvature energy integral approx: sum kappa^2 ds
    using central differences on an open polyline with circular wrap.
    """
    Q = ensure_closed_unique(P)
    Xp = np.roll(Q, -1, axis=0)
    Xm = np.roll(Q,  1, axis=0)
    d1 = (Xp - Xm) * 0.5
    d2 = (Xp - 2.0*Q + Xm)
    num = d1[:, 0]*d2[:, 1] - d1[:, 1]*d2[:, 0]
    den = (d1[:, 0]**2 + d1[:, 1]**2)**1.5 + 1e-12
    kappa = num / den
    ds = np.sqrt(((Xp - Q)**2).sum(1))
    CE = float(np.sum((kappa**2) * ds))
    return CE, float(np.mean(np.abs(kappa))), float(np.max(np.abs(kappa)))

# ---------------- Transforms & drawing ----------------
def center_and_scale_to_canvas(P, canvas_size=1200, margin=60):
    P = P - P.mean(axis=0, keepdims=True)
    max_extent = np.max(np.abs(P))
    if max_extent > 0:
        scale = (canvas_size/2 - margin) / max_extent
        P = P * scale
    P = P + np.array([canvas_size/2, canvas_size/2])
    return P

def save_path_as_png(P, out_path, canvas_size=1200, line_width=8, line_color=(0, 0, 0, 255)):
    """
    Draw outline-only stimulus with transparent background.
    """
    img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    Q = close_loop(P)
    pts = [tuple(map(float, xy)) for xy in Q]
    draw.line(pts, fill=line_color, width=line_width, joint="curve")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, "PNG")

# ---------------- HTML Grid Generation ----------------
def generate_html_grid(root_dir, Ns, RANGES, curv_level=0):
    """
    Generate an HTML grid showing all N × RANGES combinations for a given curvature level.
    """
    html_lines = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'/>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>",
        f"<title>curv{curv_level} grid</title>",
        "<style>",
        "    body { font-family: Arial, sans-serif; padding: 18px; }",
        "    table { border-collapse: collapse; }",
        "    th, td {",
        "        border: 1px solid #ccc;",
        "        padding: 8px;",
        "        text-align: center;",
        "        vertical-align: top;",
        "    }",
        "    th {",
        "        position: sticky;",
        "        top: 0;",
        "        background: #f7f7f7;",
        "        z-index: 1;",
        "    }",
        "    td.rowhdr {",
        "        position: sticky;",
        "        left: 0;",
        "        background: #f7f7f7;",
        "        z-index: 1;",
        "        font-weight: bold;",
        "    }",
        "    img.thumb {",
        "        width: 180px;",
        "        height: auto;",
        "    }",
        "    .missing {",
        "        color: #999;",
        "        font-size: 12px;",
        "        padding-top: 12px;",
        "    }",
        "</style>",
        "</head>",
        "<body>",
        "<table>",
    ]

    # Header row
    html_lines.append("<tr>")
    html_lines.append("<th>N \\ range</th>")
    for rng in RANGES:
        html_lines.append(f"<th>{rng}</th>")
    html_lines.append("</tr>")

    # Data rows
    for N in Ns:
        html_lines.append("<tr>")
        html_lines.append(f"<td class='rowhdr'>{N}</td>")
        for rng in RANGES:
            fname = f"circle_rng{rng:02d}_N{N}_curv{curv_level}_seed{RANDOM_SEED}.png"
            fpath = os.path.join("images", fname)
            img_full_path = os.path.join(root_dir, fpath)
            
            if os.path.exists(img_full_path):
                html_lines.append(f"<td><a href='{fpath}' target='_blank'><img class='thumb' src='{fpath}'/></a></td>")
            else:
                html_lines.append(f"<td><span class='missing'>N/A</span></td>")
        html_lines.append("</tr>")

    html_lines.extend([
        "</table>",
        "</body></html>",
    ])

    html_path = os.path.join(root_dir, f"curv{curv_level}_seed{RANDOM_SEED}_grid.html")
    with open(html_path, "w") as f:
        f.write("\n".join(html_lines))
    print(f"HTML grid → {os.path.abspath(html_path)}")


def generate_html_all_grid(root_dir, Ns, RANGES, curv_levels):
    """
    Generate a single HTML file that contains one table per curvature level.
    """
    html_lines = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'/>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>",
        "<title>All curv grids</title>",
        "<style>",
        "    body { font-family: Arial, sans-serif; padding: 18px; }",
        "    table { border-collapse: collapse; margin-bottom: 36px; }",
        "    th, td {",
        "        border: 1px solid #ccc;",
        "        padding: 8px;",
        "        text-align: center;",
        "        vertical-align: top;",
        "    }",
        "    th {",
        "        position: sticky;",
        "        top: 0;",
        "        background: #f7f7f7;",
        "        z-index: 1;",
        "    }",
        "    td.rowhdr {",
        "        position: sticky;",
        "        left: 0;",
        "        background: #f7f7f7;",
        "        z-index: 1;",
        "        font-weight: bold;",
        "    }",
        "    img.thumb {",
        "        width: 160px;",
        "        height: auto;",
        "        display: block;",
        "        margin: 0 auto 6px;",
        "    }",
        "    .curv-title { font-size: 18px; margin: 8px 0 6px; }",
        "    .missing {",
        "        color: #999;",
        "        font-size: 12px;",
        "        padding-top: 12px;",
        "    }",
        "</style>",
        "</head>",
        "<body>",
    ]

    for curv in curv_levels:
        html_lines.append(f"<div class='curv-title'>curv{curv}</div>")
        html_lines.append("<table>")
        # header
        html_lines.append("<tr>")
        html_lines.append("<th>N \\ range</th>")
        for rng in RANGES:
            html_lines.append(f"<th>{rng}</th>")
        html_lines.append("</tr>")

        for N in Ns:
            html_lines.append("<tr>")
            html_lines.append(f"<td class='rowhdr'>{N}</td>")
            for rng in RANGES:
                fname = f"circle_rng{rng:02d}_N{N}_curv{curv}_seed{RANDOM_SEED}.png"
                fpath = os.path.join("images", fname)
                img_full_path = os.path.join(root_dir, fpath)
                if os.path.exists(img_full_path):
                    html_lines.append(f"<td><a href='{fpath}' target='_blank'><img class='thumb' src='{fpath}'/></a></td>")
                else:
                    html_lines.append(f"<td><span class='missing'>N/A</span></td>")
            html_lines.append("</tr>")

        html_lines.append("</table>")

    html_lines.extend([
        "</body></html>",
    ])

    html_path = os.path.join(root_dir, f"all_curv_seed{RANDOM_SEED}_grid.html")
    with open(html_path, "w") as f:
        f.write("\n".join(html_lines))
    print(f"HTML grid (all curv) → {os.path.abspath(html_path)}")

# ---------------- Main ----------------
def main():
    rng = default_rng(RANDOM_SEED) 

    root_dir = os.path.join(os.path.dirname(__file__), "MainExp")
    pathlib.Path(root_dir).mkdir(parents=True, exist_ok=True)

    # Factors to KEEP
    Ns = [9, 18, 27]             # vertex counts
    curv_levels = [0, 1, 2]   # curv0/curv1/curv2

    RANGES = [12, 25, 45] 
    BASE_R = 100              # base radius
    ANGLE_JITTER_DEG = 2.0    # keep fixed (set to 0.0 for perfectly even angles)

    # stimuli per cell (N × range)
    N_STIM_PER_CELL = 1

    # Smoothing levels
    MOVAVG = {
        0: dict(win=None, passes=0),
        1: dict(win=44,  passes=2),
        2: dict(win=80,  passes=5),
    }

    # output: create a single images folder for all curvature levels
    images_dir = os.path.join(root_dir, "images")
    pathlib.Path(images_dir).mkdir(parents=True, exist_ok=True)

    # CSV log
    csv_path = os.path.join(root_dir, f"stimulus_log_seed{RANDOM_SEED}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "fname",
            "shape_family",
            "range_px",
            "N",
            "rep",
            "curv_level",
            "base_R",
            "angle_jitter_deg",
            "win",
            "passes",
            "perimeter_px",
            "area_px2",
            "curv_energy",
            "mean_abs_kappa",
            "max_abs_kappa",
            "canvas_px",
            "margin_px",
            "line_width_px",
            "seed_like",
            "engine"
        ])

        count = 0

        for N, range_px in itertools.product(Ns, RANGES):
            for rep in range(1, N_STIM_PER_CELL + 1):

                # Create a per-stimulus RNG derived from the global seed so that
                # this stimulus is reproducible regardless of other iterations.
                ss = np.random.SeedSequence([RANDOM_SEED, int(N), int(range_px), int(rep)])
                local_rng = default_rng(ss)

                # 1) One base polygon (circle + jitter)
                V0 = circle_polygon(
                    cx=0, cy=0,
                    nvertices=N,
                    rng=local_rng,
                    R=BASE_R,
                    range_px=range_px,
                    angle_jitter_deg=ANGLE_JITTER_DEG
                )

                # 2) Densify once; all curvature levels derive from this same base
                base = densify_polyline(V0, n=3000)  # open polyline

                # 3) Build each curvature level from same base
                for curv in curv_levels:
                    if curv == 0:
                        curve = ensure_closed_unique(base)
                        win, passes = 0, 0
                    else:
                        params = MOVAVG[curv]
                        win, passes = params["win"], params["passes"]
                        curve = moving_average_closed(base, win=win, passes=passes)

                    # randomize start index (avoids any consistent startpoint artifact)
                    # use a per-curvature RNG so the roll is reproducible and
                    # unique for each `curv` value
                    ss2 = np.random.SeedSequence([RANDOM_SEED, int(N), int(range_px), int(rep), int(curv)])
                    roll_rng = default_rng(ss2)
                    curve = roll_start(curve, roll_rng)

                    # center+scale for final render
                    curve = center_and_scale_to_canvas(curve, canvas_size=1200, margin=60)

                    # metrics
                    Ppx = poly_perimeter(curve)
                    Apx = poly_area(curve)
                    CE, mk, kmax = curvature_energy(curve)

                    # save into single images directory
                    fname = f"circle_rng{range_px:02d}_N{N}_curv{curv}_seed{RANDOM_SEED}.png"
                    out_path = os.path.join(images_dir, fname)
                    save_path_as_png(curve, out_path, canvas_size=1200, line_width=14)

                    # log
                    w.writerow([
                        os.path.join("images", fname),
                        "circle",
                        range_px,
                        N,
                        rep,
                        f"curv{curv}",
                        BASE_R,
                        ANGLE_JITTER_DEG,
                        win,
                        passes,
                        f"{Ppx:.3f}",
                        f"{Apx:.3f}",
                        f"{CE:.6f}",
                        f"{mk:.6f}",
                        f"{kmax:.6f}",
                        1200,
                        60,
                        8,
                        RANDOM_SEED,
                        "circle+moving_avg"
                    ])
                    count += 1

    print(f"Saved {count} stimuli → {os.path.abspath(root_dir)}")
    print(f"CSV log → {os.path.abspath(csv_path)}")

    # Generate combined HTML showing all curvature levels
    generate_html_all_grid(root_dir, Ns, RANGES, curv_levels)

if __name__ == "__main__":
    main()
