#!/usr/bin/env python3
"""
Stimuli Generation for Oriented Line Contours

This script generates visual stimuli for perceptual grouping experiments, specifically
oriented line contours with varying curvatures. The stimuli are generated as PNG images
with different numbers of vertices, angles, curvature levels, and orientations.

Output Structure:
- images/curv{0,1,2}/stim_nv{n}_ang{a}_curv{k}_s{s:.1f}_rot{+/-deg}.png
- stimuli_log.csv: Log of all generated stimuli with parameters

Usage:
    python stimuli_gen_oriented.py [options]

Example:
    python stimuli_gen_oriented.py --out my_stimuli --img-size 1024 --stroke 12

Modify the CONFIG section below to change default parameters, or use command-line arguments.
"""

import os
import math
import argparse
import numpy as np
from PIL import Image, ImageDraw
import csv
from datetime import datetime

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE VALUES TO CHANGE DEFAULTS
# =============================================================================

# Image parameters
IMG_SIZE = 512          # Width and height of output images (pixels)
STROKE_WIDTH = 9        # Thickness of the line (pixels)
MARGIN = 40             # Margin from image edges (pixels)
ANTI_ALIAS = 4          # Anti-aliasing factor (higher = smoother, but slower)

# Geometry parameters
SEGMENT_LENGTH = 1.0    # Length of each straight segment in the polyline

# Curvature parameters
SMOOTH_STRENGTH_1 = 0.8  # Smoothing strength
SMOOTH_STRENGTH_2 = 1.4  # Smoothing strength

# Stimuli variations
VERTICES = (1, 2, 3)                    # Number of vertices (turns)
ANGLES = (45, 90, 135)                  # Turn angles in degrees
CURVATURE_LEVELS = (0, 1, 2)            # 0=straight, 1=moderate curve, 2=strong curve
ORIENTATIONS = (0, 45, 90, 135, 180, -45, -90, -135)  # Rotation angles in degrees

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_dir(p: str):
    """Create directory if it doesn't exist."""
    os.makedirs(p, exist_ok=True)

def format_rot_label(deg: int) -> str:
    """Return string like '+045', '-090', '+000', '+135', '+180'."""
    sign = '+' if deg >= 0 else '-'
    return f"{sign}{abs(int(deg)):03d}"

# =============================================================================
# GEOMETRY BUILDING FUNCTIONS
# =============================================================================

def hermite_sample(p0, m0, p1, m1, n=80):
    """
    Sample points along a Hermite curve defined by positions p0,p1 and tangents m0,m1.

    Args:
        p0, p1: Start and end positions (numpy arrays)
        m0, m1: Tangent vectors at start and end
        n: Number of sample points

    Returns:
        Array of sampled points along the curve
    """
    t = np.linspace(0.0, 1.0, n)[:, None]
    h00 = (2*t**3 - 3*t**2 + 1)
    h10 = (t**3 - 2*t**2 + t)
    h01 = (-2*t**3 + 3*t**2)
    h11 = (t**3 - t**2)
    return h00*p0 + h10*m0 + h01*p1 + h11*m1

def build_polyline_by_turns(n_vertices: int, angle_deg: float, seg_len: float = 1.0) -> np.ndarray:
    """
    Build a polyline by alternating left/right turns.

    Args:
        n_vertices: Number of vertices (determines number of turns)
        angle_deg: Turn angle in degrees
        seg_len: Length of each segment

    Returns:
        Array of 2D points in y-up coordinates
    """
    assert n_vertices >= 1
    n_segments = 2 * n_vertices
    pts = [np.array([0.0, 0.0], float)]
    heading = 0.0
    turn = math.radians(angle_deg)
    sign = +1  # start by turning up
    for s in range(n_segments):
        dx = seg_len * math.cos(heading)
        dy = seg_len * math.sin(heading)
        pts.append(pts[-1] + np.array([dx, dy], float))
        if s < n_segments - 1:
            heading += sign * turn
            sign *= -1
    return np.vstack(pts)  # y-up

def fit_to_frame_general(poly_xy_up: np.ndarray, W: int, H: int, margin: int, stroke: int,
                         mode: str = "width", edge_to_edge: bool = True) -> np.ndarray:
    """
    Map y-up polyline to image coordinates, rotate chord horizontal, then scale.

    Args:
        poly_xy_up: Polyline points in y-up coordinates
        W, H: Image dimensions
        margin: Margin from edges
        stroke: Line thickness
        mode: Scaling mode - "width" (fit width) or "min" (fit both dimensions)
        edge_to_edge: If True, normalize X to touch left/right borders

    Returns:
        Points in image coordinates (y-down)
    """
    p0, pN = poly_xy_up[0], poly_xy_up[-1]
    P = poly_xy_up - (p0 + pN) / 2.0

    # rotate chord to horizontal in y-up
    v = pN - p0
    ang = math.atan2(v[1], v[0])
    R = np.array([[math.cos(-ang), -math.sin(-ang)],
                  [math.sin(-ang),  math.cos(-ang)]], float)
    P = (R @ P.T).T

    # spans (before y-flip)
    span_x = max(1e-6, P[:,0].max() - P[:,0].min())
    span_y = max(1e-6, P[:,1].max() - P[:,1].min())

    width_lim  = max(1.0, (W - 2*margin - stroke))
    height_lim = max(1.0, (H - 2*margin - stroke))

    if mode == "min":
        s = min(width_lim / span_x, height_lim / span_y)
    else:  # "width"
        s = width_lim / span_x

    P *= s

    # y-up -> image coords
    P[:,1] = -P[:,1]
    P[:,0] += W/2.0

    if edge_to_edge:
        # normalize X to exactly touch borders
        x_min, x_max = P[:,0].min(), P[:,0].max()
        x_rng = max(1e-6, x_max - x_min)
        P[:,0] = (P[:,0] - x_min) / x_rng * (W - stroke) + stroke/2.0

    return P

# =============================================================================
# SMOOTHING FUNCTIONS
# =============================================================================

def build_smooth_path_free_ends(ctrl: np.ndarray, smooth_strength: float,
                                sample_density: float = 0.30) -> np.ndarray:
    """
    Create smooth path using Hermite interpolation with free-end tangents.

    Args:
        ctrl: Control points
        smooth_strength: Global scale for tangent vectors
        sample_density: Points per unit length

    Returns:
        Smoothed path points
    """
    P = np.asarray(ctrl, float)
    M = np.zeros_like(P)

    if len(P) == 2:
        v = P[1] - P[0]
        M[0]  = smooth_strength * v
        M[-1] = smooth_strength * v
    else:
        M[0]  = smooth_strength * (P[1] - P[0])
        M[-1] = smooth_strength * (P[-1] - P[-2])

        for i in range(1, len(P) - 1):
            base_tan = 0.5 * (P[i+1] - P[i-1])
            M[i] = smooth_strength * base_tan

    out = [P[0]]
    for i in range(len(P)-1):
        seg_len = float(np.linalg.norm(P[i+1] - P[i]))
        samples = int(max(40, min(140, seg_len * sample_density)))  # cap 40–140
        seg = hermite_sample(P[i], M[i], P[i+1], M[i+1], n=samples)
        out.append(seg[1:])
    return np.vstack(out)

# =============================================================================
# CENTERING FUNCTIONS
# =============================================================================

def center_by_internal_vertices(ctrl_img: np.ndarray, path: np.ndarray, H: int) -> np.ndarray:
    """
    Center path vertically based on internal control vertices.

    Args:
        ctrl_img: Control points in image coordinates
        path: Path to center
        H: Image height

    Returns:
        Vertically centered path
    """
    if len(ctrl_img) <= 2:
        y_center = float((path[:,1].min() + path[:,1].max()) / 2.0)
    else:
        ys = ctrl_img[1:-1, 1]
        y_center = float((ys.min() + ys.max()) / 2.0)
    dy = (H / 2.0) - y_center
    P = np.asarray(path, float).copy()
    P[:, 1] += dy
    return P

def center_by_length_weighted_mean(path: np.ndarray, H: int, clip: float) -> np.ndarray:
    """
    Center path vertically using length-weighted mean of segment midpoints.

    Args:
        path: Path points
        H: Image height
        clip: Fraction to clip from ends (0.0 to 1.0)

    Returns:
        Vertically centered path
    """
    P = np.asarray(path, float).copy()
    d = np.linalg.norm(P[1:] - P[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = s[-1]
    if L <= 1e-9:
        y_ctr = float((P[:,1].min() + P[:,1].max()) / 2.0)
        P[:,1] += (H/2.0 - y_ctr)
        return P

    u = s / L
    u_mid = 0.5*(u[:-1] + u[1:])
    seg_mask = (u_mid >= clip) & (u_mid <= 1.0 - clip)
    if not np.any(seg_mask):
        seg_mask = slice(None)

    y_seg = 0.5*(P[:-1,1] + P[1:,1])
    y_mean = float(np.average(y_seg[seg_mask], weights=d[seg_mask]))
    P[:, 1] += (H / 2.0 - y_mean)
    return P

# =============================================================================
# ORIENTATION FUNCTIONS
# =============================================================================

def rotate_path_about_center(path_img: np.ndarray, W: int, H: int, deg: float) -> np.ndarray:
    """
    Rotate path around image center by specified degrees.

    Args:
        path_img: Path in image coordinates
        W, H: Image dimensions
        deg: Rotation angle in degrees

    Returns:
        Rotated path
    """
    theta = math.radians(deg)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]], float)
    P = np.asarray(path_img, float).copy()
    P -= np.array([W/2.0, H/2.0])
    P = (R @ P.T).T
    P += np.array([W/2.0, H/2.0])
    return P

def fit_inside_canvas_imgcoords(path_img: np.ndarray, W: int, H: int, margin: int, stroke: int) -> np.ndarray:
    """
    Scale and center path to fit within canvas margins.

    Args:
        path_img: Path in image coordinates
        W, H: Image dimensions
        margin: Margin from edges
        stroke: Line thickness

    Returns:
        Scaled and centered path
    """
    P = np.asarray(path_img, float).copy()
    # Shift to center for uniform scaling
    cx, cy = W/2.0, H/2.0
    P -= np.array([cx, cy])

    # Current spans
    span_x = max(1e-9, P[:,0].max() - P[:,0].min())
    span_y = max(1e-9, P[:,1].max() - P[:,1].min())

    width_lim  = max(1.0, (W - 2*margin - stroke))
    height_lim = max(1.0, (H - 2*margin - stroke))

    s = min(width_lim / span_x, height_lim / span_y)
    P *= s

    # Back to center
    P += np.array([cx, cy])
    return P

# =============================================================================
# RENDERING FUNCTIONS
# =============================================================================

def render_aa(path: np.ndarray, W: int, H: int, stroke: int,
              bg=(0, 0, 0, 0), fg=(0, 0, 0, 255), aa: int = 2) -> Image.Image:
    """
    Render antialiased line path to PIL Image.

    Args:
        path: Path points in image coordinates
        W, H: Image dimensions
        stroke: Line thickness
        bg: Background color (RGBA)
        fg: Foreground color (RGBA)
        aa: Anti-aliasing factor

    Returns:
        PIL Image with rendered path
    """
    W2, H2 = W * aa, H * aa
    pts2 = np.asarray(path, float) * aa

    img = Image.new("RGBA", (W2, H2), bg)  # transparent
    drw = ImageDraw.Draw(img)
    drw.line([tuple(p) for p in pts2], fill=fg, width=stroke * aa, joint="curve")

    if aa > 1:
        img = img.resize((W, H), Image.LANCZOS)
    return img

# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_set_free_oriented(out_root: str,
                               img_size: int = IMG_SIZE,
                               stroke: int = STROKE_WIDTH,
                               margin: int = MARGIN,
                               vertices: tuple = VERTICES,
                               angles: tuple = ANGLES,
                               curvature_levels: tuple = CURVATURE_LEVELS,
                               orientations: tuple = ORIENTATIONS,
                               s1: float = SMOOTH_STRENGTH_1,
                               s2: float = SMOOTH_STRENGTH_2,
                               aa: int = ANTI_ALIAS,
                               seg_len: float = SEGMENT_LENGTH):
    """
    Generate complete set of oriented stimuli with varying parameters.

    Args:
        out_root: Output directory root
        img_size: Image size (width and height)
        stroke: Line thickness
        margin: Margin from edges
        vertices: Tuple of vertex counts
        angles: Tuple of turn angles
        curvature_levels: Tuple of curvature levels
        orientations: Tuple of rotation angles
        s1, s2: Smooth strengths for curvature levels 1 and 2
        aa: Anti-aliasing factor
        seg_len: Segment length for geometry

    Creates:
        - Directory structure with images/curv{k}/ subdirs
        - PNG images for each parameter combination
        - stimuli_log.csv with generation details
    """
    ensure_dir(out_root)
    for k in curvature_levels:
        ensure_dir(os.path.join(out_root, f"images"))

    # CSV log
    log_path = os.path.join(out_root, "stimuli_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "filename", "out_path",
            "vertices", "angle_deg", "curvature_level", "orientation_deg",
            "smooth_strength", "img_size", "stroke", "margin"
        ])

        W = H = img_size

        for n in vertices:
            for a in angles:
                ang_label = int(a)  # Use input angle directly
                if a == 45:
                    clip_for_n1 = 0.20
                elif a == 90:
                    clip_for_n1 = 0.25
                else:
                    clip_for_n1 = 0.30

                ctrl_up = build_polyline_by_turns(n_vertices=n, angle_deg=180-a, seg_len=seg_len)
                ctrl_oriented = ctrl_up.copy()
                ctrl_oriented[:, 1] *= -1.0

                is_narrow_v = (n == 1 and a == 135)
                fit_mode = "min" if is_narrow_v else "width"
                edge2edge = False if is_narrow_v else True
                ctrl_img = fit_to_frame_general(ctrl_oriented, W=W, H=H, margin=margin,
                                                stroke=stroke, mode=fit_mode, edge_to_edge=edge2edge)

                for k in curvature_levels:
                    if k == 0:
                        base_path = ctrl_img
                        s_used = 0.0
                    else:
                        if k == 1:
                            s = s1
                        else:    
                            s = s2
                        s_used = float(s)
                        base_path = build_smooth_path_free_ends(ctrl_img, smooth_strength=s_used, sample_density=0.30)

                    if n == 1:
                        # Center all paths based on control points to ensure consistent positioning
                        ctrl_centered = center_by_length_weighted_mean(ctrl_img, H=H, clip=clip_for_n1)
                        dy = ctrl_centered[0, 1] - ctrl_img[0, 1]  # vertical shift from control points
                        base_path = base_path.copy()
                        base_path[:, 1] += dy
                    else:
                        base_path = center_by_internal_vertices(ctrl_img, base_path, H=H)

                    for ori in orientations:
                        path_rot = rotate_path_about_center(base_path, W=W, H=H, deg=float(ori))
                        path_fit = fit_inside_canvas_imgcoords(path_rot, W=W, H=H, margin=margin, stroke=stroke)

                        img = render_aa(path_fit, W=W, H=H, stroke=stroke, aa=aa)
                        rot_lbl = format_rot_label(int(ori))
                        fname = f"stim_nv{n}_ang{ang_label}_s{s_used:.1f}_rot{rot_lbl}.png"
                        out_path = os.path.join(out_root, f"images", fname)
                        img.save(out_path)

                        # --- Log entry ---
                        writer.writerow([
                            datetime.now().isoformat(timespec="seconds"),
                            fname, out_path,
                            n, a, ang_label, k, ori,
                            s_used, img_size, stroke, margin
                        ])

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Generate oriented line contour stimuli with varying curvatures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stimuli_gen_oriented.py --out my_experiment
  python stimuli_gen_oriented.py --img-size 1024 --stroke 12 --orientations "0,90,180"
  python stimuli_gen_oriented.py --s1 0.6 --s2 1.1

Modify the CONFIG section at the top of the script to change defaults.
        """
    )
    p.add_argument("--out", type=str, default="MainExp",
                   help="Output directory (default: MainExp)")
    p.add_argument("--img-size", type=int, default=IMG_SIZE,
                   help=f"Image size in pixels (default: {IMG_SIZE})")
    p.add_argument("--stroke", type=int, default=STROKE_WIDTH,
                   help=f"Line thickness in pixels (default: {STROKE_WIDTH})")
    p.add_argument("--margin", type=int, default=MARGIN,
                   help=f"Margin from image edges in pixels (default: {MARGIN})")
    p.add_argument("--aa", type=int, default=ANTI_ALIAS,
                   help=f"Anti-aliasing factor (default: {ANTI_ALIAS})")
    p.add_argument("--seg-len", type=float, default=SEGMENT_LENGTH,
                   help=f"Segment length for geometry (default: {SEGMENT_LENGTH})")
    p.add_argument("--s1", type=float, default=SMOOTH_STRENGTH_1,
                   help=f"Smooth strength for curvature level 1 (default: {SMOOTH_STRENGTH_1})")
    p.add_argument("--s2", type=float, default=SMOOTH_STRENGTH_2,
                   help=f"Smooth strength for curvature level 2 (default: {SMOOTH_STRENGTH_2})")
    p.add_argument("--orientations", type=str, default=",".join(map(str, ORIENTATIONS)),
                   help=f"Comma-separated rotation angles (default: {','.join(map(str, ORIENTATIONS))})")
    return p.parse_args()

def parse_orientations(s: str):
    """Parse comma-separated orientation string into tuple of ints."""
    vals = []
    for tok in s.split(','):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    return tuple(vals)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    orientations = parse_orientations(args.orientations)

    print("Generating stimuli with parameters:")
    print(f"  Output directory: {args.out}")
    print(f"  Image size: {args.img_size}x{args.img_size}")
    print(f"  Stroke width: {args.stroke}")
    print(f"  Margin: {args.margin}")
    print(f"  Anti-aliasing: {args.aa}x")
    print(f"  Vertices: {VERTICES}")
    print(f"  Angles: {ANGLES}")
    print(f"  Curvature levels: {CURVATURE_LEVELS}")
    print(f"  Orientations: {orientations}")
    print(f"  Smooth strengths: s1={args.s1}, s2={args.s2}")
    print()

    generate_set_free_oriented(out_root=args.out,
                               img_size=args.img_size,
                               stroke=args.stroke,
                               margin=args.margin,
                               s1=args.s1,
                               s2=args.s2,
                               aa=args.aa,
                               seg_len=args.seg_len,
                               orientations=orientations)

    print(f"Stimuli generation complete. Check {args.out}/ for results.")

