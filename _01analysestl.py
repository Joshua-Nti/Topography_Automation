#!/usr/bin/env python3
"""
_04analysestl.py
----------------
Automatic cut detection for nonplanar printing.

Generates `cuts.txt`:
    <segment_index> <transform_flag> <segment_top_z_mm>

Example:
    1 0 15.0000
    2 1 32.0000
    3 0 48.5000
    4 1 TOP
"""

import sys
import os
import numpy as np
import trimesh
from shapely.geometry import MultiPoint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# zentrale Parameter aus Config
from _10config import ANALYSE_STL


# --------------------------
# Geometry + helper functions
# --------------------------

def face_normals_and_centroids(mesh_t):
    normals = mesh_t.face_normals
    centroids = mesh_t.vertices[mesh_t.faces].mean(axis=1)
    return normals, centroids


def select_downward_faces(normals, angle_deg):
    cos_lim = np.cos(np.deg2rad(angle_deg))
    return normals[:, 2] <= -cos_lim


def build_adjacency_by_vertex(faces, mask):
    """
    Cluster all candidate faces (mask == True) by shared vertex adjacency.
    """
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return []

    vert_to_faces = {}
    for fi in idx:
        for v in faces[fi]:
            vert_to_faces.setdefault(v, []).append(fi)

    neighbors = {fi: set() for fi in idx}
    for flist in vert_to_faces.values():
        for a in flist:
            for b in flist:
                if a != b:
                    neighbors[a].add(b)

    clusters = []
    visited = set()
    for fi in idx:
        if fi in visited:
            continue
        comp = []
        stack = [fi]
        visited.add(fi)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in neighbors[cur]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        clusters.append(comp)
    return clusters


def cluster_span_and_z(mesh_t, faces_idx):
    verts = mesh_t.vertices[mesh_t.faces[faces_idx].reshape(-1)]
    xs, ys, zs = verts[:, 0], verts[:, 1], verts[:, 2]
    min_z = float(np.min(zs))
    pts_xy = np.column_stack([xs, ys])
    mp = MultiPoint(list(map(tuple, pts_xy)))
    try:
        hull = mp.convex_hull
        if hull.geom_type == "Polygon":
            hull_pts = np.array(hull.exterior.coords)
        else:
            hull_pts = np.array(hull.coords)
    except Exception:
        hull_pts = pts_xy
    if len(hull_pts) <= 1:
        span = 0.0
    else:
        diffs = hull_pts[None, :, :] - hull_pts[:, None, :]
        span = float(np.sqrt(np.max(np.sum(diffs**2, axis=2))))
    return min_z, span


def classify_cluster(span, threshold):
    if span >= threshold:
        return True, "large_bridge_or_overhang"
    else:
        return False, "tiny_bridge"


def propose_cut_zs(mesh_t, clusters, span_threshold, cut_below_mm, zmin_global):
    cuts, debug = [], []
    for comp in clusters:
        min_z, span = cluster_span_and_z(mesh_t, comp)
        wants_cut, reason = classify_cluster(span, span_threshold)
        if wants_cut:
            cz = max(min_z - cut_below_mm, zmin_global)
            cuts.append(cz)
            debug.append(dict(span=span, min_z_cluster=min_z, cut_z=cz, reason=reason))
        else:
            debug.append(dict(span=span, min_z_cluster=min_z, cut_z=None, reason=reason))
    return cuts, debug


def enforce_min_spacing(cuts, min_spacing, zmin):
    if not cuts:
        return []
    cuts = sorted(cuts)
    kept = [cuts[0]]
    for c in cuts[1:]:
        if c - kept[-1] >= min_spacing:
            kept.append(c)
    return [z for z in kept if z > zmin + 1e-6]


# --------------------------
# Segment flag + writing
# --------------------------

def build_segment_flags(debug_info, cuts, z_top):
    if not cuts:
        return [0]

    cuts_sorted = sorted(cuts)
    num_segments = len(cuts_sorted) + 1
    seg_flags = [0] * num_segments

    for info in debug_info:
        if info.get("reason") != "large_bridge_or_overhang":
            continue
        cz = info.get("cut_z")
        if cz is None:
            continue
        idx_right = np.searchsorted(cuts_sorted, cz, side="right")
        seg_index = max(1, idx_right)
        if seg_index < num_segments:
            seg_flags[seg_index] = 1
        else:
            seg_flags[-1] = 1

    return seg_flags


def write_segment_file(path_out, cuts, flags, z_top, use_top_label=True):
    cuts_sorted = sorted(cuts)
    segment_tops = cuts_sorted + [z_top]
    with open(path_out, "w") as f:
        for i, (flag, topz) in enumerate(zip(flags, segment_tops), start=1):
            if i == len(segment_tops) and use_top_label:
                f.write(f"{i} {flag} TOP\n")
            else:
                f.write(f"{i} {flag} {topz:.4f}\n")
    print(f"[analyseSTL] wrote {len(segment_tops)} segments to {path_out}")
    print("  format: <segment_index> <transform_flag> <segment_top_z_mm>")


# --------------------------
# Visualization
# --------------------------

def visualize_debug(mesh_t, clusters, cuts, span_threshold):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    base_pc = Poly3DCollection(mesh_t.triangles, alpha=0.15, facecolor=(0.7, 0.7, 0.7))
    ax.add_collection3d(base_pc)

    for comp in clusters:
        tri = mesh_t.triangles[comp]
        _, span = cluster_span_and_z(mesh_t, comp)
        color = (1, 0, 0) if span >= span_threshold else (0, 1, 0)
        pc = Poly3DCollection(tri, alpha=0.6, facecolor=color, linewidths=0.1)
        ax.add_collection3d(pc)

    if cuts:
        (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh_t.bounds
        X, Y = [xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax]
        for cz in cuts:
            Z = [cz] * 4
            plane = [list(zip(X, Y, Z))]
            pc = Poly3DCollection(plane, alpha=0.3, facecolor=(0, 0, 1))
            ax.add_collection3d(pc)

    mins, maxs = mesh_t.bounds
    ax.auto_scale_xyz([mins[0], maxs[0]], [mins[1], maxs[1]], [mins[2], maxs[2]])
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title("Detected overhang clusters and cut planes")
    plt.tight_layout()
    plt.show()


# --------------------------
# Main logic
# --------------------------

def analyseSTL(
    in_stl,
    out_cuts,
    overhang_angle_deg=None,
    bridge_span_mm_min=None,
    cut_below_mm=None,
    min_segment_height_mm=None,
    max_transform_segment_height_mm=None,  # <-- nur für Pipeline-Kompatibilität
    do_plot=False,
):
    """Analyse STL and write cuts.txt with flags"""
    if not os.path.exists(in_stl):
        raise FileNotFoundError(f"[analyseSTL] STL not found: {in_stl}")

    mesh_t = trimesh.load_mesh(in_stl)
    if mesh_t.is_empty:
        raise RuntimeError("[analyseSTL] mesh appears empty")

    # Parameter aus Config (Fallback, falls None)
    angle = overhang_angle_deg or ANALYSE_STL["overhang_angle_deg"]
    span_th = bridge_span_mm_min or ANALYSE_STL["bridge_span_mm_min"]
    cut_below = cut_below_mm or ANALYSE_STL["cut_below_mm"]
    min_h = min_segment_height_mm or ANALYSE_STL["min_segment_height_mm"]

    normals, _ = face_normals_and_centroids(mesh_t)
    mask = select_downward_faces(normals, angle)
    clusters = build_adjacency_by_vertex(mesh_t.faces, mask)

    zmin, zmax = mesh_t.bounds[0][2], mesh_t.bounds[1][2]
    raw_cuts, debug_info = propose_cut_zs(mesh_t, clusters, span_th, cut_below, zmin)
    final_cuts = enforce_min_spacing(raw_cuts, min_h, zmin)
    seg_flags = build_segment_flags(debug_info, final_cuts, zmax)
    write_segment_file(out_cuts, final_cuts, seg_flags, zmax, use_top_label=True)

    if do_plot:
        visualize_debug(mesh_t, clusters, final_cuts, span_threshold=span_th)

    return final_cuts, seg_flags, debug_info


# --------------------------
# CLI entry point
# --------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyse STL and write combined cuts.txt with transform flags.")
    parser.add_argument("stl", help="input STL file")
    parser.add_argument("--out", default="cuts.txt", help="output file (default: cuts.txt)")
    parser.add_argument("--show", action="store_true", help="show 3D debug plot")
    parser.add_argument("--angle", type=float, help="overhang angle threshold (deg)")
    parser.add_argument("--span", type=float, help="min bridge span (mm)")
    parser.add_argument("--below", type=float, help="cut offset below cluster (mm)")
    parser.add_argument("--minheight", type=float, help="min vertical distance between cuts (mm)")
    args = parser.parse_args()

    cuts, flags, dbg = analyseSTL(
        in_stl=args.stl,
        out_cuts=args.out,
        overhang_angle_deg=args.angle,
        bridge_span_mm_min=args.span,
        cut_below_mm=args.below,
        min_segment_height_mm=args.minheight,
        do_plot=args.show,
    )

    print(f"[analyseSTL] Detected {len(cuts)} cut planes, {len(flags)} segments.")
    print("[analyseSTL] cuts:", np.round(cuts, 3))
    print("[analyseSTL] segment flags:", flags)


if __name__ == "__main__":
    main()