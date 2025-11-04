import re
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from stl import mesh
import os
import time


#############################
# Geometry helpers
#############################

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def triangle_data_from_mesh(stl_path, max_angle_deg=10.0):
    """
    Load an STL (in FINAL print space, i.e. stl_parts/*.stl),
    and extract only the triangles that face 'downward'.

    A triangle is considered "downward-facing" if its normal is within
    max_angle_deg of the -Z direction. That means the surface is printing
    downward / overhanging toward the build plate.
    """
    body = mesh.Mesh.from_file(stl_path).vectors  # shape (N,3,3)
    triangles = []

    cos_max = np.cos(np.deg2rad(max_angle_deg))

    total_tris = 0
    downward_tris = 0

    for tri in body:
        total_tris += 1
        p0, p1, p2 = tri

        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        n_z = n[2]

        is_downward = (n_z <= -cos_max)
        if is_downward:
            downward_tris += 1

            aabb_min = np.min(tri, axis=0)
            aabb_max = np.max(tri, axis=0)

            triangles.append({
                "p0": p0.copy(),
                "p1": p1.copy(),
                "p2": p2.copy(),
                "normal": n.copy(),
                "aabb_min": aabb_min,
                "aabb_max": aabb_max
            })

    print("=== DEBUG triangle_data_from_mesh ===")
    print("Source STL:", stl_path)
    print("Total triangles:", total_tris)
    print("Downward-facing triangles (<= {:.1f}° from -Z): {}".format(max_angle_deg, downward_tris))

    return triangles


def build_triangle_spatial_index(triangles, cell_size=2.0):
    """
    Build a coarse 3D grid (spatial hash) for fast lookup of triangles near a point.
    Returns (grid, cell_size).
    """
    grid = {}

    def cell_id_for_coord(x, y, z):
        return (
            int(np.floor(x / cell_size)),
            int(np.floor(y / cell_size)),
            int(np.floor(z / cell_size)),
        )

    for tri in triangles:
        mn = tri["aabb_min"]
        mx = tri["aabb_max"]

        PAD = 1.0
        x0 = mn[0] - PAD
        y0 = mn[1] - PAD
        z0 = mn[2] - PAD
        x1 = mx[0] + PAD
        y1 = mx[1] + PAD
        z1 = mx[2] + PAD

        ix0, iy0, iz0 = cell_id_for_coord(x0, y0, z0)
        ix1, iy1, iz1 = cell_id_for_coord(x1, y1, z1)

        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                for iz in range(iz0, iz1 + 1):
                    key = (ix + 0, iy + 0, iz + 0)
                    if key not in grid:
                        grid[key] = []
                    grid[key].append(tri)

    return grid, cell_size


def query_triangles_near_point(p, grid, cell_size):
    """
    Return candidate triangles near point p by looking in the point's cell
    and its immediate neighbors.
    """
    x, y, z = p
    ix = int(np.floor(x / cell_size))
    iy = int(np.floor(y / cell_size))
    iz = int(np.floor(z / cell_size))

    candidates = []
    # currently only the same cell; good enough for this use case
    for dx in [0]:
        for dy in [0]:
            for dz in [0]:
                key = (ix + dx, iy + dy, iz + dz)
                if key in grid:
                    candidates.extend(grid[key])
    return candidates


def point_near_downward_surface(midpoint, tri_grid, cell_size, dist_tol=0.4):
    """
    Check if midpoint is close to ANY downward-facing triangle.
    """
    nearby_tris = query_triangles_near_point(midpoint, tri_grid, cell_size)
    if not nearby_tris:
        return False

    mx, my, mz = midpoint

    for tri in nearby_tris:
        p0 = tri["p0"]
        p1 = tri["p1"]
        p2 = tri["p2"]
        n = tri["normal"]

        aabb_min = tri["aabb_min"]
        aabb_max = tri["aabb_max"]

        PAD = 0.5
        if (mx < aabb_min[0] - PAD or mx > aabb_max[0] + PAD or
            my < aabb_min[1] - PAD or my > aabb_max[1] + PAD or
            mz < aabb_min[2] - PAD or mz > aabb_max[2] + PAD):
            continue

        # signed distance to plane
        v_mid = midpoint - p0
        dist_signed = np.dot(v_mid, n)
        if abs(dist_signed) > dist_tol:
            continue

        # barycentric inside test
        proj = midpoint - dist_signed * n

        v0 = p2 - p0
        v1 = p1 - p0
        v2 = proj - p0

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        denom = (dot00 * dot11 - dot01 * dot01)
        if abs(denom) < 1e-16:
            continue

        inv_denom = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        w = 1.0 - u - v

        if (u >= -1e-4 and v >= -1e-4 and w >= -1e-4):
            return True

    return False


def segment_near_downward_surface(
    p_start,
    p_mid,
    p_end,
    tri_grid,
    cell_size,
    dist_tol=0.4
):
    """
    Classify a subsegment as 'near downward' if start OR mid OR end
    is close to a downward-facing triangle.
    """
    if point_near_downward_surface(p_start, tri_grid, cell_size, dist_tol=dist_tol):
        return True
    if point_near_downward_surface(p_mid, tri_grid, cell_size, dist_tol=dist_tol):
        return True
    if point_near_downward_surface(p_end, tri_grid, cell_size, dist_tol=dist_tol):
        return True
    return False


def insert_Z(row, z_value):
    """
    Insert or replace the Z value in a G0/G1 row.
    Ensures the row contains a Z term equal to z_value.
    """
    pattern_Z = r'Z[-0-9.]+[.]?[0-9]*'
    match_z = re.search(pattern_Z, row)

    if match_z is not None:
        row_new = re.sub(pattern_Z, ' Z' + str(round(z_value, 3)), row)
    else:
        pattern_X = r'X[-0-9.]+[.]?[0-9]*'
        pattern_Y = r'Y[-0-9.]+[.]?[0-9]*'
        match_x = re.search(pattern_X, row)
        match_y = re.search(pattern_Y, row)

        if match_y is not None:
            row_new = row[0:match_y.end(0)] + ' Z' + str(round(z_value, 3)) + row[match_y.end(0):]
        elif match_x is not None:
            row_new = row[0:match_x.end(0)] + ' Z' + str(round(z_value, 3)) + row[match_x.end(0):]
        else:
            row_new = 'Z' + str(round(z_value, 3)) + ' ' + row
    return row_new


def replace_E(row, corr_value):
    """
    Re-scale the extrusion value E in the line.
    corr_value ~ number of subsegments.
    """
    pattern_E = r'E[-0-9.]+[.]?[0-9]*'
    match_e = re.search(pattern_E, row)
    if match_e is None:
        return row

    e_val_old = float(match_e.group(0).replace('E', ''))
    if corr_value == 0:
        e_val_new = 0.0
    else:
        e_val_new = round(e_val_old / corr_value, 6)
        if abs(e_val_new) < 1e-3:
            e_val_new = 0.0

    e_str_new = 'E' + str(e_val_new)
    row_new = row[0:match_e.start(0)] + e_str_new + row[match_e.end(0):]
    return row_new


def clean_and_set_feedrate(row, feed_mm_min):
    """
    Remove any existing F... token from the line and append our own F value.
    Guarantees newline at the end.
    """
    row_noF = re.sub(r'F[-0-9.]+[.]?[0-9]*', '', row)
    row_noF = row_noF.rstrip()
    return f"{row_noF} F{feed_mm_min:.1f}\n"


def extract_feedrate(row):
    """
    Look for an F<value> token in a G-code move row and return it as float.
    If none found, return None.
    """
    m = re.search(r'F([0-9.]+)', row)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


#############################
# Core pipeline: backtransform_data
#############################

def backtransform_data(
    data,
    interp,
    maximal_length,
    z_min,
    tri_grid,
    cell_size,
    slow_feedrate,
    medium_feedrate,
    floor_z
):
    """
    Convert transformed G-code 'data' into final printer-space toolpaths.

    NEW PART:
    - Travel moves must not go below the cut plane (floor_z).
      For non-extruding subsegments:
        sub_z = max(sub_z, floor_z)

    Slowdown handling and feedrate buffer logic as in the original version.
    """

    new_data = []

    pattern_X = r'X[-0-9.]+[.]?[0-9]*'
    pattern_Y = r'Y[-0-9.]+[.]?[0-9]*'
    pattern_Z = r'Z[-0-9.]+[.]?[0-9]*'
    pattern_G = r'\AG[01]\s'   # matches lines starting with G0 or G1

    # State across lines
    x_old, y_old = 0.0, 0.0
    z_layer = 0.0
    in_perimeter = False

    # Feedrate FSM state
    last_normal_F = None   # last known fast extrusion feedrate from slicer
    was_slow = False       # True iff we are CURRENTLY in slow mode

    # One-line buffer for "pre-brake" medium feedrate
    buffered_row = None
    buffered_is_extruding = False
    buffered_is_slow = False  # whether buffered seg was slow
    # NOTE: buffered_row should always have newline when stored.

    # debug counters
    perimeter_true_count = 0
    total_subsegments = 0
    subsegments_near_downward = 0
    subsegments_slow_candidates = 0

    def flush_buffer(row_list, force_medium=False):
        """
        Push the current buffered_row into row_list.
        - If force_medium is True and the buffered row is extruding and NOT slow,
          rewrite its feedrate to medium_feedrate before output.
        """
        nonlocal buffered_row, buffered_is_extruding, buffered_is_slow

        if buffered_row is None:
            return

        out_line = buffered_row
        if force_medium and buffered_is_extruding and (not buffered_is_slow):
            out_line = clean_and_set_feedrate(out_line, medium_feedrate)

        row_list.append(out_line)

        buffered_row = None
        buffered_is_extruding = False
        buffered_is_slow = False

    def set_buffer(new_text, is_extruding, is_slow):
        """
        Overwrite the single-line buffer with a new subsegment.
        Assumes new_text already has newline.
        """
        nonlocal buffered_row, buffered_is_extruding, buffered_is_slow
        buffered_row = new_text
        buffered_is_extruding = is_extruding
        buffered_is_slow = is_slow

    for row in data:
        stripped = row.strip()

        # ----- comment lines: perimeter / infill detection -----
        if stripped.startswith(";"):
            lower = stripped.lower()
            prev_mode = in_perimeter

            # switch perimeter mode based on comments
            if "perimeter" in lower:
                in_perimeter = True
            if "infill" in lower or ("fill" in lower and "perimeter" not in lower):
                in_perimeter = False

            if (in_perimeter is True) and (prev_mode is False):
                perimeter_true_count += 1

            # comments must be in sequence, so flush buffer first
            flush_buffer(new_data, force_medium=False)
            new_data.append(row)
            continue

        # ----- non-move lines (M-codes etc) -----
        g_match = re.search(pattern_G, row)
        if g_match is None:
            # maybe feedrate changes here
            fval = extract_feedrate(row)
            if fval is not None and abs(fval - slow_feedrate) > 1e-6:
                last_normal_F = fval
                was_slow = False

            flush_buffer(new_data, force_medium=False)
            new_data.append(row)
            continue

        # ----- G0/G1 move line -----

        # coords present?
        x_match = re.search(pattern_X, row)
        y_match = re.search(pattern_Y, row)
        z_match = re.search(pattern_Z, row)
        if (x_match is None and y_match is None and z_match is None):
            # just feedrate-style G1?
            fval = extract_feedrate(row)
            if fval is not None and abs(fval - slow_feedrate) > 1e-6:
                last_normal_F = fval
                was_slow = False

            flush_buffer(new_data, force_medium=False)
            new_data.append(row)
            continue

        # If slicer gave a Z here, that's our nominal layer height
        if z_match is not None:
            z_layer = float(z_match.group(0).replace('Z', ''))

        # get new XY target
        x_new = x_old
        y_new = y_old
        if x_match is not None:
            x_new = float(x_match.group(0).replace('X', ''))
        if y_match is not None:
            y_new = float(y_match.group(0).replace('Y', ''))

        # segment split
        dist_xy = np.linalg.norm([x_new - x_old, y_new - y_old])
        num_segm = max(int(dist_xy // maximal_length + 1), 1)

        x_vals = np.linspace(x_old, x_new, num_segm + 1)
        y_vals = np.linspace(y_old, y_new, num_segm + 1)

        # Backtransform Z at each point BEFORE travel clamping
        z_vals_bt = np.array([
            np.maximum(z_layer - interp(xv, yv), z_min)
            for (xv, yv) in zip(x_vals, y_vals)
        ])

        # base row w/ first Z and corrected E split
        base_row = insert_Z(row, z_vals_bt[0])
        base_row = replace_E(base_row, num_segm)

        for j in range(num_segm):
            sub_x = x_vals[j + 1]
            sub_y = y_vals[j + 1]

            # Extrusion detection (check E after split)
            extruding_here = False
            mE = re.search(r'E(-?\d+\.?\d*)', base_row)
            if mE:
                try:
                    e_val = float(mE.group(1))
                    if abs(e_val) > 1e-9:
                        extruding_here = True
                except ValueError:
                    pass

            # For this substep: backtransformed target Z
            sub_z_bt = z_vals_bt[j + 1]

            # Travel clamp: no extrusion -> must not go below floor_z
            if not extruding_here:
                sub_z = max(sub_z_bt, floor_z)
            else:
                sub_z = sub_z_bt

            # build the subsegment line
            single_row = re.sub(pattern_X, 'X' + str(round(sub_x, 3)), base_row)
            single_row = re.sub(pattern_Y, 'Y' + str(round(sub_y, 3)), single_row)
            single_row = re.sub(pattern_Z, 'Z' + str(round(sub_z, 3)), single_row)

            # Slowdown check: only relevant for perimeter with extrusion
            if in_perimeter and extruding_here:
                near_down = segment_near_downward_surface(
                    np.array([x_vals[j],     y_vals[j],     z_vals_bt[j]]),
                    np.array([
                        0.5 * (x_vals[j] + x_vals[j + 1]),
                        0.5 * (y_vals[j] + y_vals[j + 1]),
                        0.5 * (z_vals_bt[j] + z_vals_bt[j + 1])
                    ]),
                    np.array([x_vals[j + 1], y_vals[j + 1], z_vals_bt[j + 1]]),
                    tri_grid,
                    cell_size,
                    dist_tol=0.4
                )
                if near_down:
                    subsegments_near_downward += 1
            else:
                near_down = False

            slow_this = (in_perimeter and extruding_here and near_down)
            if slow_this:
                subsegments_slow_candidates += 1

            # ---- Feedrate FSM / Buffer ----

            fval_here = extract_feedrate(single_row)

            if slow_this:
                # We are entering or inside slow zone

                # First slow subsegment: flush previous buffer
                # but with medium_feedrate (pre-brake)
                if not was_slow:
                    flush_buffer(new_data, force_medium=True)
                else:
                    flush_buffer(new_data, force_medium=False)

                # Now write this slow line
                line_out = single_row
                if not was_slow:
                    # first slow line: explicitly set slow_feedrate
                    line_out = clean_and_set_feedrate(line_out, slow_feedrate)
                    was_slow = True
                else:
                    if not line_out.endswith("\n"):
                        line_out = line_out.rstrip() + "\n"

                new_data.append(line_out)

                # do not buffer slow lines
                buffered_row = None
                buffered_is_extruding = False
                buffered_is_slow = False

            else:
                # not a slow candidate
                if was_slow:
                    # we just left slow region -> restore fast feedrate
                    if fval_here is not None and abs(fval_here - slow_feedrate) > 1e-6:
                        # slicer already set a new fast F
                        last_normal_F = fval_here
                        restored = single_row
                        if not restored.endswith("\n"):
                            restored = restored.rstrip() + "\n"
                    else:
                        if last_normal_F is not None:
                            restored = clean_and_set_feedrate(single_row, last_normal_F)
                        else:
                            restored = single_row
                            if not restored.endswith("\n"):
                                restored = restored.rstrip() + "\n"

                    was_slow = False

                    # this 'restored' move is a candidate before next slow zone
                    flush_buffer(new_data, force_medium=False)
                    set_buffer(restored, extruding_here, is_slow=False)

                else:
                    # normal fast mode, no slow active
                    if fval_here is not None and abs(fval_here - slow_feedrate) > 1e-6:
                        last_normal_F = fval_here

                    # old buffer is no longer 'just before slow', flush without medium
                    flush_buffer(new_data, force_medium=False)

                    if not single_row.endswith("\n"):
                        single_row = single_row.rstrip() + "\n"
                    set_buffer(single_row, extruding_here, is_slow=False)

            total_subsegments += 1

        # update current XY
        x_old = x_new
        y_old = y_new
        # do not flush buffer here → it may be pre-brake for next slow

    # At the end, flush remaining buffer without medium (no more slow coming)
    flush_buffer(new_data, force_medium=False)

    # Debug
    print("=== DEBUG backtransform_data ===")
    print("perimeter_true_count:", perimeter_true_count,
          "(times we entered perimeter mode)")
    print("total_subsegments:", total_subsegments)
    print("subsegments_near_downward:", subsegments_near_downward,
          "(geometry matched downward-facing triangles)")
    print("subsegments_slow_candidates:", subsegments_slow_candidates,
          "(perimeter AND downward -> slow zone hits)")

    return new_data


#############################
# transformGCode main entry
#############################

def transformGCode(
    in_file,
    in_transform_for_interp,
    out_dir,
    surface_for_slowdown,
    maximal_length=1.0,
    x_shift=0.0,
    y_shift=0.0,
    z_desired=0.1,
    downward_angle_deg=10.0,
    slow_feedrate=180.0,
    medium_feedrate=400.0
):
    start = time.time()

    # 1. Original G-code (transform space)
    with open(in_file, 'r') as f_gcode:
        data = f_gcode.readlines()

    # 2. Interpolator for backtransform from tf_surfaces/<part>.stl
    surf_mesh_tf = mesh.Mesh.from_file(in_transform_for_interp).vectors
    surf_xy = np.reshape(surf_mesh_tf[:, :, [0, 1]], (-1, 2))
    delaunay_grid = Delaunay(surf_xy)
    interp = LinearNDInterpolator(
        delaunay_grid,
        np.reshape(surf_mesh_tf[:, :, 2], -1),
        0
    )

    # 3. Final-part mesh for:
    #    a) slowdown detection (downward-facing areas)
    #    b) travel-Z clamp floor height
    final_mesh = mesh.Mesh.from_file(surface_for_slowdown).vectors

    # a) downward-facing triangles for slowdown
    downward_triangles = triangle_data_from_mesh(
        surface_for_slowdown,
        max_angle_deg=downward_angle_deg
    )

    tri_grid, cell_size = build_triangle_spatial_index(
        downward_triangles,
        cell_size=2.0
    )

    # b) floor_z from the cut final part in print coordinates
    floor_z = float(np.min(final_mesh[:, :, 2]))

    # 4. Backtransform incl. travel clamp
    data_bt = backtransform_data(
        data=data,
        interp=interp,
        maximal_length=maximal_length,
        z_min=z_desired + 0.2,
        tri_grid=tri_grid,
        cell_size=cell_size,
        slow_feedrate=slow_feedrate,
        medium_feedrate=medium_feedrate,
        floor_z=floor_z
    )

    data_bt_string = ''.join(data_bt)

    # 5. Save
    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.basename(in_file)
    output_path = os.path.join(out_dir, file_name)

    with open(output_path, 'w', newline="\n") as f_gcode_bt:
        f_gcode_bt.write(data_bt_string)

    end = time.time()
    print('GCode generated in {:.1f}s, saved in {}'.format(end - start, output_path))
    return output_path


#############################
# Wrapper: transform selected segments only
#############################

def transformGCODE_for_segments(
    segment_ids,
    base_gcode_dir="gcode_tf",
    base_tf_dir="tf_surfaces",
    base_final_dir="stl_parts",
    out_dir="gcode_bt",
    maximal_length=1.0,
    downward_angle_deg=10.0,
    slow_feedrate=180.0,
    medium_feedrate=400.0
):
    """
    Run GCode transformation only for selected segment IDs.
    This keeps the original transformGCode() behavior and only wraps
    it in a loop over segment indices.
    """
    print("[transformGCODE_for_segments] START")
    os.makedirs(out_dir, exist_ok=True)

    for seg_id in segment_ids:
        gcode_name = f"segment_{seg_id}.gcode"
        tf_name = f"segment_{seg_id}.stl"

        in_file = os.path.join(base_gcode_dir, gcode_name)
        in_transform_for_interp = os.path.join(base_tf_dir, tf_name)
        surface_for_slowdown = os.path.join(base_final_dir, tf_name)

        if not os.path.exists(in_file):
            print(f"  WARNING: missing G-code: {in_file}")
            continue
        if not os.path.exists(in_transform_for_interp):
            print(f"  WARNING: missing transform surface: {in_transform_for_interp}")
            continue
        if not os.path.exists(surface_for_slowdown):
            print(f"  WARNING: missing STL: {surface_for_slowdown}")
            continue

        transformGCode(
            in_file=in_file,
            in_transform_for_interp=in_transform_for_interp,
            out_dir=out_dir,
            surface_for_slowdown=surface_for_slowdown,
            maximal_length=maximal_length,
            downward_angle_deg=downward_angle_deg,
            slow_feedrate=slow_feedrate,
            medium_feedrate=medium_feedrate
        )

    print("[transformGCODE_for_segments] DONE\n")


if __name__ == "__main__":
    test_segments = ["2", "4"]
    transformGCODE_for_segments(test_segments)
