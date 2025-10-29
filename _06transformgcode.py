import re
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from stl import mesh
import os
import time


#############################
# Triangle extraction / geometry analysis
#############################

def _triangle_data_from_mesh(stl_path, max_angle_deg=10.0):
    """
    Load an STL (in FINAL print space, i.e. stl_parts/*.stl),
    compute face normals, and keep only downward-facing triangles
    (normals pointing within max_angle_deg of -Z).

    Returns a list of dicts:
    {
        "p0", "p1", "p2"            ... np.array([x,y,z]) (triangle vertices)
        "normal"                    ... np.array([nx,ny,nz])
        "aabb_min", "aabb_max"      ... AABB for fast rejection in spatial grid
    }
    """
    stl_mesh = mesh.Mesh.from_file(stl_path)

    triangles = []
    total_tris = 0
    downward_tris = 0
    cos_max = np.cos(np.radians(max_angle_deg))

    all_v0 = stl_mesh.v0
    all_v1 = stl_mesh.v1
    all_v2 = stl_mesh.v2

    for (p0, p1, p2) in zip(all_v0, all_v1, all_v2):
        tri = np.array([p0, p1, p2], dtype=float)
        total_tris += 1

        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            # Degenerate triangle -> skip
            continue
        n = n / n_norm
        n_z = n[2]

        # facing downward if close to -Z
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
    print("Downward-facing triangles (<= {:.1f}° from -Z): {}".format(
        max_angle_deg, downward_tris
    ))

    return triangles


def _build_triangle_spatial_index(triangles, cell_size=1.0):
    """
    Build a coarse 3D spatial grid (dict) for the downward-facing triangles.

    Parameters
    ----------
    triangles : list of triangle dicts
    cell_size : float

    Returns
    -------
    grid : dict[(ix,iy,iz)] -> list of triangle dicts
    cell_size : float (echo back)
    """
    grid = {}

    def cell_id_for_coord(x, y, z):
        return (
            int(np.floor(x / cell_size)),
            int(np.floor(y / cell_size)),
            int(np.floor(z / cell_size))
        )

    PAD = 0.001

    for tri in triangles:
        mn = tri["aabb_min"]
        mx = tri["aabb_max"]

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
                    key = (ix, iy, iz)
                    if key not in grid:
                        grid[key] = []
                    grid[key].append(tri)

    return grid, cell_size


def _query_triangles_near_point(p, grid, cell_size):
    """
    Return candidate triangles near point p by looking in the point's cell
    and its immediate neighbors.

    p: np.array([x,y,z])
    grid: dict from (ix,iy,iz) -> list of triangles
    cell_size: float

    Returns
    -------
    list of triangle dicts (may contain duplicates; we can ignore that)
    """
    x, y, z = p
    ix = int(np.floor(x / cell_size))
    iy = int(np.floor(y / cell_size))
    iz = int(np.floor(z / cell_size))

    out_list = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                key = (ix + dx, iy + dy, iz + dz)
                if key in grid:
                    out_list.extend(grid[key])
    return out_list


def _point_near_downward_surface(point, grid, cell_size, dist_threshold=0.4):
    """
    Check if 'point' lies close (<= dist_threshold) to any downward facing
    triangle from the STL.

    distance metric: perpendicular distance from point to triangle plane,
    plus a point-in-triangle (barycentric) check.
    """
    candidates = _query_triangles_near_point(point, grid, cell_size)
    px, py, pz = point

    for tri in candidates:
        p0 = tri["p0"]
        p1 = tri["p1"]
        p2 = tri["p2"]
        normal = tri["normal"]

        # project point onto triangle plane
        v0p = np.array([px, py, pz]) - p0
        dist = abs(np.dot(v0p, normal))

        if dist <= dist_threshold:
            # now check if projected point is inside triangle
            # Barycentric method
            u = p1 - p0
            v = p2 - p0
            w = np.array([px, py, pz]) - p0

            uv = np.dot(u, v)
            wv = np.dot(w, v)
            vv = np.dot(v, v)
            uu = np.dot(u, u)
            uw = np.dot(u, w)

            denom = (uv * uv - uu * vv)
            if abs(denom) < 1e-14:
                continue

            s = (uv * wv - vv * uw) / denom
            t = (uv * uw - uu * wv) / denom

            if s >= -1e-8 and t >= -1e-8 and (s + t) <= 1.0 + 1e-8:
                # close and inside triangle projection -> call it "near downward"
                return True

    return False


def _segment_near_downward_surface(p0, p1, grid, cell_size, samples=4, dist_threshold=0.4):
    """
    Check if any sampled point along the XY segment p0->p1 is near
    a downward-facing surface.
    p0, p1 : np.array([x,y,z])

    We just linearly interpolate a few samples and call _point_near_downward_surface.
    """
    for i in range(samples + 1):
        alpha = i / float(samples)
        p = p0 * (1 - alpha) + p1 * alpha
        if _point_near_downward_surface(p, grid, cell_size, dist_threshold=dist_threshold):
            return True
    return False


#############################
# G-code manipulation helpers
#############################

def _insert_Z(row, z_value):
    """
    Insert or replace the Z term in the G-code move line.
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


def _replace_E(row, corr_value):
    """
    Re-scale the extrusion value E in the line.
    corr_value ~ number of subsegments we split the move into.
    So we divide the original extrusion amount by corr_value and
    repeat that same smaller amount for each subsegment.
    """
    pattern_E = r'E[-0-9.]+[.]?[0-9]*'
    match_e = re.search(pattern_E, row)
    if match_e is None:
        return row

    val = match_e.group(0).replace('E', '')
    try:
        val_f = float(val)
    except ValueError:
        return row

    if corr_value < 1:
        corr_value = 1

    new_val = val_f / corr_value
    row_new = re.sub(pattern_E, ' E' + str(round(new_val, 5)), row)
    return row_new


def _clean_and_set_feedrate(row, feed_value):
    """
    Remove existing F... feedrate token and append new feedrate F<feed_value>
    to the row. Keeps newline status.
    """
    pattern_F = r'F[-0-9.]+[.]?[0-9]*'
    was_nl = row.endswith("\n")
    row_no_f = re.sub(pattern_F, '', row).strip()
    row_no_f = row_no_f + ' F' + str(feed_value)
    if was_nl:
        row_no_f += '\n'
    return row_no_f


def _extract_feedrate(row):
    """
    Extract current feedrate F... from a row.
    Returns float or None.
    """
    pattern_F = r'F[-0-9.]+[.]?[0-9]*'
    match_f = re.search(pattern_F, row)
    if match_f is None:
        return None
    val = match_f.group(0).replace('F', '')
    try:
        f = float(val)
    except ValueError:
        return None
    return f


#############################
# Backtransform data helpers
#############################

def _backtransform_data(stl_path):
    """
    Load a single-surface STL which encodes the backtransform height map
    (tf_surfaces/<part>.stl). We use it to build a scattered interpolator
    x,y -> z_offset.

    Returns (interp, z_min):

    interp(x,y) ~ offset between 'transformed Z' and final printer Z.
    z_min       ~ minimum z (used as floor so we don't go negative)
    """
    stl_mesh = mesh.Mesh.from_file(stl_path)

    P = stl_mesh.points.reshape((-1, 3))
    # unique-ish points
    # using rounding to avoid crazy duplicates / floating noise
    P_uniq = np.unique(np.round(P, 6), axis=0)

    # Build interpolator for z as a function of (x,y)
    xy = P_uniq[:, :2]
    z = P_uniq[:, 2]
    tri = Delaunay(xy)
    interp = LinearNDInterpolator(tri, z, fill_value=0.0)

    z_min = float(np.min(z))

    print("=== DEBUG backtransform_data ===")
    print("Source STL:", stl_path)
    print("Unique points:", len(P_uniq))
    print("Z-min for safety floor:", z_min)

    return interp, z_min


#############################
# Main pipeline: transformGCode
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
    """
    Convert SuperSlicer output in transformed coordinates (in_file)
    into final printer-space G-code.

    Steps:
    - Build interpolator from tf_surface STL to "backtransform" Z.
    - Analyse final part mesh to detect downward-facing regions.
    - For each G1 move:
        * split long XY moves into subsegments <= maximal_length
        * compute final Z for each segment using backtransform
        * slow down perimeters that are under overhang / downward faces
        * insert recovery feedrate when leaving slow region
        * ensure Z is always in the line
        * recalc extrusion E proportionally

    Parameters
    ----------
    in_file : str
        Path to G-code in transformed coordinates (gcode_tf/<part>.gcode)
    in_transform_for_interp : str
        Path to the "tf_surfaces/<part>.stl" surface used to build height map.
    out_dir : str
        Output directory for final G-code (gcode_parts/)
    surface_for_slowdown : str
        Path to the FINAL geometry STL (stl_parts/<part>.stl).
        We analyze this mesh to detect downward-facing regions and slow perimeters.
    maximal_length : float
        Max XY segment length before we subdivide
    x_shift, y_shift : float
        (Currently unused in final coords but kept for compatibility)
    z_desired : float
        Base desired Z offset? (kept for compatibility)
    downward_angle_deg : float
        Triangles whose normals point within this angle of -Z are considered
        "downward" → triggers slowdown.
    slow_feedrate : float
        Feedrate (F value) to use in downward-facing perimeter segments
    medium_feedrate : float
        Feedrate to use for normal perimeters when forcing medium speed in
        buffered perimeter flush

    Output
    ------
    Writes a new G-code file <out_dir>/<basename>.gcode_final.gcode
    """
    os.makedirs(out_dir, exist_ok=True)

    ## 1. Backtransform interpolator
    interp, z_min = _backtransform_data(in_transform_for_interp)

    ## 2. Downward-facing surface analysis (for slowdown)
    tris_down = _triangle_data_from_mesh(
        surface_for_slowdown,
        max_angle_deg=downward_angle_deg
    )
    tri_grid, tri_cell = _build_triangle_spatial_index(tris_down, cell_size=1.0)

    ## 3. Read source G-code
    with open(in_file, 'r') as f:
        gcode_lines = f.readlines()

    out_lines = []

    # We'll maintain a buffer for "normal perimeter lines" so we can rewrite
    # feedrates at the right time when leaving slow regions.
    buffered_row = None
    buffered_is_extruding = False
    buffered_is_slow = False

    was_slow = False        # were we in slow mode last subsegment?
    last_normal_F = None    # last known fast feedrate we saw in non-slow segments

    # debug counters
    perimeter_true_count = 0
    total_subsegments = 0
    subsegments_near_downward = 0
    subsegments_slow_candidates = 0

    def flush_buffer(row_list, force_medium=False):
        """
        Push the current buffered_row into row_list.
        - If force_medium is True and the buffered row is extruding and NOT slow,
          we rewrite its feedrate to medium_feedrate.
        - After flushing, clear the buffer.
        """
        nonlocal buffered_row, buffered_is_extruding, buffered_is_slow

        if buffered_row is None:
            return

        out_line = buffered_row

        if force_medium and buffered_is_extruding and (not buffered_is_slow):
            # Remove any F... and replace with F<medium_feedrate>
            out_line = _clean_and_set_feedrate(out_line, medium_feedrate)

        row_list.append(out_line)

        buffered_row = None
        buffered_is_extruding = False
        buffered_is_slow = False

    def set_buffer(line, is_extruding, is_slow):
        """
        Update the single-line buffer with new row.
        """
        nonlocal buffered_row, buffered_is_extruding, buffered_is_slow
        buffered_row = line
        buffered_is_extruding = is_extruding
        buffered_is_slow = is_slow

    # We'll track XY position layer by layer (SuperSlicer style)
    x_old = 0.0
    y_old = 0.0
    z_layer = 0.0

    # regex patterns reused in loop
    pattern_G1 = r'^G1\b'
    pattern_X = r'X[-0-9.]+[.]?[0-9]*'
    pattern_Y = r'Y[-0-9.]+[.]?[0-9]*'
    pattern_Z = r'Z[-0-9.]+[.]?[0-9]*'
    pattern_E = r'E[-0-9.]+[.]?[0-9]*'
    pattern_COMMENT_PERIM = r';TYPE:Perimeter'

    for row in gcode_lines:
        row_stripped = row.strip()

        # Track layer Z if slicer sets it (e.g. "G1 Z...")
        z_match = re.search(pattern_Z, row)
        if z_match is not None:
            try:
                z_layer = float(z_match.group(0).replace('Z', ''))
            except ValueError:
                pass

        # Not a motion line? Just flush any buffer if needed and pass it through.
        if re.search(pattern_G1, row) is None or \
           (re.search(pattern_X, row) is None and re.search(pattern_Y, row) is None):

            # Before dumping arbitrary non-G1 or travel, flush buffer
            flush_buffer(out_lines, force_medium=False)
            out_lines.append(row)
            continue

        # It's a movement (G1...), maybe extrusion.
        # We get new X,Y targets
        x_match = re.search(pattern_X, row)
        y_match = re.search(pattern_Y, row)

        x_new = x_old
        y_new = y_old
        if x_match is not None:
            x_new = float(x_match.group(0).replace('X', ''))
        if y_match is not None:
            y_new = float(y_match.group(0).replace('Y', ''))

        # break long XY move into small segments
        dist_xy = np.linalg.norm([x_new - x_old, y_new - y_old])
        num_segm = max(int(dist_xy // maximal_length + 1), 1)

        x_vals = np.linspace(x_old, x_new, num_segm + 1)
        y_vals = np.linspace(y_old, y_new, num_segm + 1)

        # backtransform Z for each segment endpoint
        z_vals = np.array([
            np.maximum(z_layer - interp(xv, yv), z_min)
            for (xv, yv) in zip(x_vals, y_vals)
        ])

        # base row template:
        base_row = _insert_Z(row, z_vals[0])
        base_row = _replace_E(base_row, num_segm)

        for j in range(num_segm):
            sub_x = x_vals[j + 1]
            sub_y = y_vals[j + 1]
            sub_z = z_vals[j + 1]

            # coordinate-substituted single subsegment line
            single_row = re.sub(pattern_X, 'X' + str(round(sub_x, 3)), base_row)
            single_row = re.sub(pattern_Y, 'Y' + str(round(sub_y, 3)), single_row)
            single_row = re.sub(pattern_Z, 'Z' + str(round(sub_z, 3)), single_row)

            # Mark if this subsegment is extruding:
            is_extruding = (re.search(pattern_E, single_row) is not None)

            # We only consider slowdowns on "perimeter" lines with extrusion.
            is_perimeter = (re.search(pattern_COMMENT_PERIM, row) is not None)
            if is_perimeter and is_extruding:
                perimeter_true_count += 1

            total_subsegments += 1

            # We check if this subsegment is near downward-facing geometry
            # Use midpoint of this subsegment, keep same Z for test.
            mid_x = (x_vals[j] + x_vals[j + 1]) * 0.5
            mid_y = (y_vals[j] + y_vals[j + 1]) * 0.5
            mid_z = (z_vals[j] + z_vals[j + 1]) * 0.5
            p_mid = np.array([mid_x, mid_y, mid_z], dtype=float)

            # Only if this is perimeter & extruding do we consider slowdown:
            near_down = False
            if is_perimeter and is_extruding:
                near_down = _point_near_downward_surface(
                    p_mid,
                    tri_grid,
                    tri_cell,
                    dist_threshold=0.4
                )
                if near_down:
                    subsegments_near_downward += 1

            # Candidate for slow? => near downward-facing
            is_slow_candidate = (near_down and is_perimeter and is_extruding)
            if is_slow_candidate:
                subsegments_slow_candidates += 1

            # logic for slow feedrate injection and buffering:
            if is_slow_candidate:
                # We're in "slow" mode for this subsegment

                # flush whatever was buffered, forcing medium feed if appropriate
                flush_buffer(out_lines, force_medium=True)

                # Force slow feedrate on this single_row
                single_row_slow = _clean_and_set_feedrate(single_row, slow_feedrate)

                # output the slow row immediately
                if not single_row_slow.endswith("\n"):
                    single_row_slow = single_row_slow.rstrip() + "\n"
                out_lines.append(single_row_slow)

                was_slow = True

                # nach Ausgabe des slow-Segments kommt KEIN Buffer,
                # weil wir ihn schon geflusht haben und slow-Zeilen direkt schreiben
                set_buffer(None, False, False)

            else:
                # Dieses Subsegment ist NICHT slow.
                fval_here = _extract_feedrate(single_row)

                if was_slow:
                    # Wir verlassen gerade den Slow-Bereich.
                    # Wir müssen EINMAL die schnelle Feedrate wiederherstellen.
                    # Falls der Slicer hier schon eine schnelle F hat (fval_here != slow_feedrate),
                    # nehmen wir die. Sonst nehmen wir last_normal_F.
                    if fval_here is not None and abs(fval_here - slow_feedrate) > 1e-6:
                        # slicer hat eine schnelle Feedrate gesetzt
                        last_normal_F = fval_here
                        restored = single_row
                        if not restored.endswith("\n"):
                            restored = restored.rstrip() + "\n"
                    else:
                        # slicer hat hier keine schnelle F -> wir injizieren last_normal_F
                        if last_normal_F is not None:
                            restored = _clean_and_set_feedrate(single_row, last_normal_F)
                        else:
                            # fallback: einfach nur newline
                            restored = single_row
                            if not restored.endswith("\n"):
                                restored = restored.rstrip() + "\n"

                    # Dieser 'restored' Move wird NICHT gebuffert:
                    flush_buffer(out_lines, force_medium=False)
                    out_lines.append(restored)

                    was_slow = False
                    # Buffer bleibt leer nach diesem Schritt
                    set_buffer(None, False, False)

                else:
                    # Wir sind NICHT im slow-Bereich (und waren es auch nicht gerade).
                    # Wir puffern diese Zeile (max 1 Zeile Buffer).
                    # Warum? Damit wir ggf. beim nächsten Step merken,
                    # dass wir DOCH slow werden und dann medium-feedrate setzen können.
                    if fval_here is not None and abs(fval_here - slow_feedrate) > 1e-6:
                        last_normal_F = fval_here

                    # Flush vorherige Pufferzeile, ohne medium-Force
                    flush_buffer(out_lines, force_medium=False)

                    # Neue Zeile in Buffer legen
                    if not single_row.endswith("\n"):
                        single_row = single_row.rstrip() + "\n"
                    set_buffer(single_row, is_extruding, False)

        # update state for next segment
        x_old = x_new
        y_old = y_new

    # am Ende alles flushen
    flush_buffer(out_lines, force_medium=False)

    # Debug-Ausgabe
    print("=== transformGCode stats ===")
    print("Input file:", in_file)
    print("Total subsegments:", total_subsegments)
    print("Perimeter+extrusion subsegments:", perimeter_true_count)
    print("Near downward subsegments:", subsegments_near_downward)
    print("Slowdown candidates:", subsegments_slow_candidates)

    # Write output file
    base_name = os.path.basename(in_file)
    out_name = base_name.replace('.gcode', '') + "_final.gcode"
    out_path = os.path.join(out_dir, out_name)

    with open(out_path, 'w') as fw:
        fw.writelines(out_lines)

    print("Wrote transformed G-code to:", out_path)

    return out_path


if __name__ == "__main__":
    # Basic manual test hook (adjust paths for your environment)
    file_path = "gcode_tf/part.gcode"
    tf_surface = "tf_surfaces/part.stl"
    final_part = "stl_parts/part.stl"
    out_dir = "gcode_parts"

    transformGCode(
        in_file=file_path,
        in_transform_for_interp=tf_surface,
        out_dir=out_dir,
        surface_for_slowdown=final_part,
        maximal_length=0.5,
        x_shift=0.0,
        y_shift=0.0,
        z_desired=0.1,
        downward_angle_deg=10.0,
        slow_feedrate=180.0,
        medium_feedrate=400.0
    )
