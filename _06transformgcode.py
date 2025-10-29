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

    Returns
    -------
    triangles : list of dict
        Each dict:
        {
            "p0": np.array([x,y,z]),
            "p1": np.array([x,y,z]),
            "p2": np.array([x,y,z]),
            "normal": np.array([nx,ny,nz]),
            "aabb_min": np.array([xmin,ymin,zmin]),
            "aabb_max": np.array([xmax,ymax,zmax])
        }
        These triangles and AABBs are all expressed in FINAL coordinates,
        i.e. the same space as the backtransformed G-code.
    """
    body = mesh.Mesh.from_file(stl_path).vectors  # shape (N,3,3)
    triangles = []

    # We consider a triangle "downward" if its normal forms <= max_angle_deg
    # with the negative Z axis (0,0,-1).
    # angle(n, -Z) = arccos( dot(n, -Z) ) = arccos(-n_z)
    # So we require: -n_z >= cos(max_angle_deg)  <=>  n_z <= -cos_max
    cos_max = np.cos(np.deg2rad(max_angle_deg))

    total_tris = 0
    downward_tris = 0

    for tri in body:
        total_tris += 1
        p0, p1, p2 = tri

        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            # Degenerate triangle -> skip
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

    triangles: list of dicts with keys:
        "aabb_min", "aabb_max", "p0","p1","p2","normal"
    cell_size: float, size of each grid cell in mm

    Returns
    -------
    grid : dict[(ix,iy,iz)] -> list of triangle dicts
    Also returns cell_size so caller knows what was used.
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

        # compute all cells overlapped by this tri's AABB (expanded slightly)
        PAD = 1.0  # same idea as before; "downward region" can bleed a bit
        x0 = mn[0] - PAD
        y0 = mn[1] - PAD
        z0 = mn[2] - PAD
        x1 = mx[0] + PAD
        y1 = mx[1] + PAD
        z1 = mx[2] + PAD

        ix0, iy0, iz0 = cell_id_for_coord(x0, y0, z0)
        ix1, iy1, iz1 = cell_id_for_coord(x1, y1, z1)

        for ix in range(ix0, ix1+1):
            for iy in range(iy0, iy1+1):
                for iz in range(iz0, iz1+1):
                    key = (ix, iy, iz)
                    if key not in grid:
                        grid[key] = []
                    grid[key].append(tri)

    return grid, cell_size

def query_triangles_near_point(p, grid, cell_size):
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

    candidates = []
    # for dx in [-1, 0, 1]:
        # for dy in [-1, 0, 1]:
            # for dz in [-1, 0, 1]:
    for dx in [0]:
        for dy in [0]:
            for dz in [0]:
                key = (ix+dx, iy+dy, iz+dz)
                if key in grid:
                    candidates.extend(grid[key])
    return candidates

def point_near_downward_surface(midpoint, tri_grid, cell_size, dist_tol=0.4):
    """
    Check if midpoint is close to ANY downward-facing triangle, but use
    a spatial grid so we only test nearby triangles.

    tri_grid: the spatial index (dict) from build_triangle_spatial_index
    cell_size: the same cell size used to build that grid
    dist_tol: allowed distance to triangle plane
    """

    # get only nearby triangles
    nearby_tris = query_triangles_near_point(midpoint, tri_grid, cell_size)
    if not nearby_tris:
        return False

    mx, my, mz = midpoint

    for tri in nearby_tris:
        p0 = tri["p0"]
        p1 = tri["p1"]
        p2 = tri["p2"]
        n  = tri["normal"]

        aabb_min = tri["aabb_min"]
        aabb_max = tri["aabb_max"]

        # Quick local AABB reject again, but cheaper now
        PAD = 0.5
        if (mx < aabb_min[0]-PAD or mx > aabb_max[0]+PAD or
            my < aabb_min[1]-PAD or my > aabb_max[1]+PAD or
            mz < aabb_min[2]-PAD or mz > aabb_max[2]+PAD):
            continue

        # Distance to plane
        v_mid = midpoint - p0
        dist_signed = np.dot(v_mid, n)
        if abs(dist_signed) > dist_tol:
            continue

        # Project and barycentric check
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
    Uses spatial grid for performance.
    """
    if point_near_downward_surface(p_start, tri_grid, cell_size, dist_tol=dist_tol):
        return True
    if point_near_downward_surface(p_mid,   tri_grid, cell_size, dist_tol=dist_tol):
        return True
    if point_near_downward_surface(p_end,   tri_grid, cell_size, dist_tol=dist_tol):
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
    corr_value ~ number of subsegments we split the move into.
    So we divide the original extrusion amount by corr_value and
    repeat that same smaller amount for each subsegment.
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
    # strip old F
    row_noF = re.sub(r'F[-0-9.]+[.]?[0-9]*', '', row)
    row_noF = row_noF.rstrip()
    return f"{row_noF} F{feed_mm_min:.1f}\n"


def extract_feedrate(row):
    """
    Look for an F<value> token in a G-code move row and return it as float.
    If none found, return None.

    Example:
      'G1 X10.2 Y5.0 E0.123 F2400\n' -> 2400.0
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
    medium_feedrate
):
    """
    Convert transformed G-code 'data' into final printer-space toolpaths.

    Extension vs. the 'stable' version:
    - We STILL only inject slow_feedrate (F180) exactly once, at the FIRST slow subsegment,
      and we restore the previous fast feedrate once when we exit slow.
    - NEW: The *final* fast extruding subsegment immediately BEFORE we enter slow
      will be rewritten to use medium_feedrate (e.g. F400) instead of whatever fast
      feedrate it had. Only that one subsegment is modified.

      In other words, we 'pre-brake' by setting F400 on the subsegment that
      happens right before the first slow segment. No multi-line stack,
      just 1-line look-behind.
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

    # "look-behind" buffer for exactly ONE subsegment
    buffered_row = None          # text of the last subsegment we haven't flushed yet
    buffered_is_extruding = False
    buffered_is_slow = False     # whether that buffered subsegment was itself slow
    # NOTE: buffered subsegment is always already newline-terminated.

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
            out_line = clean_and_set_feedrate(out_line, medium_feedrate)

        row_list.append(out_line)

        buffered_row = None
        buffered_is_extruding = False
        buffered_is_slow = False

    def set_buffer(new_text, is_extruding, is_slow):
        """
        Overwrite the one-line buffer with a new subsegment.
        Assumes new_text already has newline.
        """
        nonlocal buffered_row, buffered_is_extruding, buffered_is_slow
        buffered_row = new_text
        buffered_is_extruding = is_extruding
        buffered_is_slow = is_slow

    for row in data:
        stripped = row.strip()

        # --- comment lines: perimeter / infill detection etc. ---
        if stripped.startswith(";"):
            lower = stripped.lower()
            prev_mode = in_perimeter

            # perimeter mode?
            if "perimeter" in lower:
                in_perimeter = True
            # infill or fill (but not perimeter) => not perimeter
            if "infill" in lower or ("fill" in lower and "perimeter" not in lower):
                in_perimeter = False

            if (in_perimeter is True) and (prev_mode is False):
                perimeter_true_count += 1

            # Before we output this comment line, we MUST flush any buffered move,
            # because comments must appear in correct chronological order.
            flush_buffer(new_data, force_medium=False)

            # comments don't change FSM for feedrate themselves.
            new_data.append(row)
            continue

        # --- lines that are NOT G0/G1 (pattern_G doesn't match) ---
        g_match = re.search(pattern_G, row)
        if g_match is None:
            # This might be "M204", "M220", "G1 F..." with no XYZE, etc.
            # That can redefine normal fast feedrate.
            fval = extract_feedrate(row)
            if fval is not None and abs(fval - slow_feedrate) > 1e-6:
                # treat this as slicer telling us a 'fast' feedrate again
                last_normal_F = fval
                was_slow = False

            # flush buffered movement first for ordering
            flush_buffer(new_data, force_medium=False)

            new_data.append(row)
            continue

        # --- G0/G1 movement command line ---

        # Grab coords
        x_match = re.search(pattern_X, row)
        y_match = re.search(pattern_Y, row)
        z_match = re.search(pattern_Z, row)

        # if it's only a feedrate move or nothing relevant
        if (x_match is None and y_match is None and z_match is None):
            fval = extract_feedrate(row)
            if fval is not None and abs(fval - slow_feedrate) > 1e-6:
                last_normal_F = fval
                was_slow = False

            flush_buffer(new_data, force_medium=False)
            new_data.append(row)
            continue

        # Update "layer Z" if slicer provided Z
        if z_match is not None:
            z_layer = float(z_match.group(0).replace('Z', ''))

        # target XY for this path
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
        base_row = insert_Z(row, z_vals[0])
        base_row = replace_E(base_row, num_segm)

        for j in range(num_segm):
            sub_x = x_vals[j + 1]
            sub_y = y_vals[j + 1]
            sub_z = z_vals[j + 1]

            # coordinate-substituted single subsegment line
            single_row = re.sub(pattern_X, 'X' + str(round(sub_x, 3)), base_row)
            single_row = re.sub(pattern_Y, 'Y' + str(round(sub_y, 3)), single_row)
            single_row = re.sub(pattern_Z, 'Z' + str(round(sub_z, 3)), single_row)

            # probe geometry for slowdown if we are currently in a perimeter region
            if in_perimeter:
                near_down = segment_near_downward_surface(
                    np.array([x_vals[j],     y_vals[j],     z_vals[j]]),
                    np.array([
                        0.5 * (x_vals[j] + x_vals[j + 1]),
                        0.5 * (y_vals[j] + y_vals[j + 1]),
                        0.5 * (z_vals[j] + z_vals[j + 1])
                    ]),
                    np.array([x_vals[j + 1], y_vals[j + 1], z_vals[j + 1]]),
                    tri_grid,
                    cell_size,
                    dist_tol=0.4
                )
                if near_down:
                    subsegments_near_downward += 1
            else:
                near_down = False

            slow_this = (in_perimeter and near_down)
            if slow_this:
                subsegments_slow_candidates += 1

            # --- FEEDRATE FSM / BUFFER LOGIK ---

            # detect extrusion on this subsegment (rough heuristic: has "E" and not E0)
            # we re-scan the final single_row for an E value:
            extruding_here = False
            mE = re.search(r'E(-?\d+\.?\d*)', single_row)
            if mE:
                try:
                    e_val = float(mE.group(1))
                    # kleine Menge > 0 => wir extrudieren Material
                    if abs(e_val) > 1e-9:
                        extruding_here = True
                except ValueError:
                    pass

            if slow_this:
                # Wir gehen in den Slow-Bereich.
                # 1) Vor dem allerersten langsamen Subsegment wollen wir
                #    GENAU EINMAL das gepufferte letzte schnelle Extrusions-
                #    subsegment mit medium_feedrate ausgeben.
                #    -> also force_medium=True beim Flush.
                if not was_slow:
                    flush_buffer(new_data, force_medium=True)
                else:
                    # wir waren schon slow, also flush_buffer normal
                    flush_buffer(new_data, force_medium=False)

                # 2) Jetzt dieses langsame Segment erzeugen:
                line_out = single_row

                # Erste Zeile im neuen Slow-Block?
                if not was_slow:
                    # setze explizit slow_feedrate (z.B. F180) EINMAL
                    line_out = clean_and_set_feedrate(line_out, slow_feedrate)
                    was_slow = True
                else:
                    # schon slow -> nicht erneut F180 reinschreiben
                    if not line_out.endswith("\n"):
                        line_out = line_out.rstrip() + "\n"

                new_data.append(line_out)

                # nach Ausgabe des slow-Segments kommt KEIN Buffer,
                # weil wir ihn schon geflusht haben und slow-Zeilen direkt schreiben
                buffered_row = None
                buffered_is_extruding = False
                buffered_is_slow = False

            else:
                # Dieses Subsegment ist NICHT slow.
                fval_here = extract_feedrate(single_row)

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
                            restored = clean_and_set_feedrate(single_row, last_normal_F)
                        else:
                            # fallback: einfach nur newline
                            restored = single_row
                            if not restored.endswith("\n"):
                                restored = restored.rstrip() + "\n"

                    # Wir SIND jetzt nicht mehr slow
                    was_slow = False

                    # Wichtig: Wir flushen NICHT sofort, weil dieses Segment kann
                    # noch der Kandidat für "medium vor nächstem slow" sein.
                    # D.h. wir legen es in den Buffer.
                    flush_buffer(new_data, force_medium=False)
                    set_buffer(restored, extruding_here, is_slow=False)

                else:
                    # wir sind im normalen schnellen Bereich geblieben
                    # (kein slow_this, was_slow == False)

                    # Wenn der Slicer hier ein F liefert, ist das unsere neue "fast" Referenz.
                    if fval_here is not None and abs(fval_here - slow_feedrate) > 1e-6:
                        last_normal_F = fval_here

                    # wir puffern diese Zeile (ersetzen evtl. alten Buffer),
                    # ABER wir flushen den alten Buffer zuerst normal,
                    # weil er definitiv NICHT mehr "direkt vor slow" sein kann.
                    flush_buffer(new_data, force_medium=False)

                    # Stelle sicher, dass newline da ist
                    if not single_row.endswith("\n"):
                        single_row = single_row.rstrip() + "\n"

                    set_buffer(single_row, extruding_here, is_slow=False)

            total_subsegments += 1

        # Update XY for Nächste G0/G1-Quelle
        x_old = x_new
        y_old = y_new

        # Ende dieses G0/G1-Blocks:
        # NICHT flush_buffer() hier — weil der letzte Subsegment dieser Zeile
        # evtl. der Kandidat für "medium vor slow" sein soll.
        # Wir flushen ihn entweder:
        #  - wenn der nächste Subsegment slow wird (mit force_medium=True),
        #  - oder wenn eine Kommentar-/M-/anderes Kommando-Zeile dazwischen kommt,
        #  - oder ganz am Ende der Funktion.

    # Ende der gesamten Datei -> flush, aber OHNE medium (es kommt ja kein slow mehr)
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
    maximal_length = 1.0,
    x_shift = 0.0,
    y_shift = 0.0,
    z_desired = 0.1,
    downward_angle_deg = 10.0,
    slow_feedrate = 180.0,
    medium_feedrate = 400.0

):
    """
    Convert SuperSlicer output in transformed coordinates (in_file)
    into final printer-space G-code.

    Parameters
    ----------
    in_file : str
        Path to G-code in transformed coordinates (gcode_tf/<part>.gcode)
    in_transform_for_interp : str
        Path to the "tf_surfaces/<part>.stl" surface that defines Z offset.
        We still use this for building 'interp' to backtransform Z.
    out_dir : str
        Output dir for final G-code (gcode_parts/)
    surface_for_slowdown : str
        Path to the FINAL geometry STL (stl_parts/<part>.stl).
        We analyze this mesh to detect downward-facing regions and slow perimeters.
    maximal_length : float
        Max XY segment length before we subdivide
    x_shift, y_shift : float
        (Currently unused in the math below; could be applied after if needed)
    z_desired : float
        Minimal Z height (z_min clamp). Your pipeline used z_desired+0.2 previously.
    downward_angle_deg : float
        Max angle from -Z for a triangle to count as "downward"
        (in FINAL geometry space)
    slow_feedrate : float
        Feedrate [mm/min] to enforce on downward-facing perimeter moves.

    Output
    ------
    Writes a final G-code file in out_dir with same basename as in_file.
    """

    start = time.time()

    # 1. Read original (transformed) G-code
    with open(in_file, 'r') as f_gcode:
        data = f_gcode.readlines()

    # 2. Build interpolation surface for Z backtransform using tf_surfaces mesh
    surf = mesh.Mesh.from_file(in_transform_for_interp).vectors
    surf_xy = np.reshape(surf[:, :, [0, 1]], (-1, 2))
    delaunay_grid = Delaunay(surf_xy)
    interp = LinearNDInterpolator(delaunay_grid,
                                  np.reshape(surf[:, :, 2], -1),
                                  0)

    # 3. Load final-part mesh (stl_parts/*.stl) and extract downward-facing triangles
    downward_triangles = triangle_data_from_mesh(
        surface_for_slowdown,
        max_angle_deg=downward_angle_deg
    )

    # build spatial index once
    tri_grid, cell_size = build_triangle_spatial_index(
        downward_triangles,
        cell_size=2.0   # you can tune this, 2.0mm is a decent start
    )

    # 4. Backtransform + slowdown logic
    data_bt = backtransform_data(
        data,
        interp,
        maximal_length,
        z_desired + 0.2,
        tri_grid,
        cell_size,
        slow_feedrate=slow_feedrate,
        medium_feedrate=medium_feedrate
    )    




    data_bt_string = ''.join(data_bt)



    # 5. Save final backtransformed code
    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.basename(in_file)  # e.g. "test_2.gcode"
    output_path = os.path.join(out_dir, file_name)

    with open(output_path, 'w', newline="\n") as f_gcode_bt:
        f_gcode_bt.write(data_bt_string)

    end = time.time()
    print('GCode generated in {:.1f}s, saved in {}'.format(end - start, output_path))



# Standalone test hook (optional)
if __name__ == "__main__":
    # Example dummy paths; adjust to something that exists in your structure
    file_path = 'gcode_tf/Body2.gcode'
    tf_surface = 'tf_surfaces/Body2.stl'
    final_part = 'stl_parts/Body2.stl'
    out_dir = 'gcode_parts'

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
