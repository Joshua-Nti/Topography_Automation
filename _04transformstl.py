# _04transformstl.py (NEU)
import os
import time
import numpy as np
from stl import mesh
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator


def transformSTL(in_body, in_transform, out_dir):
    """
    Deform (lift/warp) an STL mesh 'in_body' using a reference surface 'in_transform'
    and save the transformed mesh in 'out_dir' (same base filename).
    """
    start = time.time()
    print("[transformSTL] START")
    print(f"[transformSTL]   in_body      = {in_body}")
    print(f"[transformSTL]   in_transform = {in_transform}")
    print(f"[transformSTL]   out_dir      = {out_dir}")

    # --- Load meshes --------------------------------------------------------
    body_mesh = mesh.Mesh.from_file(in_body)
    body_vecs = body_mesh.vectors
    body_x = body_vecs[:, :, 0].reshape(-1)
    body_y = body_vecs[:, :, 1].reshape(-1)
    body_z = body_vecs[:, :, 2].reshape(-1)
    print(f"[transformSTL]   body triangles: {body_vecs.shape[0]}")

    surf_mesh = mesh.Mesh.from_file(in_transform)
    surf_vecs = surf_mesh.vectors
    print(f"[transformSTL]   surface triangles: {surf_vecs.shape[0]}")

    surf_xy = surf_vecs[:, :, [0, 1]].reshape(-1, 2)
    surf_z = surf_vecs[:, :, 2].reshape(-1)
    delaunay_grid = Delaunay(surf_xy)
    interp = LinearNDInterpolator(delaunay_grid, surf_z, 0.0)

    z_offset = interp(body_x, body_y)
    if np.isnan(z_offset).any():
        print("[transformSTL]   WARNING: NaN in z_offset -> clamping to 0.0")
        z_offset = np.where(np.isnan(z_offset), 0.0, z_offset)

    new_z = body_z + z_offset
    new_xyz = np.column_stack([body_x, body_y, new_z])
    new_triangles = new_xyz.reshape(-1, 3, 3)

    new_mesh_raw = np.zeros(new_triangles.shape[0], dtype=mesh.Mesh.dtype)
    new_mesh_raw["vectors"] = new_triangles
    new_mesh = mesh.Mesh(new_mesh_raw)

    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.basename(in_body)
    output_path = os.path.join(out_dir, file_name)
    new_mesh.save(output_path)

    end = time.time()
    print(f"[transformSTL]   WROTE: {output_path}")
    print(f"[transformSTL]   DONE in {end - start:.2f}s")
    print("[transformSTL] END\n")
    return output_path


# ---------------------------------------------------------------------------
# NEW: Wrapper function to transform only selected segments from cuts.txt
# ---------------------------------------------------------------------------
def transformSTL_for_segments(segment_ids, base_body_dir="stl_parts", base_tf_dir="tf_surfaces", out_dir="stl_tf"):
    """
    Transform STL parts for the given list of segment IDs (as strings).

    Parameters
    ----------
    segment_ids : list[str]
        Segment numbers to transform (e.g. ['2', '3', '5'])
    base_body_dir : str
        Directory with the STL segments to transform.
    base_tf_dir : str
        Directory containing the corresponding transform surfaces.
    out_dir : str
        Output directory for transformed STLs.
    """
    print("[transformSTL_for_segments] START")
    os.makedirs(out_dir, exist_ok=True)

    for seg_id in segment_ids:
        body_name = f"segment_{seg_id}.stl"
        body_path = os.path.join(base_body_dir, body_name)
        tf_path = os.path.join(base_tf_dir, body_name)

        if not os.path.exists(body_path):
            print(f"[transformSTL_for_segments] WARNING: body file not found: {body_path}")
            continue
        if not os.path.exists(tf_path):
            print(f"[transformSTL_for_segments] WARNING: transform surface not found: {tf_path}")
            continue

        transformSTL(body_path, tf_path, out_dir)

    print("[transformSTL_for_segments] DONE\n")


# ---------------------------------------------------------------------------
# Standalone test (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: transform only segments 2 and 4
    test_segments = ["2", "4"]
    transformSTL_for_segments(test_segments)
