# _04transformstl.py
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

    The idea:
    - We treat the "transform surface" as a height field z = f(x, y).
    - For every vertex (x, y, z) in the input body mesh, we compute z_offset = f(x,y).
    - We then build a new mesh with z' = z + z_offset (nonplanar lifting).
    
    Why this version exists:
    - The original code used manual string slicing with '/' and plain string concat
      to build file paths. That breaks on Windows.
    - This version uses os.path functions everywhere so it's portable.

    Parameters
    ----------
    in_body : str
        Path to the STL to be transformed (e.g. "<root>/stl_parts/test_2.stl").
    in_transform : str
        Path to the transform surface STL that defines the Z offset
        (e.g. "<root>/tf_surfaces/test_2.stl").
        Must cover the XY footprint of in_body, at least approximately.
    out_dir : str
        Directory where the transformed STL should be written
        (e.g. "<root>/stl_tf/").

    Returns
    -------
    output_path : str
        Absolute path of the written transformed STL.
        (Useful for debugging and for callers to verify existence.)

    Notes
    -----
    - If (x,y) from the body lies outside the convex hull of the transform
      surface XY, then LinearNDInterpolator(...) returns NaN. We replace NaN
      with offset 0.0 so we don't inject NaNs into the STL.
    - We also create out_dir if it doesn't exist, using os.makedirs(..., exist_ok=True).
    """

    start = time.time()

    # --- Debug info ---------------------------------------------------------
    print("[transformSTL] START")
    print(f"[transformSTL]   in_body      = {in_body}")
    print(f"[transformSTL]   in_transform = {in_transform}")
    print(f"[transformSTL]   out_dir      = {out_dir}")

    # --- Load the STL we want to transform ---------------------------------
    # body_mesh.vectors.shape = (N, 3, 3), N triangles, each with 3 (x,y,z) vertices
    body_mesh = mesh.Mesh.from_file(in_body)
    body_vecs = body_mesh.vectors  # (N,3,3)

    # Flatten xyz for interpolation
    body_x = body_vecs[:, :, 0].reshape(-1)  # shape (N*3,)
    body_y = body_vecs[:, :, 1].reshape(-1)
    body_z = body_vecs[:, :, 2].reshape(-1)

    print(f"[transformSTL]   body triangles: {body_vecs.shape[0]}")

    # --- Load the transform surface ----------------------------------------
    # This STL defines z = f(x,y). We build an interpolator from it.
    surf_mesh = mesh.Mesh.from_file(in_transform)
    surf_vecs = surf_mesh.vectors  # (M,3,3)
    print(f"[transformSTL]   surface triangles: {surf_vecs.shape[0]}")

    # Build XY grid and Z samples for interpolator
    # surf_vecs[:,:,0] -> x coords of each vertex
    # surf_vecs[:,:,1] -> y coords
    # surf_vecs[:,:,2] -> z coords
    surf_xy = surf_vecs[:, :, [0, 1]].reshape(-1, 2)    # (M*3, 2)
    surf_z  = surf_vecs[:, :, 2].reshape(-1)            # (M*3,)

    # Delaunay triangulation in XY-plane
    delaunay_grid = Delaunay(surf_xy)

    # Linear interpolation in 2D: given (x,y) -> estimated Z offset
    interp = LinearNDInterpolator(
        delaunay_grid,
        surf_z,
        0.0,  # default fill value for points outside convex hull
    )

    # --- Compute Z offset for every vertex of the body ----------------------
    z_offset = interp(body_x, body_y)
    # Replace any NaN with 0.0 to avoid contaminating geometry
    if np.isnan(z_offset).any():
        print("[transformSTL]   WARNING: NaN in z_offset -> clamping to 0.0")
        z_offset = np.where(np.isnan(z_offset), 0.0, z_offset)

    # Apply the lifting
    new_z = body_z + z_offset

    # Stack back into (x,y,z)
    new_xyz = np.column_stack([body_x, body_y, new_z])  # shape (N*3, 3)

    # Reshape to triangles again: (N, 3, 3)
    new_triangles = new_xyz.reshape(-1, 3, 3)

    # --- Create a new mesh object for saving --------------------------------
    new_mesh_raw = np.zeros(new_triangles.shape[0], dtype=mesh.Mesh.dtype)
    new_mesh_raw["vectors"] = new_triangles
    new_mesh = mesh.Mesh(new_mesh_raw)

    # --- Prepare output path in a cross-platform way ------------------------
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Get the clean base filename ("test_2.stl" from ".../stl_parts/test_2.stl")
    file_name = os.path.basename(in_body)

    # Final output path
    output_path = os.path.join(out_dir, file_name)

    # --- Save the transformed mesh -----------------------------------------
    new_mesh.save(output_path)

    end = time.time()
    print(f"[transformSTL]   WROTE: {output_path}")
    print(f"[transformSTL]   DONE in {end - start:.2f}s")
    print("[transformSTL] END\n")

    return output_path


# ---------------------------------------------------------------------------
# Standalone test hook (optional manual test)
# This is mostly for quick debugging if you run this script directly,
# but in the pipeline we normally call transformSTL(...) from _00all.py.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example dummy paths; adjust or ignore.
    example_in_body      = os.path.join("stl_parts", "Body2.stl")
    example_in_transform = os.path.join("tf_surfaces", "Body2.stl")
    example_out_dir      = "stl_tf"

    transformSTL(
        in_body=example_in_body,
        in_transform=example_in_transform,
        out_dir=example_out_dir,
    )