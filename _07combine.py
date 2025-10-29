import os

def _find_payload_start_index(lines):
    """
    For a non-first G-code file:
    We want to skip that file's warmup/header and jump into "real printing".

    Heuristic:
    1. Collect all indices of lines that start with ';TYPE:'.
    2. Prefer the first ';TYPE:' line that is NOT ';TYPE:Custom'
       (because ';TYPE:Custom' is usually priming / temp / wipe towers etc.).
    3. If we don't have anything except ';TYPE:Custom', fall back to that.
    4. If there is no ';TYPE:' at all (weird slice), fall back to 0.

    Returns:
        start_idx (int): index of the chosen ';TYPE:' line in `lines`.
    """

    type_all = []
    type_non_custom = []

    for idx, row in enumerate(lines):
        if row.startswith(";TYPE:"):
            type_all.append(idx)
            if not row.startswith(";TYPE:Custom"):
                type_non_custom.append(idx)

    if type_non_custom:
        return type_non_custom[0]

    if type_all:
        return type_all[0]

    return 0


def _extract_print_payload(lines, is_first_file, context_lines_before_type=2):
    """
    For the first file:
        -> return the whole file unchanged (we keep homing, temps, purge line, etc.).

    For every later file:
        -> start around the first relevant ';TYPE:' block, skipping the file's own
           startup routine.

        BUT: we include a small "context prefix" before ';TYPE:' (default 2 lines).
        That's important because right before ';TYPE:Perimeter' or ';TYPE:Solid infill'
        you often get positioning moves like:
            G1 Z10.2
            G1 E0.7 F2100
        which are needed for correct Z and extrusion state.
    """

    if is_first_file:
        return lines[:]

    type_idx = _find_payload_start_index(lines)

    # include a few lines BEFORE that type marker to keep Z positioning etc.
    start_idx = max(type_idx - context_lines_before_type, 0)

    return lines[start_idx:]


def combineGCode(in_folder, out_file, context_lines_before_type=2):
    """
    Combine multiple per-part G-code files into one final program.

    Rules:
    - Sort all *.gcode files in in_folder.
    - For the first file:
        write the entire file (includes start G-code, temps, homing).
    - For subsequent files:
        strip away their own startup header and temps, and only append
        the actual "print moves", starting just BEFORE the first ';TYPE:' block.

      We annotate each appended block with nice comments so you can debug.

    The `context_lines_before_type` argument controls how many lines
    before ';TYPE:' we also keep when appending later segments.
    """

    in_folder_abs = os.path.abspath(in_folder)
    out_file_abs  = os.path.abspath(out_file)

    print("[combineGCode] Combining G-code from:", in_folder_abs)
    print("[combineGCode] Target:", out_file_abs)

    # 1. collect all .gcode files
    gcode_files = [
        f for f in os.listdir(in_folder_abs)
        if f.lower().endswith(".gcode") and not f.startswith(".")
    ]
    gcode_files.sort()

    print("[combineGCode] Order:", gcode_files)

    # 2. stitch them
    with open(out_file_abs, "w", newline="\n") as fout:
        for idx, fname in enumerate(gcode_files):
            full_path = os.path.join(in_folder_abs, fname)
            print(f"[combineGCode]   + appending {full_path}")

            with open(full_path, "r") as fin:
                lines = fin.readlines()

            payload = _extract_print_payload(
                lines,
                is_first_file=(idx == 0),
                context_lines_before_type=context_lines_before_type,
            )

            # helpful separators in the merged file
            fout.write(f"; --- START OF SEGMENT {fname} ---\n")
            for line in payload:
                fout.write(line)
            fout.write(f"; --- END OF SEGMENT {fname} ---\n")

    print("[combineGCode] Done:", out_file_abs)