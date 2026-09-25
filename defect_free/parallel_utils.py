"""
Utilities for merging movement batches in parallel.
"""

from typing import Any, Dict, List, Set, Tuple
import numpy as np

Move = Dict[str, Tuple[int, int]]
Batch = Dict[str, Any]


def manhattan_path_set(move: Move) -> Set[Tuple[int, int]]:
    """Return the Manhattan path between source and target inclusive."""
    sr, sc = move['from']
    tr, tc = move['to']
    path = {(sr, sc)}
    # Horizontal segment
    if sc != tc:
        dc = 1 if tc > sc else -1
        for c in range(sc + dc, tc + dc, dc):
            path.add((sr, c))
    # Vertical segment
    if sr != tr:
        dr = 1 if tr > sr else -1
        for r in range(sr + dr, tr + dr, dr):
            path.add((r, tc))
    return path


def build_batch_metadata(moves: List[Move]) -> Dict[str, Any]:
    """Precompute reusable geometric metadata for a move batch."""
    sources = {tuple(move['from']) for move in moves}
    targets = {tuple(move['to']) for move in moves}
    path_union: Set[Tuple[int, int]] = set()
    for move in moves:
        path_union.update(manhattan_path_set(move))

    rows_with_horizontal_moves = set()
    cols_with_vertical_moves = set()
    active_cols: Dict[int, Set[int]] = {}
    active_rows: Dict[int, Set[int]] = {}

    for move in moves:
        sr, sc = move['from']
        tr, tc = move['to']
        if sr == tr:
            rows_with_horizontal_moves.add(sr)
            active_rows.setdefault(sr, set()).update(range(min(sc, tc), max(sc, tc) + 1))
        if sc == tc:
            cols_with_vertical_moves.add(sc)
            active_cols.setdefault(sc, set()).update(range(min(sr, tr), max(sr, tr) + 1))

    return {
        'moves': moves,
        'sources': sources,
        'targets': targets,
        'path_union': path_union,
        'rows_with_horizontal_moves': rows_with_horizontal_moves,
        'cols_with_vertical_moves': cols_with_vertical_moves,
        'active_cols': active_cols,
        'active_rows': active_rows,
    }


def merge_batch_metadata_in_place(meta1: Dict[str, Any], meta2: Dict[str, Any]) -> None:
    """Merge meta2 into meta1 after a successful parallelization check."""
    meta1['moves'].extend(meta2['moves'])
    meta1['sources'].update(meta2['sources'])
    meta1['targets'].update(meta2['targets'])
    meta1['path_union'].update(meta2['path_union'])
    meta1['rows_with_horizontal_moves'].update(meta2['rows_with_horizontal_moves'])
    meta1['cols_with_vertical_moves'].update(meta2['cols_with_vertical_moves'])

    for col, rows in meta2['active_cols'].items():
        meta1['active_cols'].setdefault(col, set()).update(rows)

    for row, cols in meta2['active_rows'].items():
        meta1['active_rows'].setdefault(row, set()).update(cols)


def can_parallelize_moves(
    field: np.ndarray,
    meta1: Dict[str, Any],
    meta2: Dict[str, Any],
) -> bool:
    """Return True if two batches can be executed in parallel under all constraints."""
    field_shape = field.shape
    sources_all = meta1['sources'] | meta2['sources']
    moves1 = meta1['moves']
    moves2 = meta2['moves']
    moves_all = moves1 + moves2

    def is_static(row: int, col: int) -> bool:
        return field[row, col] == 1 and (row, col) not in sources_all

    if len(sources_all) < len(moves_all):
        return False
    if len(meta1['targets'] | meta2['targets']) < len(moves_all):
        return False

    # 1) Static-atom interior path-block
    for m in moves_all:
        sr, sc = m['from']
        tr, tc = m['to']
        if sr == tr:
            for c in range(min(sc, tc) + 1, max(sc, tc)):
                if is_static(sr, c):
                    return False
        if sc == tc:
            for r in range(min(sr, tr) + 1, max(sr, tr)):
                if is_static(r, sc):
                    return False

    # 2) Cross-trap prevention with static atoms
    rows_with_horizontal_moves = meta1['rows_with_horizontal_moves'] | meta2['rows_with_horizontal_moves']
    cols_with_vertical_moves = meta1['cols_with_vertical_moves'] | meta2['cols_with_vertical_moves']
    
    # Check for cross-trap scenarios
    for row in rows_with_horizontal_moves:
        for col in cols_with_vertical_moves:
            if is_static(row, col):
                return False

    # Vertical x Vertical static-atom row blocking
    # If a static atom sits at (r, c_static) on an active column c_static,
    # no other column may have ANY move touching row r.
    active_cols = meta1['active_cols'].copy()
    for col, rows in meta2['active_cols'].items():
        active_cols.setdefault(col, set()).update(rows)

    static_rows_by_active_col: Dict[int, Set[int]] = {}
    for col in active_cols:
        static_rows = set()
        for row in range(field_shape[0]):
            if is_static(row, col):
                static_rows.add(row)
        static_rows_by_active_col[col] = static_rows

    for static_col, static_rows in static_rows_by_active_col.items():
        if not static_rows:
            continue
        for other_col, rows in active_cols.items():
            if other_col == static_col:
                continue
            if static_rows & rows:
                return False

    # Horizontal x Horizontal static-atom column blocking
    active_rows = meta1['active_rows'].copy()
    for row, cols in meta2['active_rows'].items():
        active_rows.setdefault(row, set()).update(cols)

    static_cols_by_active_row: Dict[int, Set[int]] = {}
    for row in active_rows:
        static_cols = set()
        for col in range(field_shape[1]):
            if is_static(row, col):
                static_cols.add(col)
        static_cols_by_active_row[row] = static_cols

    for static_row, static_cols in static_cols_by_active_row.items():
        if not static_cols:
            continue
        for other_row, cols in active_rows.items():
            if other_row == static_row:
                continue
            if static_cols & cols:
                return False

    # 5) Move-to-move path-disjointness
    if meta1['path_union'] & meta2['path_union']:
        return False

    # 6) Cross-row strict column-ordering
    for m1 in moves1:
        r1, c1 = m1['from']; _, c1t = m1['to']
        for m2 in moves2:
            r2, c2 = m2['from']; _, c2t = m2['to']
            if r1 != r2:
                if c1 > c2 and c1t <= c2t:
                    return False
                if c1 < c2 and c1t >= c2t:
                    return False
                if c1 == c2 and c1t != c2t:
                    return False

    # 7) Cross-column strict row-ordering
    for m1 in moves1:
        r1, c1 = m1['from']; r1t, _ = m1['to']
        for m2 in moves2:
            r2, c2 = m2['from']; r2t, _ = m2['to']
            if c1 != c2:
                if r1 > r2 and r1t <= r2t:
                    return False
                if r1 < r2 and r1t >= r2t:
                    return False
                if r1 == r2 and r1t != r2t:
                    return False

    return True


def merge_parallel_batches(
    batches: List[Batch],
    initial_field: np.ndarray,
    policy: str = "phase_aware",
) -> List[Batch]:
    """
    Merge sequential batches into larger parallel batches when possible.
    Prints initial and reduced batch counts.
    """
    print(f"Initial number of batches: {len(batches)}")

    def _legacy_merge_contiguous(seq: List[Batch], start_field: np.ndarray) -> List[Batch]:
        merged_local: List[Batch] = []
        i = 0
        field_before = start_field
        batch_metadata = [build_batch_metadata(batch.get('moves', [])) for batch in seq]

        while i < len(seq):
            base = seq[i]
            moves1 = base.get('moves', []).copy()
            time = base.get('time', 0)
            end_state = base.get('state')
            base_meta = batch_metadata[i]
            j = i + 1

            while j < len(seq):
                cand = seq[j]
                moves2 = cand.get('moves', [])
                cand_meta = batch_metadata[j]
                if can_parallelize_moves(field_before, base_meta, cand_meta):
                    moves1.extend(moves2)
                    merge_batch_metadata_in_place(base_meta, cand_meta)
                    time = max(time, cand.get('time', 0))
                    end_state = cand.get('state')
                    j += 1
                else:
                    break

            merged_local.append({'type': 'parallel_merge', 'moves': moves1, 'state': end_state, 'time': time})
            if end_state is not None:
                field_before = end_state
            i = j
        return merged_local

    if policy not in {"phase_aware", "contiguous"}:
        raise ValueError(f"Unknown batch merge policy: {policy!r}")

    # The submitted paper used one linear, contiguous greedy pass over the
    # complete movement sequence. Later experiments introduced phase-aware
    # first-fit packing; keep that behavior opt-in through the policy argument.
    if policy == "contiguous" or not any(batch.get('phase') is not None for batch in batches):
        merged = _legacy_merge_contiguous(batches, initial_field)
        print(f"Reduced to {len(merged)} parallel batches.")
        return merged

    # Phase-aware merge: within each phase, greedily group all compatible batches
    # (not just contiguous ones). Phase ordering is preserved.
    merged: List[Batch] = []
    field_before_phase = initial_field
    i = 0
    while i < len(batches):
        phase_id = batches[i].get('phase')
        phase_group: List[Batch] = []
        while i < len(batches) and batches[i].get('phase') == phase_id:
            phase_group.append(batches[i])
            i += 1

        # If the phase is missing/empty, keep legacy behavior for that segment.
        if phase_id is None or not phase_group:
            merged.extend(_legacy_merge_contiguous(phase_group, field_before_phase))
            if phase_group and phase_group[-1].get('state') is not None:
                field_before_phase = phase_group[-1]['state']
            continue

        # Greedy first-fit packing by compatibility against the same phase-start field.
        bins: List[Dict[str, Any]] = []
        for batch in phase_group:
            cand_meta = build_batch_metadata(batch.get('moves', []))
            placed = False
            for b in bins:
                if can_parallelize_moves(field_before_phase, b['meta'], cand_meta):
                    merge_batch_metadata_in_place(b['meta'], cand_meta)
                    b['time'] = max(b['time'], batch.get('time', 0))
                    b['state'] = batch.get('state')
                    placed = True
                    break
            if not placed:
                bins.append(
                    {
                        'meta': cand_meta,
                        'time': batch.get('time', 0),
                        'state': batch.get('state'),
                    }
                )

        for b in bins:
            merged.append(
                {
                    'type': 'parallel_merge',
                    'phase': phase_id,
                    'moves': b['meta']['moves'],
                    'state': b['state'],
                    'time': b['time'],
                }
            )

        if phase_group[-1].get('state') is not None:
            field_before_phase = phase_group[-1]['state']

    print(f"Reduced to {len(merged)} parallel batches.")
    return merged
