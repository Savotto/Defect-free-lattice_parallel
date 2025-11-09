"""
Utilities for merging movement batches in parallel.
"""

from typing import List, Dict, Tuple
import numpy as np

Move = Dict[str, Tuple[int, int]]
Batch = Dict[str, any]


def manhattan_path(move: Move) -> List[Tuple[int, int]]:
    """Return the Manhattan path between source and target inclusive."""
    sr, sc = move['from']
    tr, tc = move['to']
    path = [(sr, sc)]
    # Horizontal segment
    dc = 1 if tc > sc else -1
    for c in range(sc + dc, tc + dc, dc):
        path.append((sr, c))
    # Vertical segment
    dr = 1 if tr > sr else -1
    for r in range(sr + dr, tr + dr, dr):
        path.append((r, tc))
    return path


def can_parallelize_moves(field: np.ndarray, moves1: List[Move], moves2: List[Move]) -> bool:
    """
    Return True if moves1 and moves2 can be executed in parallel under all constraints.
    """
    # Gather static atoms (not moving in either batch)
    sources1 = {tuple(m['from']) for m in moves1}
    sources2 = {tuple(m['from']) for m in moves2}
    all_atoms = set(zip(*np.where(field == 1)))
    static_atoms = all_atoms - sources1 - sources2

    # Combine moves for global checks
    moves_all = moves1 + moves2

    # 1) Static-atom interior path-block
    for m in moves_all:
        sr, sc = m['from']
        tr, tc = m['to']
        if sr == tr:
            for c in range(min(sc, tc) + 1, max(sc, tc)):
                if (sr, c) in static_atoms:
                    return False
        if sc == tc:
            for r in range(min(sr, tr) + 1, max(sr, tr)):
                if (r, sc) in static_atoms:
                    return False


    # 4) Endpoint exclusivity
    srcs = {tuple(m['from']) for m in moves_all}
    tars = {tuple(m['to']) for m in moves_all}
    if len(srcs) < len(moves_all) or len(tars) < len(moves_all):
        return False

    # 5) Move-to-move path-disjointness
    paths1 = [set(manhattan_path(m)) for m in moves1]
    paths2 = [set(manhattan_path(m)) for m in moves2]
    for i, p1 in enumerate(paths1):
        for j, p2 in enumerate(paths2):
            inter = p1 & p2
            # allow shared destination
            shared = {tuple(moves1[i]['to'])} if moves1[i]['to'] == moves2[j]['to'] else set()
            if inter - shared:
                return False

    # 6) Intra-line left/right monotonicity
    nrows, ncols = field.shape
    cut_col = ncols // 2
    cut_row = nrows // 2
    # Check rows
    moves_by_row = {}
    for m in moves_all:
        r, c = m['from']
        moves_by_row.setdefault(r, []).append(m)
    for r, mlist in moves_by_row.items():
        left = sorted([m['from'][1] for m in mlist if m['from'][1] <= cut_col], reverse=True)
        if left != sorted(left, reverse=True):
            return False
        right = sorted([m['from'][1] for m in mlist if m['from'][1] > cut_col])
        if right != sorted(right):
            return False
    # Check columns
    moves_by_col = {}
    for m in moves_all:
        r, c = m['from']
        moves_by_col.setdefault(c, []).append(m)
    for c, mlist in moves_by_col.items():
        upper = sorted([m['from'][0] for m in mlist if m['from'][0] <= cut_row], reverse=True)
        if upper != sorted(upper, reverse=True):
            return False
        lower = sorted([m['from'][0] for m in mlist if m['from'][0] > cut_row])
        if lower != sorted(lower):
            return False

    # 7) Cross-row strict column-ordering
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

    # 8) Cross-column strict row-ordering
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


def merge_parallel_batches(batches: List[Batch], initial_field: np.ndarray) -> List[Batch]:
    """
    Merge sequential batches into larger parallel batches when possible.
    Prints initial and reduced batch counts.
    """
    print(f"Initial number of batches: {len(batches)}")
    merged: List[Batch] = []
    i = 0
    field_before = initial_field.copy()

    while i < len(batches):
        base = batches[i]
        moves1 = base.get('moves', []).copy()
        time = base.get('time', 0)
        end_state = base['state']
        j = i + 1

        while j < len(batches):
            cand = batches[j]
            moves2 = cand.get('moves', [])
            if can_parallelize_moves(field_before, moves1, moves2):
                moves1.extend(moves2)
                time = max(time, cand.get('time', 0))
                end_state = cand['state']
                j += 1
            else:
                break

        merged.append({'type': 'parallel_merge', 'moves': moves1, 'state': end_state, 'time': time})
        field_before = end_state.copy()
        i = j

    print(f"Reduced to {len(merged)} parallel batches.")
    return merged