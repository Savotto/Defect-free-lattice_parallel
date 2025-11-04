"""
parallel_utils.py

Utilities for merging movement batches in parallel.
Implements global parallelization respecting AOD physics constraints.
Used for achieving improved move scaling (~N^0.472).
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
    if tc != sc:
        dc = 1 if tc > sc else -1
        for c in range(sc + dc, tc + dc, dc):
            path.append((sr, c))
    # Vertical segment
    if tr != sr:
        dr = 1 if tr > sr else -1
        for r in range(sr + dr, tr + dr, dr):
            path.append((r, tc))
    return path


def can_parallelize_moves(field: np.ndarray, moves1: List[Move], moves2: List[Move]) -> bool:
    """
    Return True if moves1 and moves2 can be executed in parallel under all AOD restrictions.
    """

    # Gather static atoms (not moving in either batch)
    sources1 = {tuple(m['from']) for m in moves1}
    sources2 = {tuple(m['from']) for m in moves2}
    all_atoms = set(zip(*np.where(field == 1)))
    static_atoms = all_atoms - sources1 - sources2

    moves_all = moves1 + moves2

    # 1) Static atom blocking within the same line
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

    # 2) Cross-row/column no-passing vs statics in other active lines
    statics_by_row = {}
    statics_by_col = {}
    for (r_nm, c_nm) in static_atoms:
        statics_by_row.setdefault(r_nm, set()).add(c_nm)
        statics_by_col.setdefault(c_nm, set()).add(r_nm)

    rows_in_step = {m['from'][0] for m in moves_all}
    cols_in_step = {m['from'][1] for m in moves_all}

    # Horizontal moves cannot cross static columns in other active rows
    for m in moves_all:
        (sr, sc), (tr, tc) = m['from'], m['to']
        if sr == tr:
            lo, hi = sorted((sc, tc))
            for r_other in rows_in_step - {sr}:
                for c_nm in statics_by_row.get(r_other, ()):
                    if lo < c_nm < hi:
                        return False
    # Vertical moves cannot cross static rows in other active columns
    for m in moves_all:
        (sr, sc), (tr, tc) = m['from'], m['to']
        if sc == tc:
            lo, hi = sorted((sr, tr))
            for c_other in cols_in_step - {sc}:
                for r_nm in statics_by_col.get(c_other, ()):
                    if lo < r_nm < hi:
                        return False

    # 3) Unique starting and ending positions
    srcs = {tuple(m['from']) for m in moves_all}
    tars = {tuple(m['to']) for m in moves_all}
    if len(srcs) < len(moves_all) or len(tars) < len(moves_all):
        return False

    # 4) No path intersection (excluding identical target)
    paths1 = [set(manhattan_path(m)) for m in moves1]
    paths2 = [set(manhattan_path(m)) for m in moves2]
    for i, p1 in enumerate(paths1):
        for j, p2 in enumerate(paths2):
            inter = p1 & p2
            shared = {tuple(moves1[i]['to'])} if moves1[i]['to'] == moves2[j]['to'] else set()
            if inter - shared:
                return False

    # 5) Strict ordering within same line
    for i in range(len(moves_all)):
        r1s, c1s = moves_all[i]['from']
        r1t, c1t = moves_all[i]['to']
        for j in range(i + 1, len(moves_all)):
            r2s, c2s = moves_all[j]['from']
            r2t, c2t = moves_all[j]['to']
            # Same row → preserve column order
            if r1s == r2s:
                if c1s < c2s and not (c1t <= c2t): return False
                if c1s > c2s and not (c1t >= c2t): return False
                if c1s == c2s and c1t != c2t: return False
            # Same column → preserve row order
            if c1s == c2s:
                if r1s < r2s and not (r1t <= r2t): return False
                if r1s > r2s and not (r1t >= r2t): return False
                if r1s == r2s and r1t != r2t: return False

    # 6) Strict ordering across lines
    for m1 in moves1:
        r1, c1 = m1['from']; r1t, c1t = m1['to']
        for m2 in moves2:
            r2, c2 = m2['from']; r2t, c2t = m2['to']
            # Across rows → column monotonicity
            if r1 != r2:
                if c1 > c2 and c1t <= c2t: return False
                if c1 < c2 and c1t >= c2t: return False
                if c1 == c2 and c1t != c2t: return False
            # Across columns → row monotonicity
            if c1 != c2:
                if r1 > r2 and r1t <= r2t: return False
                if r1 < r2 and r1t >= r2t: return False
                if r1 == r2 and r1t != r2t: return False

    return True


def merge_parallel_batches(batches: List[Batch], initial_field: np.ndarray) -> List[Batch]:
    """
    Globally merge sequential batches into larger parallel super-steps when possible.
    Prints initial and reduced batch counts.
    """
    print(f"Initial number of batches: {len(batches)}")
    merged: List[Batch] = []
    used = set()
    field_before = initial_field.copy()

    n = len(batches)
    idx = 0
    while len(used) < n:
        # Find next unused batch
        while idx < n and idx in used:
            idx += 1
        if idx >= n:
            break
        seed = idx
        used.add(seed)

        group_indices = [seed]
        moves_union: List[Move] = list(batches[seed].get('moves', []))
        end_state = batches[seed]['state']
        time = batches[seed].get('time', 0)

        # Try to add compatible batches
        added = True
        while added:
            added = False
            for j in range(n):
                if j in used:
                    continue
                cand_moves = batches[j].get('moves', [])
                if can_parallelize_moves(field_before, moves_union, cand_moves):
                    moves_union.extend(cand_moves)
                    time = max(time, batches[j].get('time', 0))
                    end_state = batches[j]['state']
                    used.add(j)
                    group_indices.append(j)
                    added = True

        merged.append({
            'type': 'parallel_merge',
            'moves': moves_union,
            'state': end_state,
            'time': time,
            'indices': group_indices
        })

    print(f"Reduced to {len(merged)} parallel batches.")
    return merged
