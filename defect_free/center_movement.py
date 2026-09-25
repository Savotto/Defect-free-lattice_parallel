"""
Center-based movement module for atom rearrangement in optical lattices.
Implements strategies that place the target zone in the center of the field.
"""
import numpy as np
import time
from defect_free.parallel_utils import merge_parallel_batches
from defect_free.base_movement import BaseMovementManager

class CenterMovementManager(BaseMovementManager):
    """
    Implements center-based movement strategies for atom rearrangement.
    These strategies place the target zone in the center of the field and move atoms accordingly.
    """
    
    def _resolve_square_target_region(self, side_length: int):
        """Pick a target square near the center, optionally favoring denser windows."""
        field_height, field_width = self.simulator.initial_size
        center_row = (field_height - side_length) // 2
        center_col = (field_width - side_length) // 2

        constraints = self.simulator.constraints
        occupancy_threshold = float(
            constraints.get("low_occupancy_adaptive_target_threshold", 0.55)
        )
        search_radius = int(
            constraints.get("low_occupancy_adaptive_target_search_radius", 10)
        )
        min_gain = int(
            constraints.get("low_occupancy_adaptive_target_min_gain", 4)
        )
        adaptive_enabled = bool(
            constraints.get("low_occupancy_adaptive_target_enabled", False)
        )

        if (
            not adaptive_enabled
            or self.simulator.target_shape != "square"
            or self.simulator.occupation_prob > occupancy_threshold
            or search_radius <= 0
        ):
            return (
                center_row,
                center_col,
                center_row + side_length,
                center_col + side_length,
            )

        field = self.simulator.field
        if field is None or side_length <= 0:
            return (
                center_row,
                center_col,
                center_row + side_length,
                center_col + side_length,
            )

        max_row_start = field_height - side_length
        max_col_start = field_width - side_length
        row_range = range(
            max(0, center_row - search_radius),
            min(max_row_start, center_row + search_radius) + 1,
        )
        col_range = range(
            max(0, center_col - search_radius),
            min(max_col_start, center_col + search_radius) + 1,
        )

        prefix = field.cumsum(axis=0).cumsum(axis=1)

        def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
            total = int(prefix[r1 - 1, c1 - 1])
            if r0 > 0:
                total -= int(prefix[r0 - 1, c1 - 1])
            if c0 > 0:
                total -= int(prefix[r1 - 1, c0 - 1])
            if r0 > 0 and c0 > 0:
                total += int(prefix[r0 - 1, c0 - 1])
            return total

        centered_atoms = rect_sum(
            center_row,
            center_col,
            center_row + side_length,
            center_col + side_length,
        )
        best_region = (
            center_row,
            center_col,
            center_row + side_length,
            center_col + side_length,
        )
        best_atoms = centered_atoms
        best_distance = 0

        for row_start in row_range:
            for col_start in col_range:
                atoms = rect_sum(
                    row_start,
                    col_start,
                    row_start + side_length,
                    col_start + side_length,
                )
                distance = abs(row_start - center_row) + abs(col_start - center_col)
                if atoms > best_atoms or (atoms == best_atoms and distance < best_distance):
                    best_atoms = atoms
                    best_distance = distance
                    best_region = (
                        row_start,
                        col_start,
                        row_start + side_length,
                        col_start + side_length,
                    )

        if best_atoms - centered_atoms < min_gain:
            return (
                center_row,
                center_col,
                center_row + side_length,
                center_col + side_length,
            )

        return best_region

    def initialize_target_region(self):
        """Calculate and initialize the center-based target region."""
        if self.target_region is not None:
            return  # Already initialized
            
        side_length = self.simulator.side_length
        start_row, start_col, end_row, end_col = self._resolve_square_target_region(side_length)

        if self.simulator.target_shape != "square":
            raise ValueError("The paper release supports square targets only.")

        self.target_region = (start_row, start_col, end_row, end_col)
        self.target_mask = np.zeros(self.simulator.field.shape, dtype=bool)
        self.target_mask[start_row:end_row, start_col:end_col] = True
        self.simulator.target_mask = self.target_mask.copy()
    
    def center_atoms_in_line(
        self,
        line_idx,
        is_row,
        target_start_idx,
        target_end_idx,
        use_split_assignment_policy=False,
    ):
        """
        Fill mask-approved target sites in a single row or column.

        Target sites are taken from `self.target_mask` inside
        [target_start_idx, target_end_idx). Source sites are any occupied sites in
        the whole line.

        When `use_split_assignment_policy` is True, sources are split by count into
        left/right groups (new policy). Otherwise legacy lattice-side splitting is
        used.
        
        Args:
            line_idx: Row or column index to process
            is_row: True if processing a row, False if processing a column
            target_start_idx: Starting index of target region in the relevant dimension
            target_end_idx: Ending index of target region in the relevant dimension
        
        Returns:
            Number of atoms successfully moved
        """
        if is_row:
            line = self.simulator.field[line_idx, :].tolist()
            target_mask_line = self.target_mask[line_idx, :]
        else:
            line = self.simulator.field[:, line_idx].tolist()
            target_mask_line = self.target_mask[:, line_idx]

        target_indices = [
            idx for idx in range(target_start_idx, target_end_idx)
            if bool(target_mask_line[idx])
        ]
        if not target_indices:
            return 0

        center_idx = (target_start_idx + target_end_idx) // 2
        left_targets = [idx for idx in target_indices if idx < center_idx]
        right_targets = [idx for idx in target_indices if idx >= center_idx]

        # Build a working 1D line and collect all moves first, then execute in one batch.
        working_line = line.copy()
        all_moves = []
        max_distance = 0

        def path_is_clear(from_idx, to_idx):
            start_idx = min(from_idx, to_idx) + 1
            end_idx = max(from_idx, to_idx)
            return all(working_line[idx] == 0 for idx in range(start_idx, end_idx))

        if use_split_assignment_policy:
            source_indices = [idx for idx, occupied in enumerate(line) if occupied]
            fill_count = min(len(source_indices), len(target_indices))
            if fill_count == 0:
                return 0

            left_quota = min(len(left_targets), fill_count // 2)
            right_quota = min(len(right_targets), fill_count - left_quota)

            assigned = left_quota + right_quota
            if assigned < fill_count:
                remaining = fill_count - assigned
                extra_left = min(len(left_targets) - left_quota, remaining)
                left_quota += extra_left
                remaining -= extra_left
                if remaining > 0:
                    extra_right = min(len(right_targets) - right_quota, remaining)
                    right_quota += extra_right

            left_sources = source_indices[:left_quota]
            right_sources = source_indices[left_quota:left_quota + right_quota]
            left_destinations = left_targets[-left_quota:] if left_quota > 0 else []
            right_destinations = right_targets[:right_quota] if right_quota > 0 else []

            pending = list(zip(left_sources, left_destinations)) + list(zip(right_sources, right_destinations))
            while pending:
                moved_any = False
                next_pending = []
                for source_idx, target_idx in pending:
                    if source_idx == target_idx:
                        continue
                    if working_line[source_idx] == 0:
                        continue
                    if working_line[target_idx] == 1:
                        next_pending.append((source_idx, target_idx))
                        continue
                    if not path_is_clear(source_idx, target_idx):
                        next_pending.append((source_idx, target_idx))
                        continue

                    from_pos = (line_idx, source_idx) if is_row else (source_idx, line_idx)
                    to_pos = (line_idx, target_idx) if is_row else (target_idx, line_idx)
                    all_moves.append({'from': from_pos, 'to': to_pos})

                    working_line[source_idx] = 0
                    working_line[target_idx] = 1

                    distance = abs(target_idx - source_idx)
                    max_distance = max(max_distance, distance)
                    moved_any = True

                if not moved_any:
                    break
                pending = next_pending
        else:
            atom_indices = [idx for idx, occupied in enumerate(line) if occupied]
            left_atoms = sorted([idx for idx in atom_indices if idx < center_idx])
            right_atoms = sorted([idx for idx in atom_indices if idx >= center_idx])

            for target_idx in sorted(left_targets, reverse=True):
                if working_line[target_idx] == 1:
                    continue
                chosen = None
                for atom_idx in reversed(left_atoms):
                    if atom_idx >= target_idx:
                        continue
                    if path_is_clear(atom_idx, target_idx):
                        chosen = atom_idx
                        break
                if chosen is None:
                    continue
                from_pos = (line_idx, chosen) if is_row else (chosen, line_idx)
                to_pos = (line_idx, target_idx) if is_row else (target_idx, line_idx)
                all_moves.append({'from': from_pos, 'to': to_pos})
                working_line[chosen] = 0
                working_line[target_idx] = 1
                max_distance = max(max_distance, abs(target_idx - chosen))
                left_atoms.remove(chosen)

            for target_idx in sorted(right_targets):
                if working_line[target_idx] == 1:
                    continue
                chosen = None
                for atom_idx in right_atoms:
                    if atom_idx <= target_idx:
                        continue
                    if path_is_clear(atom_idx, target_idx):
                        chosen = atom_idx
                        break
                if chosen is None:
                    continue
                from_pos = (line_idx, chosen) if is_row else (chosen, line_idx)
                to_pos = (line_idx, target_idx) if is_row else (target_idx, line_idx)
                all_moves.append({'from': from_pos, 'to': to_pos})
                working_line[chosen] = 0
                working_line[target_idx] = 1
                max_distance = max(max_distance, abs(target_idx - chosen))
                right_atoms.remove(chosen)

        # Execute all moves in parallel
        if all_moves:
            # Calculate time based on maximum distance
            move_time = self.calculate_realistic_movement_time(max_distance)
            
            # Apply transport efficiency to the moves
            updated_field, successful_moves, failed_moves = self.apply_transport_efficiency(
                all_moves, self.simulator.field
            )
            
            # Record batch move in history
            move_type = 'parallel_row_move' if is_row else 'parallel_column_move'
            self.simulator.movement_history.append({
                'type': move_type,
                'moves': successful_moves + failed_moves,  # Record all attempted moves
                'state': updated_field.copy(),
                'time': move_time,
                'successful': len(successful_moves),
                'failed': len(failed_moves)
            })
            
            # Update simulator's field with final state
            self.simulator.field = updated_field
        
        return len(all_moves)

    def move_atoms_to_target_in_line(self, line_idx, is_row, target_start_idx, target_end_idx):
        raise ValueError("The paper release supports square targets only.")
    
    def axis_wise_centering(self, axis='row', show_visualization=True, use_split_assignment_policy=False):
        """
        Unified axis-wise centering strategy for atom rearrangement.
        
        Args:
            axis: 'row' or 'column' to specify centering direction
            show_visualization: Whether to visualize the rearrangement
            
        Returns:
            Tuple of (final_lattice, execution_time)
        """
        start_time = time.time()
        self.simulator.movement_history = []
        self.initialize_target_region()
        
        # Get target region boundaries
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        
        # Process each line (row or column) in the target region
        total_moves_made = 0
        
        if (axis == 'row'):
            for row in range(target_start_row, target_end_row):
                moves_made = self.center_atoms_in_line(
                    line_idx=row, 
                    is_row=True,
                    target_start_idx=target_start_col, 
                    target_end_idx=target_end_col,
                    use_split_assignment_policy=use_split_assignment_policy,
                )
                total_moves_made += moves_made
        else:  # column
            for col in range(target_start_col, target_end_col):
                moves_made = self.center_atoms_in_line(
                    line_idx=col, 
                    is_row=False,
                    target_start_idx=target_start_row, 
                    target_end_idx=target_end_row,
                    use_split_assignment_policy=use_split_assignment_policy,
                )
                total_moves_made += moves_made
        
        self.simulator.target_lattice = self.simulator.field.copy()
        
        # Animate if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
            
        execution_time = time.time() - start_time
        
        return self.simulator.target_lattice, execution_time
    
    def row_wise_centering(self, show_visualization=True, use_split_assignment_policy=False):
        """Row-wise centering strategy (calls the unified axis_wise_centering)."""
        return self.axis_wise_centering(
            axis='row',
            show_visualization=show_visualization,
            use_split_assignment_policy=use_split_assignment_policy,
        )
    
    def column_wise_centering(self, show_visualization=True, use_split_assignment_policy=False):
        """Column-wise centering strategy (calls the unified axis_wise_centering)."""
        return self.axis_wise_centering(
            axis='column',
            show_visualization=show_visualization,
            use_split_assignment_policy=use_split_assignment_policy,
        )

    def shape_axis_wise_target_fill(self, axis='row', show_visualization=True):
        raise ValueError("The paper release supports square targets only.")

    def shape_row_wise_target_fill(self, show_visualization=True):
        raise ValueError("The paper release supports square targets only.")

    def shape_column_wise_target_fill(self, show_visualization=True):
        raise ValueError("The paper release supports square targets only.")

    def shape_filling_strategy(self, show_visualization=True):
        raise ValueError("The paper release supports square targets only.")
    
    def spread_outer_atoms(self, show_visualization=True, use_split_assignment_policy=False):
        """
        Spreads atoms in rows above and below the target zone outward from the center.
        Only processes atoms that are horizontally aligned with the target zone.
        Atoms left of the horizontal center move leftward, while atoms right of the center move rightward.
        
        Args:
            show_visualization: Whether to visualize the rearrangement
            
        Returns:
            Tuple of (final_lattice, number_of_moves, execution_time)
        """
        start_time = time.time()
        self.simulator.movement_history = []
        self.initialize_target_region()
        
        # Get target region boundaries
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        center_col = (target_start_col + target_end_col) // 2

        total_moves_made = 0
        
        # First process rows above the target zone
        for row in range(0, target_start_row):
            moves_made = self.spread_atoms_in_row(
                row,
                target_start_col,
                target_end_col,
                center_col,
                use_split_assignment_policy=use_split_assignment_policy,
            )
            total_moves_made += moves_made
            
        # Then process rows below the target zone
        for row in range(target_end_row, self.simulator.initial_size[0]):
            moves_made = self.spread_atoms_in_row(
                row,
                target_start_col,
                target_end_col,
                center_col,
                use_split_assignment_policy=use_split_assignment_policy,
            )
            total_moves_made += moves_made
        
        # Animate if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
            
        execution_time = time.time() - start_time
        
        return self.simulator.field.copy(), total_moves_made, execution_time
    
    def spread_atoms_in_row(
        self,
        row,
        target_start_col,
        target_end_col,
        center_col,
        use_split_assignment_policy=False,
    ):
        """
        Moves atoms in a single row outside the target zone outward from the center.
        Only processes atoms that are horizontally aligned with the target zone.
        Atom assignment is split by atom count (not by current side): half of the
        atoms are assigned to the left outward slots and half to the right outward
        slots, preserving order inside each half.
        
        Args:
            row: Row index to process
            target_start_col: Starting column of target region (left edge)
            target_end_col: Ending column of target region (right edge)
            center_col: Center column of the target region
        
        Returns:
            Number of atoms successfully moved
        """
        # Find all atoms in this row that are horizontally aligned with the target zone
        atom_cols = [col for col in range(target_start_col, target_end_col) 
                    if self.simulator.field[row, col] == 1]
        
        if not atom_cols:
            return 0  # No atoms in this row within target zone horizontal bounds
        
        # Create a working copy of the field
        working_field = self.simulator.field.copy()
        moves_executed = 0
        
        # For parallel execution, we will collect all moves first
        all_moves = []
        max_distance = 0

        if use_split_assignment_policy:
            source_indices = sorted(atom_cols)
            fill_count = len(source_indices)
            if fill_count == 0:
                return 0

            left_slots = list(range(target_start_col, center_col))
            right_slots = list(range(center_col, target_end_col))

            left_quota = min(len(left_slots), fill_count // 2)
            right_quota = min(len(right_slots), fill_count - left_quota)

            assigned = left_quota + right_quota
            if assigned < fill_count:
                remaining = fill_count - assigned
                extra_left = min(len(left_slots) - left_quota, remaining)
                left_quota += extra_left
                remaining -= extra_left
                if remaining > 0:
                    extra_right = min(len(right_slots) - right_quota, remaining)
                    right_quota += extra_right

            left_sources = source_indices[:left_quota]
            right_sources = source_indices[left_quota:left_quota + right_quota]
            left_destinations = left_slots[:left_quota] if left_quota > 0 else []
            right_destinations = right_slots[-right_quota:] if right_quota > 0 else []

            def path_is_clear(from_col, to_col):
                start_col = min(from_col, to_col) + 1
                end_col = max(from_col, to_col)
                return all(working_field[row, col] == 0 for col in range(start_col, end_col))

            pending = list(zip(left_sources, left_destinations)) + list(zip(right_sources, right_destinations))
            while pending:
                moved_any = False
                next_pending = []
                for source_col, target_col in pending:
                    if source_col == target_col:
                        continue
                    if working_field[row, source_col] == 0:
                        continue
                    if working_field[row, target_col] == 1:
                        next_pending.append((source_col, target_col))
                        continue
                    if not path_is_clear(source_col, target_col):
                        next_pending.append((source_col, target_col))
                        continue

                    from_pos = (row, source_col)
                    to_pos = (row, target_col)
                    all_moves.append({'from': from_pos, 'to': to_pos})

                    working_field[row, source_col] = 0
                    working_field[row, target_col] = 1

                    distance = abs(target_col - source_col)
                    max_distance = max(max_distance, distance)
                    moved_any = True

                if not moved_any:
                    break
                pending = next_pending
        else:
            left_atoms = [col for col in atom_cols if col < center_col]
            right_atoms = [col for col in atom_cols if col >= center_col]

            left_atoms.sort()
            new_left_positions = set()
            for col in left_atoms:
                new_col = col
                while (
                    new_col > target_start_col
                    and working_field[row, new_col - 1] == 0
                    and (new_col - 1) not in new_left_positions
                ):
                    new_col -= 1
                if new_col == col:
                    continue
                all_moves.append({'from': (row, col), 'to': (row, new_col)})
                new_left_positions.add(new_col)
                working_field[row, col] = 0
                working_field[row, new_col] = 1
                max_distance = max(max_distance, abs(new_col - col))

            right_atoms.sort(reverse=True)
            new_right_positions = set()
            for col in right_atoms:
                new_col = col
                while (
                    new_col < target_end_col - 1
                    and working_field[row, new_col + 1] == 0
                    and (new_col + 1) not in new_right_positions
                ):
                    new_col += 1
                if new_col == col:
                    continue
                all_moves.append({'from': (row, col), 'to': (row, new_col)})
                new_right_positions.add(new_col)
                working_field[row, col] = 0
                working_field[row, new_col] = 1
                max_distance = max(max_distance, abs(new_col - col))
        
        # Execute all moves in parallel
        if all_moves:
            # Calculate time based on maximum distance
            move_time = self.calculate_realistic_movement_time(max_distance)
            
            # Apply transport efficiency to the moves
            updated_field, successful_moves, failed_moves = self.apply_transport_efficiency(
                all_moves, self.simulator.field
            )
            
            # Record batch move in history
            self.simulator.movement_history.append({
                'type': 'parallel_outward_spread',
                'moves': successful_moves + failed_moves,  # Record all attempted moves
                'state': updated_field.copy(),
                'time': move_time,
                'successful': len(successful_moves),
                'failed': len(failed_moves)
            })
            
            moves_executed = len(successful_moves)
            
            # Update simulator's field with final state
            self.simulator.field = updated_field.copy()
        
        return moves_executed
        
    def move_corner_blocks(self, show_visualization=True):
        """
        Moves atoms from corner regions into clean zones above or below target zone.
        First checks which groups of corner blocks can move directly, and moves them in parallel:
        - If all four corners can move, move them all at once
        - Otherwise try to move upper, lower, left or right corner pairs in parallel
        - Then moves any remaining individual corners that can move directly
        - Only then attempts to clean obstacles for the remaining corner blocks.
        Preserves the shape of corner blocks by moving them as a unit.
        
        Args:
            show_visualization: Whether to visualize the rearrangement
            
        Returns:
            Tuple of (final_lattice, moves_made, execution_time)
        """
        start_time = time.time()
        self.simulator.movement_history = []
        self.initialize_target_region()
        
        # Get target region boundaries
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        
        # Get field dimensions
        field_height, field_width = self.simulator.initial_size
        
        # Per-side corner widths can be asymmetric when target placement is not
        # perfectly centered or when side differences are odd.
        left_corner_width = target_start_col
        right_corner_width = field_width - target_end_col
        top_corner_height = target_start_row
        bottom_corner_height = field_height - target_end_row

        if max(left_corner_width, right_corner_width, top_corner_height, bottom_corner_height) <= 0:
            print("No corner blocks to move (initial size <= target size)")
            return self.simulator.field.copy(), 0, 0.0
        
        # Define corner regions based on the projection of the target zone
        corner_regions = {
            'upper_left': {
                'start_row': 0,
                'start_col': 0,
                'end_row': target_start_row,
                'end_col': target_start_col
            },
            'upper_right': {
                'start_row': 0,
                'start_col': target_end_col,
                'end_row': target_start_row,
                'end_col': field_width
            },
            'lower_left': {
                'start_row': target_end_row,
                'start_col': 0,
                'end_row': field_height,
                'end_col': target_start_col
            },
            'lower_right': {
                'start_row': target_end_row,
                'start_col': target_end_col,
                'end_row': field_height,
                'end_col': field_width
            }
        }
        
        # Find atoms in corner regions
        corners = {
            'upper_left': [],
            'upper_right': [],
            'lower_left': [],
            'lower_right': []
        }
        
        # Find atoms in all corners
        for corner_name, region in corner_regions.items():
            for row in range(region['start_row'], region['end_row']):
                for col in range(region['start_col'], region['end_col']):
                    if self.simulator.field[row, col] == 1:
                        corners[corner_name].append((row, col))
        
        # Count total atoms in corners
        total_corner_atoms = sum(len(atoms) for atoms in corners.values())
        print(f"Found {total_corner_atoms} atoms in corner regions")
        
        if total_corner_atoms == 0:
            print("No atoms in corner regions")
            return self.simulator.field.copy(), 0, 0.0
        
        # Calculate movement offsets for each corner
        offset_map = {
            'upper_left': (0, left_corner_width),      # Move right by left-corner width
            'upper_right': (0, -right_corner_width),   # Move left by right-corner width
            'lower_left': (0, left_corner_width),      # Move right by left-corner width
            'lower_right': (0, -right_corner_width)    # Move left by right-corner width
        }
        
        # Prepare for movement
        working_field = self.simulator.field.copy()
        total_moves_made = 0
        
        # PHASE 1: Check which corners can move 
        movable_corners = {}
        corner_obstacles = {}
        
        for corner_name, corner_atoms in corners.items():
            if not corner_atoms:
                continue  # Skip empty corners
                
            offset_row, offset_col = offset_map[corner_name]
            if offset_col == 0:
                continue
            can_move = True
            obstacles = []
            
            # Check if the destination area is clear
            for atom_pos in corner_atoms:
                row, col = atom_pos
                new_row, new_col = row + offset_row, col + offset_col
                
                # Check if destination is within field bounds
                if (new_row < 0 or new_row >= field_height or 
                    new_col < 0 or new_col >= field_width):
                    can_move = False
                    break
                
                # Check if destination is occupied by an atom not part of this corner
                if working_field[new_row, new_col] == 1 and (new_row, new_col) not in corner_atoms:
                    can_move = False
                    obstacles.append((new_row, new_col))
            
            if can_move:
                movable_corners[corner_name] = corner_atoms
            else:
                corner_obstacles[corner_name] = obstacles
        
        # PHASE 2: Group movable corners and move them in parallel
        # Define groups based on spatial arrangement
        corner_groups = {
            'all': ['upper_left', 'upper_right', 'lower_left', 'lower_right'],
            'upper': ['upper_left', 'upper_right'],
            'lower': ['lower_left', 'lower_right'],
            'left': ['upper_left', 'lower_left'],
            'right': ['upper_right', 'lower_right']
        }
        
        # Check which groups can be moved
        movable_groups = {}
        for group_name, corner_names in corner_groups.items():
            # Check if all corners in this group can be moved
            if all(corner_name in movable_corners for corner_name in corner_names):
                movable_groups[group_name] = [
                    (corner_name, movable_corners[corner_name]) 
                    for corner_name in corner_names
                ]
        
        # Keep track of which corners have been moved
        moved_corners = set()
        
        # Choose the largest movable group - prioritize 'all' if possible
        chosen_group = None
        if 'all' in movable_groups:
            chosen_group = movable_groups['all']
            print("Moving all four corner blocks in parallel")
            moved_corners.update(['upper_left', 'upper_right', 'lower_left', 'lower_right'])
        elif any(group in movable_groups for group in ['upper', 'lower', 'left', 'right']):
            # Choose the largest available group (they should all be the same size - 2)
            for group_name in ['upper', 'lower', 'left', 'right']:
                if group_name in movable_groups:
                    chosen_group = movable_groups[group_name]
                    print(f"Moving {group_name} corner blocks in parallel")
                    moved_corners.update(corner_name for corner_name, _ in chosen_group)
                    break
        
        # Move the chosen group in parallel if one exists
        if chosen_group:
            all_moves = []
            max_distance = 0
            
            for corner_name, corner_atoms in chosen_group:
                offset_row, offset_col = offset_map[corner_name]
                direction = "right" if offset_col > 0 else "left"
                print(f"Moving {corner_name} corner block {direction} by {abs(offset_col)} positions")
                
                # Add moves for this corner
                for atom_pos in corner_atoms:
                    row, col = atom_pos
                    new_row, new_col = row + offset_row, col + offset_col
                    
                    from_pos = (row, col)
                    to_pos = (new_row, new_col)
                    all_moves.append({'from': from_pos, 'to': to_pos})
                    
                    # Update working field
                    working_field[row, col] = 0
                    working_field[new_row, new_col] = 1
                    
                    # Track maximum distance
                    distance = abs(offset_col)
                    max_distance = max(max_distance, distance)
            
            # Record the parallel move in history
            if all_moves:
                move_time = self.calculate_realistic_movement_time(max_distance)
                
                # Apply transport efficiency to the moves
                updated_field, successful_moves, failed_moves = self.apply_transport_efficiency(
                    all_moves, self.simulator.field
                )
                
                group_type = 'all_corners' if len(moved_corners) == 4 else '_'.join(moved_corners)
                self.simulator.movement_history.append({
                    'type': f'parallel_{group_type}_move',
                    'moves': successful_moves + failed_moves,  # Record all attempted moves
                    'state': updated_field.copy(),
                    'time': move_time,
                    'successful': len(successful_moves),
                    'failed': len(failed_moves)
                })
                total_moves_made += len(successful_moves)
                
                # Update simulator's field
                self.simulator.field = updated_field.copy()
        else:
            print("No corner blocks can be moved in groups")
        
        # PHASE 3: Move any remaining individual corners that can be moved directly
        # Find all movable corners that haven't been moved yet
        remaining_movable = {name: atoms for name, atoms in movable_corners.items() 
                             if name not in moved_corners}
        
        if remaining_movable:
            print(f"\nMoving {len(remaining_movable)} remaining individual movable corners")
            
            # Process each movable corner one by one
            for corner_name, corner_atoms in remaining_movable.items():
                offset_row, offset_col = offset_map[corner_name]
                direction = "right" if offset_col > 0 else "left"
                print(f"Moving {corner_name} corner block {direction} by {abs(offset_col)} positions")
                
                corner_moves = []
                
                # Add moves for this corner
                for atom_pos in corner_atoms:
                    row, col = atom_pos
                    new_row, new_col = row + offset_row, col + offset_col
                    
                    from_pos = (row, col)
                    to_pos = (new_row, new_col)
                    corner_moves.append({'from': from_pos, 'to': to_pos})
                    
                    # Update working field
                    working_field[row, col] = 0
                    working_field[new_row, new_col] = 1
                
                # Record the move in history
                if corner_moves:
                    move_time = self.calculate_realistic_movement_time(abs(offset_col))
                    
                    # Apply transport efficiency to the moves
                    updated_field, successful_moves, failed_moves = self.apply_transport_efficiency(
                        corner_moves, self.simulator.field
                    )
                    
                    self.simulator.movement_history.append({
                        'type': f'move_{corner_name}_block',
                        'moves': successful_moves + failed_moves,  # Record all attempted moves
                        'state': updated_field.copy(),
                        'time': move_time,
                        'successful': len(successful_moves),
                        'failed': len(failed_moves)
                    })
                    total_moves_made += len(successful_moves)
                    
                    # Update simulator's field
                    self.simulator.field = updated_field.copy()
                    
                    # Mark this corner as moved
                    moved_corners.add(corner_name)
        
        # Animate if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
            
        execution_time = time.time() - start_time
        
        return self.simulator.field.copy(), total_moves_made, execution_time

    def center_filling_strategy(self, show_visualization=True):
        """
        Center filling entry point.

        By default this delegates to `blind_center_filling_strategy`, so
        `center_filling_strategy`, `blind_center_filling_strategy` (atlas_classic),
        and one-iteration `iterative_blind_center_filling` share identical behavior.

        Set constraint `use_legacy_center_filling_strategy=True` to use the older
        comprehensive multi-phase implementation below.

        Legacy implementation summary:
        1. Apply row-wise centering and column-wise centering
        2. Next iteratively:
            a. Spreads atoms outside the target zone outward from center
            b. Applies column-wise centering to utilize the repositioned atoms
            c. Continues until no further improvement
        3. Then moves corner blocks into clean zones
        4. Applies column-wise centering to move corner atoms into the target zone
        5. First repair attempt for remaining defects
        6. If defects remain, iteratively:
            a. Apply full squeezing (row-wise + column-wise centering)
            b. Repair remaining defects
            c. Continue until perfect fill or no improvement
        
        Args:
            show_visualization: Whether to visualize the rearrangement
            
        Returns:
            Tuple of (final_lattice, fill_rate, execution_time)
        """
        if not bool(self.simulator.constraints.get("use_legacy_center_filling_strategy", False)):
            return self.blind_center_filling_strategy(show_visualization=show_visualization)

        start_time = time.time()
        total_movement_history = []
        self.initialize_target_region()
        print("\nCenter filling strategy starting...")
        target_start_row, target_start_col, target_end_row, target_end_col = self.target_region
        
        # Flag to indicate early completion
        early_exit = False
        
        # Check if target zone is already defect-free
        initial_defects = self.count_target_defects()
        if initial_defects == 0:
            print("Target zone is already defect-free! No movements needed.")
            self.simulator.target_lattice = self.simulator.field.copy()
            early_exit = True
        
        # Continue with algorithm if not already perfect
        if not early_exit:
            # Step 1: Iteratively apply row-wise and column-wise centering until convergence
            print("\nStep 1: Row-wise and column-wise centering...")

            # Row-wise centering
            print("Step 1.1: Applying row-wise centering...")
            row_start_time = time.time()
            self.simulator.movement_history = []
            self.row_wise_centering(show_visualization=False)
                    
            # Save movement history
            total_movement_history.extend(self.simulator.movement_history)
            row_moves_made = len(self.simulator.movement_history)
                    
            # Calculate total physical time from movement history
            physical_row_time = sum(move['time'] for move in self.simulator.movement_history)
            print(f"Row-wise centering complete in {time.time() - row_start_time:.3f} seconds, physical time: {physical_row_time:.6f} seconds")
            print(f"Made {row_moves_made} moves during row-wise centering")
                    
            # Check if target zone is full after row-centering
            defects_after_row = self.count_target_defects()
            print(f"Defects after row-centering: {defects_after_row}")
                    
            # Check if we've achieved perfect fill (very unlikely but check anyway)
            if defects_after_row == 0:
                print("Perfect arrangement achieved after row-centering!")
                self.simulator.target_lattice = self.simulator.field.copy()
                early_exit = True
            else:
                # Column-wise centering
                print("Step 1.2: Applying column-wise centering...")
                col_start_time = time.time()
                self.simulator.movement_history = []
                self.column_wise_centering(show_visualization=False)
                        
                # Save movement history
                total_movement_history.extend(self.simulator.movement_history)
                col_moves_made = len(self.simulator.movement_history)
                        
                # Calculate total physical time from movement history
                physical_col_time = sum(move['time'] for move in self.simulator.movement_history)
                print(f"Column-wise centering complete in {time.time() - col_start_time:.3f} seconds, physical time: {physical_col_time:.6f} seconds")
                print(f"Made {col_moves_made} moves during column-wise centering")
                        
                # Count defects after column-centering
                defects_after_col = self.count_target_defects()
                print(f"Defects after column-centering: {defects_after_col}")
                        
                # Check if we've achieved perfect fill
                if defects_after_col == 0:
                    print("Perfect arrangement achieved after column-centering!")
                    self.simulator.target_lattice = self.simulator.field.copy()
                    early_exit = True

            # Track defects for later steps
            previous_defects = defects_after_col if not early_exit else 0

            if not early_exit:              
                # Step 2: Iterative spread-squeeze cycle
                print("\nStep 2: Starting iterative spread-squeeze cycles...")
                
                # Initialize tracking variables for the iteration
                max_iterations = 10  # Prevent infinite loops in edge cases
                min_improvement = 1  # Minimum number of defects that must be fixed to continue
                previous_defects = defects_after_col
                spread_squeeze_time = 0
                spread_squeeze_moves = 0
                iteration = 0
                
                # Continue iterations until no significant improvement or max iterations reached
                while iteration < max_iterations and not early_exit:
                    iteration += 1
                    print(f"\nSpread-squeeze cycle {iteration}/{max_iterations}...")
                    
                    # Spread atoms outward
                    spread_start_time = time.time()
                    self.simulator.movement_history = []
                    _, spread_moves, spread_time = self.spread_outer_atoms(
                        show_visualization=False  # Don't show animation yet
                    )
                    
                    # Save movement history
                    total_movement_history.extend(self.simulator.movement_history)
                    
                    # Calculate total physical time from movement history
                    physical_spread_time = sum(move['time'] for move in self.simulator.movement_history)
                    print(f"Spread phase complete: {spread_moves} atoms moved in {time.time() - spread_start_time:.3f} seconds, physical time: {physical_spread_time:.6f} seconds")
                    
                    spread_squeeze_moves += spread_moves
                    spread_squeeze_time += spread_time
                    
                    # First apply row-wise centering to align atoms if atom loss probability != 0
                    if self.simulator.constraints.get('atom_loss_probability', 0) > 0:
                        row_squeeze_start_time = time.time()
                        self.simulator.movement_history = []
                        _, row_squeeze_time = self.row_wise_centering(
                            show_visualization=False  # Don't show animation yet
                        )
                        # Save movement history
                        total_movement_history.extend(self.simulator.movement_history)

                        # Calculate total physical time from movement history
                        physical_row_squeeze_time = sum(move['time'] for move in self.simulator.movement_history)
                        print(f"Row squeeze phase complete in {time.time() - row_squeeze_start_time:.3f} seconds, physical time: {physical_row_squeeze_time:.6f} seconds")

                        spread_squeeze_time += row_squeeze_time
                    
                    # Then apply column-wise centering
                    col_squeeze_start_time = time.time()
                    self.simulator.movement_history = []
                    _, col_squeeze_time = self.column_wise_centering(
                        show_visualization=False  # Don't show animation yet
                    )
                    
                    # Save movement history
                    total_movement_history.extend(self.simulator.movement_history)
                    
                    # Calculate total physical time from movement history
                    physical_col_squeeze_time = sum(move['time'] for move in self.simulator.movement_history)
                    print(f"Column squeeze phase complete in {time.time() - col_squeeze_start_time:.3f} seconds, physical time: {physical_col_squeeze_time:.6f} seconds")
                    
                    spread_squeeze_time += col_squeeze_time
                    
                    # Count defects after this iteration
                    current_defects = self.count_target_defects()
                    
                    # Calculate improvement
                    defects_fixed = previous_defects - current_defects
                    print(f"Defects after cycle {iteration}: {current_defects} (fixed {defects_fixed} defects)")
                    
                    # Check if we've achieved perfect fill
                    if current_defects == 0:
                        print("Perfect arrangement achieved after spread-squeeze cycles!")
                        self.simulator.target_lattice = self.simulator.field.copy()
                        early_exit = True
                        break
                        
                    # Check if we should continue
                    if defects_fixed < min_improvement:
                        print(f"Stopping iterations: improvement ({defects_fixed}) below threshold ({min_improvement})")
                        break
                        
                    # Update for next iteration
                    previous_defects = current_defects
            
            if not early_exit:
                # Step 3: Move corner blocks
                print("\nStep 3: Moving corner blocks into clean zones...")
                corner_start_time = time.time()
                self.simulator.movement_history = []
                
                # Move corner blocks
                self.move_corner_blocks(
                    show_visualization=False  # Don't show animation yet
                )
                
                # Save movement history
                total_movement_history.extend(self.simulator.movement_history)
                
                # Calculate total physical time from movement history
                physical_corner_time = sum(move['time'] for move in self.simulator.movement_history)
                print(f"Corner block movement complete in {time.time() - corner_start_time:.3f} seconds, physical time: {physical_corner_time:.6f} seconds")
                
                # Step 4: Apply column-wise centering to move atoms from clean zones into target zone
                print("\nStep 4: Applying column-wise centering to incorporate corner blocks...")
                corner_squeeze_start_time = time.time()
                self.simulator.movement_history = []
                self.column_wise_centering(
                    show_visualization=False  # Don't show animation yet
                )
                
                # Save movement history
                total_movement_history.extend(self.simulator.movement_history)
                
                # Calculate total physical time from movement history
                physical_corner_squeeze_time = sum(move['time'] for move in self.simulator.movement_history)
                print(f"Corner squeeze complete in {time.time() - corner_squeeze_start_time:.3f} seconds, physical time: {physical_corner_squeeze_time:.6f} seconds")
                
                # Count defects after corner block movement and squeezing
                defects_after_corner = self.count_target_defects()
                print(f"Defects after corner block movements: {defects_after_corner}")
                
                # Check if we've achieved perfect fill
                if defects_after_corner == 0:
                    print("Perfect arrangement achieved after corner block movements!")
                    self.simulator.target_lattice = self.simulator.field.copy()
                    early_exit = True
                else:
                    # Additional spread-squeeze cycles after corner movement
                    print("\nStep 4b: Additional spread-squeeze cycles after corner movement...")
                    
                    # Initialize tracking variables for the iteration
                    max_additional_iterations = 5  # Maximum number of additional cycles
                    min_improvement = 1  # Minimum defects that must be fixed to continue
                    previous_defects = defects_after_corner
                    additional_iteration = 0
                    
                    # Continue iterations until no significant improvement or max iterations reached
                    while additional_iteration < max_additional_iterations and not early_exit:
                        additional_iteration += 1
                        print(f"\nPost-corner spread-squeeze cycle {additional_iteration}/{max_additional_iterations}...")
                        
                        # Spread atoms outward
                        spread_start_time = time.time()
                        self.simulator.movement_history = []
                        _, spread_moves, spread_time = self.spread_outer_atoms(
                            show_visualization=False  # Don't show animation yet
                        )
                        
                        # Save movement history
                        total_movement_history.extend(self.simulator.movement_history)
                        
                        # Calculate total physical time from movement history
                        physical_spread_time = sum(move['time'] for move in self.simulator.movement_history)
                        print(f"Post-corner spread phase complete: {spread_moves} atoms moved in {time.time() - spread_start_time:.3f} seconds, physical time: {physical_spread_time:.6f} seconds")
                        
                        # First apply row-wise centering to align atoms if atom loss probability != 0
                        if self.simulator.constraints.get('atom_loss_probability', 0) > 0:
                            row_squeeze_start_time = time.time()
                            self.simulator.movement_history = []
                            _, row_squeeze_time = self.row_wise_centering(
                                show_visualization=False  # Don't show animation yet
                            )
                            # Save movement history
                            total_movement_history.extend(self.simulator.movement_history)

                            # Calculate total physical time from movement history
                            physical_row_squeeze_time = sum(move['time'] for move in self.simulator.movement_history)
                            print(f"Post-corner row squeeze phase complete in {time.time() - row_squeeze_start_time:.3f} seconds, physical time: {physical_row_squeeze_time:.6f} seconds")
                        
                        # Then apply column-wise centering
                        col_squeeze_start_time = time.time()
                        self.simulator.movement_history = []
                        _, col_squeeze_time = self.column_wise_centering(
                            show_visualization=False  # Don't show animation yet
                        )
                        
                        # Save movement history
                        total_movement_history.extend(self.simulator.movement_history)
                        
                        # Calculate total physical time from movement history
                        physical_col_squeeze_time = sum(move['time'] for move in self.simulator.movement_history)
                        print(f"Post-corner column squeeze phase complete in {time.time() - col_squeeze_start_time:.3f} seconds, physical time: {physical_col_squeeze_time:.6f} seconds")
                        
                        # Count defects after this iteration
                        current_defects = self.count_target_defects()
                        
                        # Calculate improvement
                        defects_fixed = previous_defects - current_defects
                        print(f"Defects after post-corner cycle {additional_iteration}: {current_defects} (fixed {defects_fixed} defects)")
                        
                        # Check if we've achieved perfect fill
                        if current_defects == 0:
                            print("Perfect arrangement achieved after post-corner spread-squeeze cycles!")
                            self.simulator.target_lattice = self.simulator.field.copy()
                            early_exit = True
                            break
                            
                        # Check if we should continue
                        if defects_fixed < min_improvement:
                            print(f"Stopping post-corner iterations: improvement ({defects_fixed}) below threshold ({min_improvement})")
                            break
                            
                        # Update for next iteration
                        previous_defects = current_defects


            if not early_exit:
                # Step 5: First repair attempt
                print(f"\nStep 5: First repair attempt for {defects_after_corner} defects...")
                repair_start_time = time.time()
                self.simulator.movement_history = []
                self.repair_defects(
                    show_visualization=False  # Don't show animation yet
                )
                
                # Save movement history
                total_movement_history.extend(self.simulator.movement_history)
                
                # Calculate total physical time from movement history
                physical_repair_time = sum(move['time'] for move in self.simulator.movement_history)
                print(f"First repair attempt complete in {time.time() - repair_start_time:.3f} seconds, physical time: {physical_repair_time:.6f} seconds")
                
                # Count defects after initial repair
                defects_after_repair = self.count_target_defects()
                print(f"Defects after first repair: {defects_after_repair}")
                
                # Check if we've achieved perfect fill
                if defects_after_repair == 0:
                    print("Perfect arrangement achieved after first repair attempt!")
                    self.simulator.target_lattice = self.simulator.field.copy()
                    early_exit = True
            
            if not early_exit:
                # Step 6: Iterative squeeze and repair for remaining defects
                print(f"\nStep 6: Iterative squeeze and repair for remaining defects...")
                
                # Parameters for the iterative process
                max_squeeze_repair_iterations = 3
                previous_defect_count = defects_after_repair
                
                for squeeze_repair_iteration in range(max_squeeze_repair_iterations):
                    if early_exit:
                        break
                        
                    print(f"\nSqueeze-repair iteration {squeeze_repair_iteration + 1}/{max_squeeze_repair_iterations}...")
                    
                    # Apply full squeezing to reposition atoms better
                    squeeze_start_time = time.time()
                    self.simulator.movement_history = []
                    
                    # Apply row-wise centering
                    print("Applying row-wise centering...")
                    self.row_wise_centering(show_visualization=False)
                    
                    # Apply column-wise centering
                    print("Applying column-wise centering...")
                    self.column_wise_centering(show_visualization=False)
                    
                    # Save movement history
                    total_movement_history.extend(self.simulator.movement_history)
                    
                    # Calculate total physical time from movement history
                    physical_squeeze_time = sum(move['time'] for move in self.simulator.movement_history)
                    print(f"Squeezing complete in {time.time() - squeeze_start_time:.3f} seconds, physical time: {physical_squeeze_time:.6f} seconds")
                    
                    # Check if squeezing fixed any defects
                    defects_after_squeeze = self.count_target_defects()
                    defects_fixed_by_squeeze = previous_defect_count - defects_after_squeeze
                    
                    if defects_fixed_by_squeeze > 0:
                        print(f"Squeezing fixed {defects_fixed_by_squeeze} defects directly!")
                    
                    # Check if we've achieved perfect fill
                    if defects_after_squeeze == 0:
                        print(f"Perfect arrangement achieved after squeeze iteration {squeeze_repair_iteration + 1}!")
                        self.simulator.target_lattice = self.simulator.field.copy()
                        early_exit = True
                        break
                    
                    # Apply repair for remaining defects
                    repair_start_time = time.time()
                    self.simulator.movement_history = []
                    
                    # Apply direct defect repair
                    self.repair_defects(
                        show_visualization=False
                    )
                    
                    # Save movement history
                    total_movement_history.extend(self.simulator.movement_history)
                    
                    # Calculate total physical time from movement history
                    physical_repair_time = sum(move['time'] for move in self.simulator.movement_history)
                    print(f"Repair attempt {squeeze_repair_iteration + 1} complete in {time.time() - repair_start_time:.3f} seconds, physical time: {physical_repair_time:.6f} seconds")
                    
                    # Count defects after this repair iteration
                    current_defects = self.count_target_defects()
                    
                    # Calculate overall improvement
                    defects_fixed = previous_defect_count - current_defects
                    print(f"Defects after squeeze-repair {squeeze_repair_iteration + 1}: {current_defects} (fixed {defects_fixed} defects)")
                    
                    # Check if we've achieved perfect fill
                    if current_defects == 0:
                        print(f"Perfect arrangement achieved after squeeze-repair iteration {squeeze_repair_iteration + 1}!")
                        self.simulator.target_lattice = self.simulator.field.copy()
                        early_exit = True
                        break
                        
                    # Check if we made any progress
                    if defects_fixed <= 0 and squeeze_repair_iteration > 0:
                        print(f"No improvement in this iteration - stopping further squeeze-repair attempts")
                        break
                    
                    # Update for next iteration
                    previous_defect_count = current_defects

        # Calculate final fill rate
        target_size = self.get_target_size()
        final_defects = self.count_target_defects()
        final_fill_rate = 1.0 - (final_defects / target_size)
        
        # Calculate retention rate as atoms in target zone / atoms initially loaded in the lattice
        atoms_in_target = self.count_target_atoms()
        retention_rate = atoms_in_target / self.simulator.total_atoms if self.simulator.total_atoms > 0 else 0
        
        # Calculate overall metrics
        execution_time = time.time() - start_time
        print(f"\nCenter filling strategy completed in {execution_time:.3f} seconds")
        print(f"Final fill rate: {final_fill_rate:.2%}")
        print(f"Remaining defects: {final_defects}")
        print(f"Final retention rate: {retention_rate:.2%}")
        
        # Restore complete movement history
        self.simulator.movement_history = total_movement_history
        
        # Animate if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
        
        total_physical_time = sum(move['time'] for move in self.simulator.movement_history)
        print(f"Total physical movement time: {total_physical_time:.6f} seconds")
        print(f"Total time: {execution_time + total_physical_time:.6f} seconds")
            
        return self.simulator.target_lattice, final_fill_rate, execution_time


    def blind_center_filling_strategy(
        self,
        show_visualization=True,
        use_batch_merging=True,
        force_original_split_policy=False,
    ):
        """
        A modified center-based filling strategy that pre-computes all movements on a planning
        lattice, then executes them on the real lattice with actual transport efficiency.
        """
        movement_policy = str(self.simulator.constraints.get("movement_policy", "atlas_classic"))
        if movement_policy != "atlas_classic":
            raise ValueError("The paper release supports the atlas_classic policy only.")

        if self.simulator.target_shape != 'square':
            raise ValueError("The paper release supports square targets only.")

        start_time = time.time()

        # Store the initial real lattice and atom count
        initial_real_lattice = self.simulator.field.copy()
        initial_total_atoms = int(initial_real_lattice.sum())

        # Save and disable atom loss for planning
        original_loss_prob = self.simulator.constraints.get('atom_loss_probability', 0.05)
        self.simulator.constraints['atom_loss_probability'] = 0.0

        # Initialize target region
        self.initialize_target_region()
        tsr, tsc, ter, tec = self.target_region

        planned_moves = []
        phase_counter = 0

        def consume_phase_batches(phase_label: str) -> None:
            nonlocal phase_counter
            phase_counter += 1
            phase_id = f"{phase_label}:{phase_counter}"
            for batch in self.simulator.movement_history:
                tagged = dict(batch)
                tagged['phase'] = phase_id
                planned_moves.append(tagged)

        early_exit = False

        shape_aware = False
        row_center = self.row_wise_centering
        col_center = self.column_wise_centering
        mode_name = "square"
        first_row_split_policy = not force_original_split_policy
        first_col_split_policy = not force_original_split_policy
        first_spread_split_policy = not force_original_split_policy

        print(f"\nPhase 1: Planning movements with virtual perfect transport ({mode_name})...")

        # --- Row-wise centering ---
        print("  Planning row-wise centering...")
        self.simulator.movement_history = []
        if shape_aware:
            row_center(show_visualization=False)
        else:
            row_center(
                show_visualization=False,
                use_split_assignment_policy=first_row_split_policy,
            )
            first_row_split_policy = False
        consume_phase_batches("row_center")
        if shape_aware:
            self.simulator.movement_history = []
            discard_trapped_shape_atoms(self)
            consume_phase_batches("row_center_discard")

        # --- Column-wise centering ---
        print("  Planning column-wise centering...")
        self.simulator.movement_history = []
        if shape_aware:
            col_center(show_visualization=False)
        else:
            col_center(
                show_visualization=False,
                use_split_assignment_policy=first_col_split_policy,
            )
            first_col_split_policy = False
        consume_phase_batches("col_center")
        if shape_aware:
            self.simulator.movement_history = []
            discard_trapped_shape_atoms(self)
            consume_phase_batches("col_center_discard")

        # Early exit if planning copy is perfect
        if self.is_target_shape_complete():
            print("  Planning lattice is defect-free; skipping further planning.")
            early_exit = True

        # --- Iterative spread-squeeze cycles ---
        if not early_exit:
            print("  Starting iterative spread-squeeze cycles...")
            cycle = 0
            previous_defects = self.count_target_defects()
            
            while not early_exit:
                cycle += 1
                print(f"  Planning spread-squeeze cycle {cycle}...")
                
                # Track defects before this cycle
                defects_before_cycle = self.count_target_defects()
                
                # Spread atoms outward
                self.simulator.movement_history = []
                _, spread_moves, _ = self.spread_outer_atoms(
                    show_visualization=False,
                    use_split_assignment_policy=first_spread_split_policy,
                )
                first_spread_split_policy = False
                
                if spread_moves == 0:
                    print("    No more atoms to spread; ending spread-squeeze cycles.")
                    break
                
                consume_phase_batches(f"spread_cycle{cycle}_spread")

                # Column-wise centering
                self.simulator.movement_history = []
                if shape_aware:
                    col_center(show_visualization=False)
                else:
                    col_center(
                        show_visualization=False,
                        use_split_assignment_policy=first_col_split_policy,
                    )
                consume_phase_batches(f"spread_cycle{cycle}_col_center")
                if shape_aware:
                    self.simulator.movement_history = []
                    discard_trapped_shape_atoms(self)
                    consume_phase_batches(f"spread_cycle{cycle}_discard")

                # Check if planning lattice is perfect
                if self.is_target_shape_complete():
                    print("    Planning lattice is now perfect; ending spread-squeeze cycles.")
                    early_exit = True
                    break
                
                # Check for improvement
                defects_after_cycle = self.count_target_defects()
                defects_fixed = defects_before_cycle - defects_after_cycle
                
                print(f"    Cycle {cycle} fixed {defects_fixed} defects ({defects_after_cycle} remaining)")
                
                # If no improvement, stop iterating
                if defects_fixed <= 0:
                    print("    No further improvement; ending spread-squeeze cycles.")
                    break

        # --- Corner block movements ---
        if not early_exit:
            print("  Planning corner block movements...")
            self.simulator.movement_history = []
            self.move_corner_blocks(show_visualization=False)
            consume_phase_batches("corner_blocks")
            if shape_aware:
                self.simulator.movement_history = []
                discard_trapped_shape_atoms(self)
                consume_phase_batches("corner_blocks_discard")

            if self.is_target_shape_complete():
                print("    Planning lattice is now perfect after corner moves.")
                early_exit = True

        # --- Final column-wise centering ---
        if not early_exit:
            print("  Planning final column-wise centering...")
            self.simulator.movement_history = []
            if shape_aware:
                col_center(show_visualization=False)
            else:
                col_center(
                    show_visualization=False,
                    use_split_assignment_policy=first_col_split_policy,
                )
            consume_phase_batches("final_col_center")
            if shape_aware:
                self.simulator.movement_history = []
                discard_trapped_shape_atoms(self)
                consume_phase_batches("final_col_center_discard")

            if self.is_target_shape_complete():
                print("    Planning lattice is now perfect after final centering.")
                early_exit = True

        # --- Final spread-squeeze cycles ---
        if not early_exit:
            print("  Starting final spread-squeeze cycles...")
            cycle = 0
            
            while not early_exit:
                cycle += 1
                print(f"  Planning final spread-squeeze cycle {cycle}...")
                
                # Track defects before this cycle
                defects_before_cycle = self.count_target_defects()
                
                # Spread atoms outward
                self.simulator.movement_history = []
                _, spread_moves, _ = self.spread_outer_atoms(
                    show_visualization=False,
                    use_split_assignment_policy=first_spread_split_policy,
                )
                first_spread_split_policy = False
                
                if spread_moves == 0:
                    print("    No more atoms to spread; ending final spread-squeeze cycles.")
                    break
                
                consume_phase_batches(f"final_spread_cycle{cycle}_spread")

                # Column-wise centering
                self.simulator.movement_history = []
                if shape_aware:
                    col_center(show_visualization=False)
                else:
                    col_center(
                        show_visualization=False,
                        use_split_assignment_policy=first_col_split_policy,
                    )
                consume_phase_batches(f"final_spread_cycle{cycle}_col_center")
                if shape_aware:
                    self.simulator.movement_history = []
                    discard_trapped_shape_atoms(self)
                    consume_phase_batches(f"final_spread_cycle{cycle}_discard")

                # Check if planning lattice is perfect
                if self.is_target_shape_complete():
                    print("    Planning lattice is now perfect; skipping defect repair.")
                    early_exit = True
                    break
                
                # Check for improvement
                defects_after_cycle = self.count_target_defects()
                defects_fixed = defects_before_cycle - defects_after_cycle
                
                print(f"    Final cycle {cycle} fixed {defects_fixed} defects ({defects_after_cycle} remaining)")
                
                # If no improvement, stop iterating
                if defects_fixed <= 0:
                    print("    No further improvement; ending final spread-squeeze cycles.")
                    break

        # --- Defect repair planning ---
        disable_defect_repair = bool(
            self.simulator.constraints.get("disable_defect_repair_planning", False)
        )
        if not early_exit and disable_defect_repair:
            print("  Planning defect repair disabled; skipping.")
        elif not early_exit:
            print("  Planning defect repair...")
            self.simulator.movement_history = []
            self.repair_defects(show_visualization=False)
            consume_phase_batches("defect_repair")
            if shape_aware:
                self.simulator.movement_history = []
                discard_trapped_shape_atoms(self)
                consume_phase_batches("defect_repair_discard")

        # Save planned final state and fill rate
        planned_final_state = self.simulator.field.copy()
        defects = self.count_target_defects(planned_final_state)
        excess = self.count_excess_atoms_in_target_region(planned_final_state)
        planning_fill = 1 - defects / max(self.get_target_size(), 1)
        print(f"  Planning completed: {planning_fill:.2%} fill, {defects} defects, {excess} excess atoms.")

        # Optionally merge batches that can be executed in parallel
        # Record initial batch count so we can report parallelism effectiveness.
        initial_planned_batches = len(planned_moves)
        planned_type_counts = {}
        for batch in planned_moves:
            batch_type = batch.get('type', '')
            planned_type_counts[batch_type] = planned_type_counts.get(batch_type, 0) + 1
        if use_batch_merging:
            planned_moves = merge_parallel_batches(
                planned_moves,
                initial_real_lattice,
                policy=str(self.simulator.constraints.get("batch_merge_policy", "phase_aware")),
            )
        reduced_planned_batches = len(planned_moves)
        # Save counts on simulator for external callers (benchmarks, logging)
        try:
            self.simulator.last_planned_initial_batches = int(initial_planned_batches)
            self.simulator.last_planned_reduced_batches = int(reduced_planned_batches)
            self.simulator.last_planned_type_counts = dict(planned_type_counts)
        except Exception:
            # If simulator does not support attribute setting for some reason, ignore
            pass
        if use_batch_merging:
            print(f"  Parallelizable batches reduced to {reduced_planned_batches} steps.")
        else:
            print(f"  Batch merging disabled; keeping {reduced_planned_batches} planned steps.")

        # --- Phase 2: Execute on real lattice ---
        print("\nPhase 2: Executing planned movements with actual transport efficiency...")
        self.simulator.field = initial_real_lattice.copy()
        self.simulator.constraints['atom_loss_probability'] = original_loss_prob
        self.simulator.movement_history = []

        for idx, batch in enumerate(planned_moves, 1):
            moves = batch.get('moves', [])
            if not moves:
                continue

            if batch.get('type') == 'discard_shape_atom':
                current = self.simulator.field.copy()
                discarded_moves = []
                for move in moves:
                    pos = move['from']
                    if current[pos] == 1:
                        current[pos] = 0
                        discarded_moves.append(move)
                if discarded_moves:
                    self.simulator.movement_history.append({
                        'type': 'discard_shape_atom',
                        'moves': discarded_moves,
                        'state': current.copy(),
                        'time': 0.0,
                        'successful': len(discarded_moves),
                        'failed': 0,
                    })
                    self.simulator.field = current
                continue

            # filter out invalid moves
            current = self.simulator.field.copy()
            valid = [m for m in moves if current[m['from']] == 1]
            if not valid:
                continue

            # compute max distance and apply
            max_d = max(abs(m['to'][0]-m['from'][0]) + abs(m['to'][1]-m['from'][1]) for m in valid)
            t_physical = self.calculate_realistic_movement_time(max_d)
            updated, succ, fail = self.apply_transport_efficiency(valid, current)

            self.simulator.movement_history.append({
                'type': batch.get('type',''),
                'moves': succ + fail,
                'state': updated.copy(),
                'time': t_physical,
                'successful': len(succ),
                'failed': len(fail)
            })
            self.simulator.field = updated

            if idx % 10 == 0 or idx == len(planned_moves):
                print(f"  Executed batch {idx}/{len(planned_moves)}: "
                    f"{len(succ)} succeeded, {len(fail)} failed")

        if shape_aware:
            discard_trapped_shape_atoms(self)

        # Compute final metrics
        final_defects = self.count_target_defects()
        final_excess = self.count_excess_atoms_in_target_region()
        final_fill = 1 - final_defects / max(self.get_target_size(), 1)
        retention = self.count_target_atoms() / initial_total_atoms if initial_total_atoms else 0
        exec_time = time.time() - start_time

        print(f"\nBlind center filling completed in {exec_time:.3f}s:")
        print(f"  Final fill rate: {final_fill:.2%}, defects: {final_defects}, excess atoms: {final_excess}")
        print(f"  Retention rate: {retention:.2%}")
        
        # Animate if requested
        if show_visualization and self.simulator.visualizer:
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
        
        self.simulator.target_lattice = self.simulator.field.copy()
        return self.simulator.target_lattice, final_fill, exec_time
    
    def iterative_blind_center_filling(
        self,
        max_iterations=5,
        min_improvement=0.0,
        show_visualization=True,
        use_batch_merging=True,
    ):
        """
        Iteratively applies the blind center filling strategy until the maximum iterations 
        are reached or no meaningful improvement is made, always using the latest lattice.
        
        Args:
            max_iterations: Maximum number of iterations to attempt. If None, run until
                perfect fill or no meaningful improvement.
            min_improvement: Minimum improvement in fill rate to count as progress
            show_visualization: Whether to visualize the final rearrangement
            use_batch_merging: Whether to merge compatible movement batches
            
        Returns:
            Tuple of (final_lattice, fill_rate, execution_time, iterations_used)
        """
        start_time = time.time()
        
        # Initialize target region once for all iterations
        self.initialize_target_region()
        target_region = self.target_region
        target_start_row, target_start_col, target_end_row, target_end_col = target_region
        target_size = self.get_target_size()
        
        # Store the original state just for reporting purposes
        original_total_atoms = np.sum(self.simulator.field)
        
        # Track metrics for reporting
        fill_rates = []
        iterations_used = 0
        all_movement_history = []
        iteration_stats = []
        no_improvement_streak = 0
        use_hybrid_first_iteration_only = bool(
            self.simulator.constraints.get(
                "atlas_classic_use_hybrid_first_iteration_only",
                True,
            )
        )
        use_original_split_policy = bool(
            self.simulator.constraints.get(
                "atlas_classic_use_original_split_policy",
                False,
            )
        )
        # Near-perfect grace: allow a few extra retries when only a tiny number of
        # defects remain and progress has temporarily stalled.
        near_perfect_defect_threshold = 1
        near_perfect_extra_iterations_allowed = int(
            self.simulator.constraints.get("near_perfect_grace_iterations", 1)
        )
        near_perfect_extra_iterations_used = 0
        
        print("\nStarting Iterative Blind Center Filling Strategy")
        print(f"Target region size: {self.simulator.side_length}x{self.simulator.side_length}")
        print(f"Target positions: {target_size}")
        print(f"Initial atoms: {original_total_atoms}")
        
        # Iterate for a maximum number of iterations (or indefinitely when max_iterations=None)
        iteration = 0
        while True:
            iteration += 1
            if max_iterations is None:
                print(f"\nIteration {iteration}/unbounded:")
            else:
                print(f"\nIteration {iteration}/{max_iterations}:")
            
            # Reset movement history for this iteration
            self.simulator.movement_history = []
            
            # Run the blind center filling strategy on the current lattice state
            iteration_start_time = time.time()
            force_original_split_policy = bool(
                use_original_split_policy
                or (use_hybrid_first_iteration_only and iteration > 1)
            )
            final_lattice, fill_rate, _ = self.blind_center_filling_strategy(
                show_visualization=False,
                use_batch_merging=use_batch_merging,
                force_original_split_policy=force_original_split_policy,
            )
            
            # Save the movement history from this iteration with iteration marker
            iteration_history = self.simulator.movement_history.copy()
            for record in iteration_history:
                record['iteration'] = iteration
            all_movement_history.extend(iteration_history)
            
            iteration_time = time.time() - iteration_start_time
            iterations_used = iteration
            
            # Calculate fill rate to verify
            defects = self.count_target_defects()
            actual_fill_rate = 1.0 - (defects / target_size)
            fill_rates.append(actual_fill_rate)
            iteration_physical_time = float(sum(move.get('time', 0.0) for move in iteration_history))
            iteration_moves = int(len(iteration_history))
            atoms_in_target = int(self.count_target_atoms())
            retention_rate = (atoms_in_target / original_total_atoms) if original_total_atoms > 0 else 0.0
            iteration_stats.append({
                'iteration': iteration,
                'computational_time': float(iteration_time),
                'physical_time': iteration_physical_time,
                'total_time': float(iteration_time + iteration_physical_time),
                'moves': iteration_moves,
                'fill_rate': float(actual_fill_rate),
                'defects': int(defects),
                'atoms_in_target': atoms_in_target,
                'retention_rate': float(retention_rate),
            })
            
            print(f"Fill rate: {actual_fill_rate:.2%}")
            print(f"Defects remaining: {defects} out of {target_size} positions")
            print(f"Time: {iteration_time:.2f} seconds")
            
            # Check if we've achieved perfect fill
            if defects == 0:
                print(f"Perfect fill achieved in {iteration} iterations!")
                break
            
            # Check improvement if not the first iteration
            if iteration > 1:
                improvement = actual_fill_rate - fill_rates[-2]
                print(f"Improvement: {improvement:.2%}")
                
                # Stop only after two consecutive iterations with no meaningful improvement.
                if improvement <= min_improvement:
                    no_improvement_streak += 1
                    print(
                        f"No-improvement streak: {no_improvement_streak}/2 "
                        f"(threshold: {min_improvement:.2%})"
                    )
                    if no_improvement_streak >= 2:
                        defects_remaining = defects
                        if (
                            defects_remaining <= near_perfect_defect_threshold
                            and near_perfect_extra_iterations_used < near_perfect_extra_iterations_allowed
                        ):
                            near_perfect_extra_iterations_used += 1
                            no_improvement_streak = 0
                            print(
                                f"Near-perfect grace iteration {near_perfect_extra_iterations_used}/"
                                f"{near_perfect_extra_iterations_allowed} "
                                f"(defects remaining: {defects_remaining})"
                            )
                        else:
                            print(
                                f"Improvement at/below threshold for two consecutive iterations "
                                f"({improvement:.2%} <= {min_improvement:.2%}) - stopping iterations"
                            )
                            break
                else:
                    no_improvement_streak = 0
            
            # Check if we've reached the maximum iterations
            if max_iterations is not None and iteration == max_iterations:
                print(f"Reached maximum iterations ({max_iterations})")
                break
        
        # Store all movement history for visualization
        self.simulator.movement_history = all_movement_history
        self.simulator.target_lattice = self.simulator.field.copy()
        self.simulator.last_iteration_stats = iteration_stats
        
        # Visualize the final result if requested
        if show_visualization and self.simulator.visualizer:
            print("\nGenerating animation of all iterations...")
            self.simulator.visualizer.animate_movements(self.simulator.movement_history)
        
        execution_time = time.time() - start_time
        final_fill_rate = fill_rates[-1]
        
        max_iterations_label = "unbounded" if max_iterations is None else str(max_iterations)
        print(f"\nIterative Blind Center Filling Results:")
        print(f"Iterations used: {iterations_used}/{max_iterations_label}")
        print(f"Final fill rate: {final_fill_rate:.2%}")
        print(f"Remaining defects: {int(target_size * (1-final_fill_rate))}")
        print(f"Total execution time: {execution_time:.3f} seconds")
        
        return self.simulator.field.copy(), final_fill_rate, execution_time, iterations_used
