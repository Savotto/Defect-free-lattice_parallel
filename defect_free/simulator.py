"""
Core simulator module defining the LatticeSimulator class with initialization and constants.
"""
import numpy as np
import time
from typing import Tuple, Dict, Optional
from defect_free.movement import MovementManager

class LatticeSimulator:
    """
    Simulates a quantum atom lattice with physical constraints.
    """
    # Physical constants
    SITE_DISTANCE = 5.0  # μm
    MAX_ACCELERATION = 2750.0  # m/s² (PowerMove)
    TRAP_TRANSFER_TIME = 60e-6  # seconds (60μs)
    ATOM_LOSS_PROBABILITY = 0.05 # Probability of atom loss per move
    MAX_VELOCITY = 0.13  # m/s (Parallel Assembly of Arbitrary Defect-Free Atom Arrays with a Multitweezer Algorithm)
    DEFAULT_OPTIMIZED_FORMULA_COEFFS = {
        "intercept": 0.95388099,
        "size_over_100": 0.03330644,
        "occupation": 0.00958237,
        "loss_pct": -0.01298460,
        "size_loss_interaction": 0.02666895,
        "occupation_loss_interaction": 0.00662708,
        "mid_occ_quadratic_loss": 0.08237286,
        "size_sq_loss_interaction": -0.01384795,
        "min_margin": 0.96,
        "max_margin": 1.05,
    }
    DEFAULT_OPTIMIZED_MOVE_FORMULA_COEFFS = {
        "intercept": 0.14671017,
        "occupation": 0.79643931,
        "occupation_sq": -0.71503701,
        "loss": -1.74785463,
        "occupation_loss": 2.93305808,
        # Sizing never assumes better transport than this reference loss:
        # for p_loss below it, the target zone is sized as if p_loss equaled it.
        # This replaces the former zero-loss side correction (delta_0) and the
        # high-loss step penalty (b5) edge-case terms.
        "min_sizing_loss": 0.01,
    }

    PAPER_REPRODUCIBILITY_PROFILE = "es2026sep05_610"
    PAPER_MOVE_FORMULA_COEFFS = {
        # Full-precision fitted values used by the simulations. The manuscript
        # displays these rounded to four decimal places.
        "intercept": 0.14671017,
        "occupation": 0.79643931,
        "occupation_sq": -0.71503701,
        "loss": -1.74785463,
        "occupation_loss": 2.93305808,
        "high_loss_step_penalty": 0.15,
        "high_loss_threshold": 0.05,
        "zero_loss_probability_threshold": 1e-12,
        "zero_loss_large_size_threshold": 100,
        "zero_loss_side_delta": 1,
    }
    
    def __init__(self, 
                 initial_size: Tuple[int, int] = (50, 50),
                 occupation_prob: float = 0.5,
                 physical_constraints: Dict = None,
                 reproducibility_profile: Optional[str] = None):
        """
        Initialize the lattice simulator with configurable physical constraints.
        
        Args:
            initial_size: Initial lattice dimensions (rows, columns)
            occupation_prob: Probability of atom occupation (0.0 to 1.0)
            physical_constraints: Override default physical constraints
        """
        self.initial_size = initial_size
        self.occupation_prob = occupation_prob
        
        # Initialize lattices
        self.slm_lattice = None  # Initial lattice
        self.field = None        # Working lattice during rearrangement
        self.target_lattice = None  # Final target lattice
        self.target_mask = None
        
        # Size of the target defect-free region
        self.side_length = min(initial_size)
        self.total_atoms = 0
        self.target_shape = 'square'
        self.target_shape_params = {}
        
        # Configure physical constraints
        self.constraints = {
            'site_distance': self.SITE_DISTANCE,
            'max_acceleration': self.MAX_ACCELERATION,
            'trap_transfer_time': self.TRAP_TRANSFER_TIME,
            'atom_loss_probability': self.ATOM_LOSS_PROBABILITY,
            'movement_policy': 'atlas_classic',
            'max_velocity': self.MAX_VELOCITY,
            'target_sizing_policy': 'optimized_move_formula',
            'optimized_formula_coefficients': self.DEFAULT_OPTIMIZED_FORMULA_COEFFS,
            'optimized_move_formula_coefficients': self.DEFAULT_OPTIMIZED_MOVE_FORMULA_COEFFS,
        }

        if reproducibility_profile is not None:
            if reproducibility_profile != self.PAPER_REPRODUCIBILITY_PROFILE:
                raise ValueError(
                    f"Unknown reproducibility profile: {reproducibility_profile!r}"
                )
            self.constraints.update({
                'reproducibility_profile': reproducibility_profile,
                'movement_policy': 'atlas_classic',
                'target_sizing_policy': 'paper_move_formula',
                'optimized_move_formula_coefficients': dict(self.PAPER_MOVE_FORMULA_COEFFS),
                'disable_defect_repair_planning': True,
                'atlas_classic_use_original_split_policy': True,
                'batch_merge_policy': 'contiguous',
                'near_perfect_grace_iterations': 0,
            })
        
        if physical_constraints:
            if reproducibility_profile is not None:
                allowed_profile_overrides = {'atom_loss_probability'}
                disallowed = set(physical_constraints) - allowed_profile_overrides
                if disallowed:
                    raise ValueError(
                        "The paper reproducibility profile only permits overriding "
                        f"atom_loss_probability; received: {sorted(disallowed)}"
                    )
            self.constraints.update(physical_constraints)
            if (
                'target_sizing_policy' not in physical_constraints
                and (
                    'fill_confidence_z' in physical_constraints
                    or 'target_atom_reserve_margin' in physical_constraints
                )
            ):
                self.constraints['target_sizing_policy'] = 'legacy'
            
        # Movement tracking
        self.movement_history = []
        self.movement_time = 0.0
        self.total_transfer_time = 0.0
        
        # Initialize movement manager
        self.movement_manager = MovementManager(self)
        
        # Visualizer will be assigned externally
        self.visualizer = None
        
    def generate_initial_lattice(self, seed: Optional[int] = None) -> np.ndarray:
        """Generate a random lattice with the specified occupation probability."""
        if seed is not None:
            np.random.seed(seed)
            
        # Generate initial lattice directly in the field
        self.field = np.zeros(self.initial_size, dtype=int)
        
        # Place the initial lattice directly in the field.
        start_row = 0
        start_col = 0
        
        # Create random distribution based on occupation probability
        initial_region = np.random.random(self.initial_size) < self.occupation_prob
        self.field[start_row:start_row+self.initial_size[0], 
                 start_col:start_col+self.initial_size[1]] = initial_region
        
        # Store the initial SLM lattice
        self.slm_lattice = self.field.copy()
        self.total_atoms = np.sum(self.field)
        
        return self.field

    @staticmethod
    def _normalize_lookup_number(value) -> str:
        """Normalize numeric lookup keys so JSON and Python dicts resolve consistently."""
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return format(number, "g")

    def _resolve_optimized_side_offset(
        self,
        lattice_size: int,
        atom_loss_prob: float,
    ) -> int:
        """
        Resolve a calibrated side offset from fine-grained or coarse lookup tables.

        Lookup precedence:
        1. size|occupation|loss
        2. size|loss
        3. default_side_offset / default
        """
        tables = self.constraints.get("optimized_static_side_offset_table", {}) or {}
        occ_key = self._normalize_lookup_number(self.occupation_prob)
        size_key = self._normalize_lookup_number(lattice_size)
        loss_key = self._normalize_lookup_number(atom_loss_prob)

        fine_table = tables.get("size_occ_loss", {}) or {}
        coarse_table = tables.get("size_loss", {}) or {}

        fine_lookup = f"{size_key}|{occ_key}|{loss_key}"
        if fine_lookup in fine_table:
            return int(fine_table[fine_lookup])

        coarse_lookup = f"{size_key}|{loss_key}"
        if coarse_lookup in coarse_table:
            return int(coarse_table[coarse_lookup])

        if "default_side_offset" in tables:
            return int(tables["default_side_offset"])
        if "default" in tables:
            return int(tables["default"])
        return int(self.constraints.get("optimized_static_default_side_offset", 0))

    def _calculate_optimized_formula_margin(
        self,
        lattice_size: int,
        atom_loss_prob: float,
    ) -> float:
        """
        Compute the calibrated dynamic margin used by the optimized target-size formula.
        """
        coeffs = self.constraints.get('optimized_formula_coefficients', {}) or {}
        size_scaled = float(lattice_size) / 100.0
        loss_pct = float(atom_loss_prob) * 100.0
        occupation = float(self.occupation_prob)

        margin = (
            float(coeffs.get('intercept', 0.95388099))
            + float(coeffs.get('size_over_100', 0.03330644)) * size_scaled
            + float(coeffs.get('occupation', 0.00958237)) * occupation
            + float(coeffs.get('loss_pct', -0.01298460)) * loss_pct
            + float(coeffs.get('size_loss_interaction', 0.02666895)) * size_scaled * loss_pct
            + float(coeffs.get('occupation_loss_interaction', 0.00662708)) * occupation * loss_pct
            + float(coeffs.get('mid_occ_quadratic_loss', 0.08237286)) * ((occupation - 0.7) ** 2) * loss_pct
            + float(coeffs.get('size_sq_loss_interaction', -0.01384795)) * (size_scaled ** 2) * loss_pct
        )

        min_margin = float(coeffs.get('min_margin', 0.96))
        max_margin = float(coeffs.get('max_margin', 1.05))
        return float(np.clip(margin, min_margin, max_margin))

    def _calculate_optimized_move_formula_steps(
        self,
        total_atoms: float,
        lattice_size: int,
        atom_loss_prob: float,
    ) -> float:
        """
        Compute the calibrated effective move count directly from size, realized occupation, and loss.
        """
        coeffs = self.constraints.get('optimized_move_formula_coefficients', {}) or {}
        realized_occupation = 0.0
        field_sites = float(self.initial_size[0] * self.initial_size[1])
        if field_sites > 0:
            realized_occupation = float(total_atoms) / field_sites

        effective_steps = np.sqrt(float(lattice_size)) * (
            float(coeffs.get('intercept', 0.14671017))
            + float(coeffs.get('occupation', 0.79643931)) * realized_occupation
            + float(coeffs.get('occupation_sq', -0.71503701)) * (realized_occupation ** 2)
            + float(coeffs.get('loss', -1.74785463)) * atom_loss_prob
            + float(coeffs.get('occupation_loss', 2.93305808)) * realized_occupation * atom_loss_prob
        )
        high_loss_threshold = coeffs.get('high_loss_threshold')
        if high_loss_threshold is not None and atom_loss_prob >= float(high_loss_threshold):
            effective_steps += float(coeffs.get('high_loss_step_penalty', 0.0)) * np.sqrt(
                float(lattice_size)
            )
        min_expected_steps = float(self.constraints.get('min_expected_steps_per_atom', 1.0))
        return float(max(effective_steps, min_expected_steps))

    def _resolve_target_sizing_policy(self) -> str:
        """
        Resolve the effective sizing policy, including no-repair auto-upgrade.
        """
        sizing_policy = self.constraints.get("target_sizing_policy", "legacy")
        disable_repair = bool(self.constraints.get("disable_defect_repair_planning", False))
        auto_no_repair = bool(self.constraints.get("auto_no_repair_target_sizing", True))

        if (
            sizing_policy == "optimized_move_formula"
            and disable_repair
            and auto_no_repair
        ):
            return "optimized_move_formula_no_repair"
        return sizing_policy

    def _postprocess_square_side(self, side_length: int) -> int:
        """Clamp and postprocess a square target side to preserve existing behavior."""
        min_dim = min(self.initial_size)
        side_length = max(0, min(int(side_length), min_dim))
        if side_length == min_dim - 1:
            side_length = min_dim - 2
        return max(side_length, 0)

    def _get_transport_model(self, total_atoms: float) -> Dict[str, float]:
        """Calculate reusable survivor-model diagnostics for target sizing."""
        field_height, field_width = self.initial_size
        lattice_size = max(field_height, field_width)

        base_lattice_size = float(self.constraints.get('base_lattice_size_for_steps', 30.0))
        base_steps = float(self.constraints.get('base_expected_steps_per_atom', 2.0))
        min_expected_steps = float(self.constraints.get('min_expected_steps_per_atom', 1.0))

        scaling_factor = np.sqrt(lattice_size / base_lattice_size)
        expected_steps = max(base_steps * scaling_factor, min_expected_steps)
        atom_loss_prob = float(self.constraints.get('atom_loss_probability', 0.0))
        transport_success_rate = (1 - atom_loss_prob) ** expected_steps
        expected_survivors = float(total_atoms * transport_success_rate)
        survivor_std = float(
            np.sqrt(total_atoms * transport_success_rate * (1 - transport_success_rate))
        )

        return {
            "field_height": field_height,
            "field_width": field_width,
            "lattice_size": lattice_size,
            "scaling_factor": float(scaling_factor),
            "expected_steps": float(expected_steps),
            "atom_loss_prob": atom_loss_prob,
            "transport_success_rate": float(transport_success_rate),
            "expected_survivors": expected_survivors,
            "survivor_std": survivor_std,
        }

    def calculate_max_defect_free_size(self, strategy=None) -> int:
        """
        Calculate the maximum possible size of a defect-free square lattice based on available atoms,
        with scaling of expected movements based on lattice size and a confidence bound on
        surviving atoms under transport loss.
        
        Args:
            strategy: Retained for API compatibility. Only 'center' is supported.
        
        Returns:
            The side length of the maximum possible square lattice
        """
        # Count total atoms in the field
        total_atoms = np.sum(self.field)

        if strategy not in (None, 'center'):
            raise ValueError(f"Unknown strategy: {strategy}. Only 'center' is supported.")

        model = self._get_transport_model(total_atoms)
        sizing_policy = self._resolve_target_sizing_policy()

        if sizing_policy in {
            "optimized_move_formula",
            "optimized_move_formula_no_repair",
            "paper_move_formula",
        }:
            fill_confidence_z = 0.0
            coeffs = self.constraints.get('optimized_move_formula_coefficients', {}) or {}
            if sizing_policy == "paper_move_formula":
                sizing_loss_prob = float(model["atom_loss_prob"])
            else:
                min_sizing_loss = float(coeffs.get('min_sizing_loss', 0.01))
                sizing_loss_prob = max(float(model["atom_loss_prob"]), min_sizing_loss)

            expected_steps = self._calculate_optimized_move_formula_steps(
                total_atoms=total_atoms,
                lattice_size=model["lattice_size"],
                atom_loss_prob=sizing_loss_prob,
            )
            transport_success_rate = (1 - sizing_loss_prob) ** expected_steps
            reserve_margin = 1.0
            confident_survivors = max(total_atoms * transport_success_rate, 0.0)
            usable_atoms = confident_survivors
            max_square_size = self._postprocess_square_side(int(np.floor(np.sqrt(usable_atoms))))
            if sizing_policy == "paper_move_formula":
                zero_loss_threshold = float(
                    coeffs.get('zero_loss_probability_threshold', 1e-12)
                )
                large_size_threshold = int(
                    coeffs.get('zero_loss_large_size_threshold', 100)
                )
                side_delta = int(coeffs.get('zero_loss_side_delta', 1))
                if (
                    model["lattice_size"] >= large_size_threshold
                    and sizing_loss_prob <= zero_loss_threshold
                ):
                    max_square_size = self._postprocess_square_side(
                        max_square_size - side_delta
                    )
            model["expected_steps"] = expected_steps
            model["transport_success_rate"] = transport_success_rate
            model["expected_survivors"] = confident_survivors
            model["survivor_std"] = 0.0
        elif sizing_policy == "optimized_formula":
            fill_confidence_z = 0.0
            reserve_margin = self._calculate_optimized_formula_margin(
                lattice_size=model["lattice_size"],
                atom_loss_prob=model["atom_loss_prob"],
            )
            confident_survivors = max(model["expected_survivors"], 0.0)
            usable_atoms = confident_survivors * reserve_margin
            max_square_size = self._postprocess_square_side(int(np.floor(np.sqrt(usable_atoms))))
        elif sizing_policy == "optimized_static":
            fill_confidence_z = float(
                self.constraints.get("optimized_static_fill_confidence_z", 0.0)
            )
            reserve_margin = float(
                self.constraints.get("optimized_static_reserve_margin", 1.0)
            )
            confident_survivors = max(
                model["expected_survivors"] - fill_confidence_z * model["survivor_std"],
                0.0,
            )
            usable_atoms = confident_survivors * reserve_margin
            baseline_square_size = int(np.floor(np.sqrt(usable_atoms)))
            baseline_square_size = self._postprocess_square_side(baseline_square_size)
            side_offset = self._resolve_optimized_side_offset(
                lattice_size=model["lattice_size"],
                atom_loss_prob=model["atom_loss_prob"],
            )
            max_square_size = self._postprocess_square_side(
                baseline_square_size + side_offset
            )
        else:
            # Confidence-based sizing parameters.
            # - fill_confidence_z controls the tail risk: 2.33 ~= 99% one-sided confidence.
            # - target_atom_reserve_margin is an optional extra margin for pathing/selection overhead.
            fill_confidence_z = float(self.constraints.get('fill_confidence_z', 2.33))
            reserve_margin = float(self.constraints.get('target_atom_reserve_margin', 0.95))
            confident_survivors = max(
                model["expected_survivors"] - fill_confidence_z * model["survivor_std"],
                0.0,
            )
            usable_atoms = confident_survivors * reserve_margin
            max_square_size = self._postprocess_square_side(int(np.floor(np.sqrt(usable_atoms))))

        # Print diagnostic information
        print(f"Lattice dimensions: {model['field_height']}x{model['field_width']}")
        print(f"Target sizing policy: {sizing_policy}")
        print(f"Scaling factor: {model['scaling_factor']:.2f}")
        print(f"Expected steps per atom: {model['expected_steps']:.2f}")
        print(f"Transport success rate: {model['transport_success_rate']:.4f}")
        print(f"Expected survivors: {model['expected_survivors']:.1f}")
        print(f"Survivor std-dev: {model['survivor_std']:.2f}")
        print(f"Confidence z-score: {fill_confidence_z:.2f}")
        print(f"Reserve margin: {reserve_margin:.2f}")
        if sizing_policy == "optimized_static":
            resolved_offset = self._resolve_optimized_side_offset(
                lattice_size=model["lattice_size"],
                atom_loss_prob=model["atom_loss_prob"],
            )
            print(f"Optimized side offset: {resolved_offset}")
        if sizing_policy in {"optimized_move_formula", "optimized_move_formula_no_repair"}:
            coeffs = self.constraints.get('optimized_move_formula_coefficients', {}) or {}
            min_sizing_loss = float(coeffs.get('min_sizing_loss', 0.01))
            if model["atom_loss_prob"] < min_sizing_loss:
                print(
                    "Sizing loss floor applied: "
                    f"p_loss_for_sizing={min_sizing_loss:.3f} "
                    f"(actual p_loss={model['atom_loss_prob']:.3f})"
                )

        # Update the side_length attribute
        self.side_length = max_square_size
        
        return max_square_size

    def rearrange_for_defect_free(
        self,
        strategy='center',
        show_visualization=True,
        target_shape='square',
        target_shape_params: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Rearrange atoms to create a defect-free region using the center strategy.
        
        This method performs the complete atom rearrangement process:
        1. Determine the optimal target region based on available atoms
        2. Calculate the maximum possible defect-free square
        3. Apply the selected filling strategy to create the defect-free region
        
        Args:
            strategy: Which filling strategy to use. Only 'center' is supported.
            show_visualization: Whether to show animation
            
        Returns:
            Tuple of (target_lattice, fill_rate, execution_time)
        """
        start_time = time.time()
        
        # Reset movement tracking
        self.movement_history = []
        self.movement_time = 0.0
        self.target_shape = target_shape
        self.target_shape_params = target_shape_params or {}
        self.target_mask = None
        self.movement_manager.reset_target_definition()
        
        print("Step 1: Determining optimal target region size...")
        # Use the optimal target size calculation
        self.side_length = self.calculate_max_defect_free_size(strategy=strategy)
        total_atoms = np.sum(self.field)
        
        # Calculate atom loss probability
        atom_loss_prob = self.constraints.get('atom_loss_probability', 0.0)
        
        print(f"Step 2: Calculated target bounding box: {self.side_length}x{self.side_length}")
        print(f"Requested target shape: {self.target_shape}")
        
        if atom_loss_prob > 0:
            print(f"(Accounting for {atom_loss_prob:.1%} atom loss probability per move)")
        
        # Apply the selected strategy
        print(f"Step 3: Applying {strategy} filling strategy...")
        result = self.movement_manager.rearrange_for_defect_free(
            strategy=strategy,
            show_visualization=show_visualization
        )
        if self.target_mask is not None:
            target_sites = int(np.count_nonzero(self.target_mask))
            print(f"Target sites in mask: {target_sites}")
            print(f"Using {target_sites} atoms out of {total_atoms} available")
            print(f"Utilization ratio: {target_sites / total_atoms:.2%}")
        
        # Add execution time tracking
        execution_time = time.time() - start_time
        print(f"Total rearrangement time: {execution_time:.3f} seconds")
        
        target_lattice, fill_rate, _ = result
        return target_lattice, fill_rate, execution_time
