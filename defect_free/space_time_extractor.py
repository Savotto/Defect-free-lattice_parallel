"""
Space-time trajectory extraction module for the defect-free lattice simulator.
Processes movement history to extract continuous atom trajectories over time.
"""
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any, Optional

def extract_space_time_data(simulator) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    Extract space-time trajectories from movement history.
    
    Args:
        simulator: LatticeSimulator instance with movement history
        
    Returns:
        Dictionary mapping atom IDs to their trajectories [(time, row, col), ...]
    """
    trajectories = {}
    current_positions = {}  # Track current positions of all atoms
    atom_loss_events = []   # Track atom loss events

    # Initialize the atom positions from the initial lattice
    initial_field = simulator.slm_lattice.copy()
    atom_positions = np.where(initial_field == 1)

    # Assign IDs to initial atoms
    atom_id = 0
    for row, col in zip(atom_positions[0], atom_positions[1]):
        current_positions[atom_id] = (row, col)
        trajectories[atom_id] = [(0, row, col)]  # (time, row, col)
        atom_id += 1

    # For each time step, record all atom positions (excluding lost atoms)
    all_time_slices = []
    # Initial time slice
    all_time_slices.append({aid: pos for aid, pos in current_positions.items()})

    # Process each movement step
    current_time = 0
    for step_idx, step in enumerate(simulator.movement_history):
        # Update the time based on the physical time of this step
        step_time = step.get('time', 0)
        current_time += step_time
        time_point = step_idx + 1

        # Get successful and failed moves
        successful_moves = []
        failed_moves = []

        # Some moves might be directly in the 'moves' field
        for move in step.get('moves', []):
            if 'from' in move and 'to' in move:
                # Check if this was a successful or failed move
                from_pos = move.get('from')
                if from_pos in current_positions.values():
                    successful_moves.append(move)
                else:
                    failed_moves.append(move)

        # Process successful moves
        for move in successful_moves:
            from_pos = move.get('from')
            to_pos = move.get('to')

            # Find which atom this is
            moved_atom_id = None
            for aid, pos in list(current_positions.items()):
                if pos == from_pos:
                    moved_atom_id = aid
                    break

            if moved_atom_id is not None:
                # Update the atom's position
                current_positions[moved_atom_id] = to_pos
                # Add to trajectory
                trajectories[moved_atom_id].append((time_point, to_pos[0], to_pos[1]))

        # Process failed moves (atom loss)
        lost_ids = []
        for move in failed_moves:
            from_pos = move.get('from')

            # Find which atom this is
            lost_atom_id = None
            for aid, pos in list(current_positions.items()):
                if pos == from_pos:
                    lost_atom_id = aid
                    break

            if lost_atom_id is not None:
                # Record loss event
                atom_loss_events.append({
                    'atomId': lost_atom_id,
                    'time': time_point,
                    'position': from_pos
                })
                # Remove from current positions immediately
                lost_ids.append(lost_atom_id)

        for lost_atom_id in lost_ids:
            del current_positions[lost_atom_id]

        # For every atom still present, add a trajectory point for this time step (even if not moved)
        for aid, pos in current_positions.items():
            # Only add if this is not already the last recorded position for this atom
            if trajectories[aid][-1][1:] != pos:
                trajectories[aid].append((time_point, pos[0], pos[1]))
            else:
                # Still add a time point to ensure presence at this time
                trajectories[aid].append((time_point, pos[0], pos[1]))

        # Record all atom positions for this time slice (after losses)
        all_time_slices.append({aid: pos for aid, pos in current_positions.items()})

    # Add final target region information
    target_region = None
    if hasattr(simulator.movement_manager, 'target_region') and simulator.movement_manager.target_region:
        target_region = simulator.movement_manager.target_region

    metadata = {
        'totalTimeSteps': len(simulator.movement_history),
        'totalPhysicalTime': sum(step.get('time', 0) for step in simulator.movement_history),
        'initialAtoms': len(trajectories),
        'finalAtoms': len(current_positions),
        'atomLossEvents': len(atom_loss_events),
        'targetRegion': target_region
    }

    # Instead of returning only trajectories, return all atom positions at each time slice
    # Format: list of dicts, each dict: {atomId: (row, col)}, for each time step
    return {
        'trajectories': trajectories,
        'lossEvents': atom_loss_events,
        'metadata': metadata,
        'allTimeSlices': all_time_slices  # for every time step, atom positions
    }

def export_to_json(trajectory_data: Dict, filename: str = 'atom_trajectories.json') -> str:
    """
    Export trajectories to JSON for visualization.
    """
    all_time_slices = trajectory_data.get('allTimeSlices', None)
    loss_events = trajectory_data['lossEvents']
    trajectories = trajectory_data['trajectories']
    metadata = trajectory_data['metadata']

    # Build a mapping: atom_id -> time step when lost (if ever)
    atom_loss_time = {}
    for event in loss_events:
        atom_loss_time[event['atomId']] = event['time']

    json_data = {
        'atoms': [],
        'lossEvents': loss_events,
        'metadata': metadata
    }

    if all_time_slices is not None:
        # For each time slice, add all atoms present at that time, but only before their loss time
        for t, atom_dict in enumerate(all_time_slices):
            for atom_id, (row, col) in atom_dict.items():
                loss_time = atom_loss_time.get(atom_id, None)
                # Change condition: atom is removed at its loss time
                if loss_time is None or t < loss_time:
                    json_data['atoms'].append({
                        'atomId': int(atom_id),
                        'x': int(col),
                        'y': int(row),
                        't': int(t)
                    })
    else:
        # Fallback: use trajectories (old style)
        for atom_id, path in trajectories.items():
            loss_time = atom_loss_time.get(atom_id, None)
            for time_point, row, col in path:
                if loss_time is None or time_point < loss_time:
                    json_data['atoms'].append({
                        'atomId': int(atom_id),
                        'x': int(col),
                        'y': int(row),
                        't': int(time_point)
                    })

    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=2)

    return filename

def generate_html_file(json_file_path: str, output_html: str) -> str:
    """
    Generate a standalone HTML file with the space-time visualization.
    Embeds the JSON data directly in the HTML to avoid CORS issues.
    """
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    
    # Convert the JSON data to a JavaScript-friendly string
    json_str = json.dumps(json_data)
    
    # Create HTML with embedded data and D3.js visualization code
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lattice Space-Time Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-top: 0;
        }
        .controls {
            margin: 20px 0;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .visualization {
            width: 100%;
            height: 600px;
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
            position: relative;
        }
        button {
            padding: 8px 12px;
            background: #4285f4;
            border: none;
            border-radius: 4px;
            color: white;
            cursor: pointer;
        }
        button:hover {
            background: #3b77db;
        }
        input[type="range"] {
            flex: 1;
        }
        .info-panel {
            margin-top: 20px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 5px;
        }
        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 10px 0;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .legend-color {
            width: 15px;
            height: 15px;
            border-radius: 50%;
        }
        .time-label {
            font-weight: bold;
            min-width: 60px;
        }
        .loading {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            background: rgba(255,255,255,0.8);
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Lattice Space-Time Visualization</h1>
        
        <div class="controls">
            <span class="time-label">Time: <span id="time-value">0</span></span>
            <input type="range" id="time-slider" min="0" value="0" step="1">
            <button id="play-button">Play</button>
        </div>
        
        <div class="visualization">
            <svg id="visualization-svg"></svg>
            <div class="loading" id="loading-indicator">Loading trajectory data...</div>
        </div>
        
        <div class="info-panel">
            <h3>Visualization Information</h3>
            <div id="metadata"></div>
            <div class="legend" id="atom-legend"></div>
        </div>
    </div>

    <script>
        // Embedded data - avoid CORS issues with local files
        const trajectoryData = DATA_PLACEHOLDER;

        // Main visualization script
        (function() {
            /* State variables */
            let atomData = [];
            let lossEvents = [];
            let metadata = {};
            let timeSlice = 0;
            let maxTime = 100;
            let playing = false;
            let playInterval;
            let colorScale;

            /* DOM elements */
            const svg = d3.select('#visualization-svg');
            const timeSlider = document.getElementById('time-slider');
            const timeValue = document.getElementById('time-value');
            const playButton = document.getElementById('play-button');
            const loadingIndicator = document.getElementById('loading-indicator');
            const metadataContainer = document.getElementById('metadata');
            const legendContainer = document.getElementById('atom-legend');

            /* Process the embedded data */
            function initializeData() {
                atomData = trajectoryData.atoms;
                lossEvents = trajectoryData.lossEvents || [];
                metadata = trajectoryData.metadata || {};
                loadingIndicator.style.display = 'none';
                setupVisualization();
            }

            function setupVisualization() {
                maxTime = d3.max(atomData, d => d.t);
                timeSlider.max = maxTime;
                const atomIds = [...new Set(atomData.map(d => d.atomId))];
                colorScale = d3.scaleOrdinal(d3.schemeCategory10)
                    .domain(atomIds);
                setupLegend(atomIds);
                displayMetadata();
                render();
                setupEventListeners();
            }

            function setupLegend(atomIds) {
                legendContainer.innerHTML = '';
                atomIds.forEach(atomId => {
                    const legendItem = document.createElement('div');
                    legendItem.className = 'legend-item';
                    const colorBox = document.createElement('div');
                    colorBox.className = 'legend-color';
                    colorBox.style.backgroundColor = colorScale(atomId);
                    const label = document.createElement('span');
                    label.textContent = `Atom ${atomId}`;
                    legendItem.appendChild(colorBox);
                    legendItem.appendChild(label);
                    legendContainer.appendChild(legendItem);
                });
            }

            function displayMetadata() {
                const metadataHTML = `
                    <p><strong>Total Time Steps:</strong> ${metadata.totalTimeSteps || 'N/A'}</p>
                    <p><strong>Physical Time:</strong> ${(metadata.totalPhysicalTime || 0).toFixed(6)} seconds</p>
                    <p><strong>Initial Atoms:</strong> ${metadata.initialAtoms || 'N/A'}</p>
                    <p><strong>Final Atoms:</strong> ${metadata.finalAtoms || 'N/A'}</p>
                    <p><strong>Atom Loss Events:</strong> ${metadata.atomLossEvents || 0}</p>
                `;
                metadataContainer.innerHTML = metadataHTML;
            }

            function setupEventListeners() {
                timeSlider.addEventListener('input', function() {
                    timeSlice = parseInt(this.value, 10);
                    timeValue.textContent = timeSlice;
                    render();
                });
                playButton.addEventListener('click', function() {
                    playing = !playing;
                    this.textContent = playing ? 'Pause' : 'Play';
                    if (playing) {
                        playInterval = setInterval(() => {
                            timeSlice = (timeSlice + 1) % (maxTime + 1);
                            timeSlider.value = timeSlice;
                            timeValue.textContent = timeSlice;
                            render();
                            if (timeSlice === maxTime) {
                                clearInterval(playInterval);
                                playing = false;
                                playButton.textContent = 'Play';
                            }
                        }, 200);
                    } else {
                        clearInterval(playInterval);
                    }
                });
                window.addEventListener('resize', render);
            }

            function render() {
                svg.selectAll("*").remove();
                const container = svg.node().parentElement;
                const width = container.clientWidth;
                const height = container.clientHeight;
                const margin = { top: 20, right: 20, bottom: 40, left: 40 };
                svg.attr("width", width)
                   .attr("height", height);
                const xMax = d3.max(atomData, d => d.x);
                const yMax = d3.max(atomData, d => d.y);
                const xScale = d3.scaleLinear()
                    .domain([0, xMax])
                    .range([margin.left, width - margin.right]);
                const yScale = d3.scaleLinear()
                    .domain([0, yMax])
                    .range([height - margin.bottom, margin.top]);
                drawLatticeGrid(xScale, yScale, margin, width, height, xMax, yMax);
                const sliceData = atomData.filter(d => d.t === timeSlice);
                svg.selectAll(".atom")
                    .data(sliceData)
                    .enter()
                    .append("circle")
                    .attr("class", "atom")
                    .attr("cx", d => xScale(d.x))
                    .attr("cy", d => yScale(d.y))
                    .attr("r", 8)
                    .attr("fill", d => colorScale(d.atomId))
                    .attr("stroke", "black")
                    .attr("stroke-width", 1);
                if (metadata.targetRegion) {
                    const [startRow, startCol, endRow, endCol] = metadata.targetRegion;
                    svg.append("rect")
                        .attr("x", xScale(startCol) - 0.5)
                        .attr("y", yScale(startRow) - 0.5)
                        .attr("width", xScale(endCol) - xScale(startCol))
                        .attr("height", yScale(endRow) - yScale(startRow))
                        .attr("fill", "none")
                        .attr("stroke", "gold")
                        .attr("stroke-width", 2)
                        .attr("stroke-dasharray", "5,5");
                }
                svg.append("text")
                    .attr("x", width / 2)
                    .attr("y", height - 5)
                    .attr("text-anchor", "middle")
                    .text("X Position");
                svg.append("text")
                    .attr("transform", "rotate(-90)")
                    .attr("x", -height / 2)
                    .attr("y", 15)
                    .attr("text-anchor", "middle")
                    .text("Y Position");
                svg.append("text")
                    .attr("x", width - margin.right)
                    .attr("y", margin.top)
                    .attr("text-anchor", "end")
                    .attr("font-weight", "bold")
                    .text(`Time: ${timeSlice}`);
            }

            function drawLatticeGrid(xScale, yScale, margin, width, height, xMax, yMax) {
                for (let x = 0; x <= xMax; x++) {
                    svg.append("line")
                        .attr("x1", xScale(x))
                        .attr("y1", yScale(0))
                        .attr("x2", xScale(x))
                        .attr("y2", yScale(yMax))
                        .attr("stroke", "#ccc")
                        .attr("stroke-width", 0.5);
                }
                for (let y = 0; y <= yMax; y++) {
                    svg.append("line")
                        .attr("x1", xScale(0))
                        .attr("y1", yScale(y))
                        .attr("x2", xScale(xMax))
                        .attr("y2", yScale(y))
                        .attr("stroke", "#ccc")
                        .attr("stroke-width", 0.5);
                }
            }

            initializeData();
        })();
    </script>
</body>
</html>
"""
    # Replace the DATA_PLACEHOLDER with the actual JSON data
    html_content = html_content.replace('DATA_PLACEHOLDER', json_str)
    
    # Write the HTML file
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    return output_html