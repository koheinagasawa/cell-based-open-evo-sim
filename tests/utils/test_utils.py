import numpy as np
import csv
import uuid
import hashlib
import json
from collections import defaultdict
from pathlib import Path

class RunConfig:
    def __init__(self, seed, config_dict, commit="unknown"):
        self.seed = seed                        # Seed for random number genenerator
        self.commit = commit                    # GitHub commit ID
        self.config = config_dict               # The configurations
        self.run_id = str(uuid.uuid4())[:8]     # Unique ID of this run
        self.config_hash = self._hash_config()  # Hash of the configurations
        self.dir = None                         # Directory to save this config and output results

    def _hash_config(self):
        s = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:8]

    def create_dir(self, base_dir):
        self.dir = Path(base_dir) / f"{self.run_id}_{self.config_hash}"
        self.dir.mkdir(parents=True, exist_ok=False)
        return self.dir

    def save(self):
        if self.dir is None:
            raise ValueError("Directory not set")
        with open(self.dir / "run.json", "w") as f:
            json.dump({
                "seed": self.seed,
                "commit": self.commit,
                "config": self.config,
                "config_hash": self.config_hash,
                "run_id": self.run_id
            }, f, indent=2)
            
class Recorder:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.position_log = defaultdict(list)  # {cell_id: [positions]}
        self.state_log = defaultdict(list)     # {cell_id: [states]}
        self.metric_log = []  # [{step, cell_id, pos_0, pos_1, ..., age}]

    def record(self, step, cell):
        cid = cell.id
        pos = cell.position
        state = cell.state

        self.position_log[cid].append(pos.copy())
        self.state_log[cid].append(state.copy())

        record = {
            "step": step,
            "cell_id": cid,
            "age": cell.age,
        }

        # Add each dimension of position as pos_0, pos_1, ...
        for i in range(len(pos)):
            record[f"pos_{i}"] = pos[i]

        # Add each state component as state_0, state_1, ...
        for i in range(len(state)):
            record[f"state_{i}"] = state[i]

        self.metric_log.append(record)

    def save_all(self):
        # Save metrics.csv
        fieldnames = list(self.metric_log[0].keys())
        with open(self.output_dir / "metrics.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metric_log)

        # Save raw arrays
        np.savez(self.output_dir / "raw_data.npz",
                 positions={k: np.array(v) for k, v in self.position_log.items()},
                 states={k: np.array(v) for k, v in self.state_log.items()})
        
        
        print(f"✅ Experiment results saved in: {self.output_dir}")
        
def prepare_run(config_dict, commit="test"):
    """
    Initialize RunConfig and Recorder and return them.
    Leaves actual simulation loop to caller.
    """
    run_config = RunConfig(seed=42, config_dict=config_dict, commit=commit)
    run_dir = run_config.create_dir("outputs")
    run_config.save()

    recorder = Recorder(run_dir)
    return run_config, recorder

def plot_state_trajectories(recorder, show=True):
    import matplotlib.pyplot as plt

    num_cells = len(recorder.state_log)
    fig, axes = plt.subplots(num_cells, 1, figsize=(8, 3 * num_cells), sharex=True)

    if num_cells == 1:
        axes = [axes]  # Ensure iterable

    for ax, (cell_id, states) in zip(axes, recorder.state_log.items()):
        states = np.array(states)
        short_id = cell_id[:6]
        for i in range(states.shape[1]):
            ax.plot(states[:, i], label=f"state_{i}")
        ax.set_title(f"Cell {short_id}")
        ax.set_ylabel("State Value")
        ax.legend()
        ax.grid()

    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(recorder.output_dir / "state_plot.png")
        plt.close()

def plot_2D_position_trajectories(recorder, show=True):
    import matplotlib.pyplot as plt

    for cell_id, positions in recorder.position_log.items():
        positions = np.array(positions)
        short_id = cell_id[:6]
        plt.plot(positions[:, 0], positions[:, 1], marker='o', label=f"{short_id}")

        # Mark start and end positions
        plt.text(positions[0, 0], positions[0, 1], f"{short_id}_start", fontsize=8, color='green')
        plt.text(positions[-1, 0], positions[-1, 1], f"{short_id}_end", fontsize=8, color='red')

    plt.title("Cell Position Trajectories")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.legend()
    plt.grid()
    if show:
        plt.show()
    else:
        plt.savefig(recorder.output_dir / "position_plot.png")
        plt.close()