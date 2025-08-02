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