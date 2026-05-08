#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model files for the last generation's population")
    parser.add_argument("--experiment", "-e", required=True, help="Path to local experiment folder")
    parser.add_argument("--ip", required=True, help="Remote IP address")
    parser.add_argument("--port", "-p", type=int, required=True, help="SSH port")
    parser.add_argument("--remote-path", "-r", required=True, help="Path to experiment folder on remote")
    parser.add_argument("--ssh-key", "-k", default="~/.ssh/runpod_id_ed25519", help="Path to SSH key")
    args = parser.parse_args()

    experiment_path = Path(args.experiment)
    summary_file = experiment_path / "experiment_summary.json"

    with open(summary_file) as f:
        data = json.load(f)

    last_gen = data["generations"][-1]
    population = last_gen["population_snapshot"]

    # Ensure local models directory exists
    local_models_dir = experiment_path / "individuals" / "models"
    local_models_dir.mkdir(parents=True, exist_ok=True)

    for individual in population:
        ind_id = individual["id"]
        remote_file = f"{args.remote_path}/individuals/models/{ind_id}_model.pt"
        
        cmd = [
            "rsync", "-avz", "-e", f"ssh -p {args.port} -i {args.ssh_key}",
            f"root@{args.ip}:{remote_file}",
            str(local_models_dir) + "/"
        ]
        
        print(f"Downloading {ind_id}_model.pt...")
        subprocess.run(cmd)
