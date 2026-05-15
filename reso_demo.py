#!/usr/bin/env python3
"""ResoForge demo runner.

This script provides a runnable Python scaffold for the sample Reso file
and demonstrates a simplified thermodynamic pipeline.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from reso_parser import parse_reso_file

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np


class Resource:
    def __init__(self, path: str):
        self.path = path
        self.children: Dict[str, "Resource"] = {}

    def mount(self, name: str, child: "Resource"):
        self.children[name] = child
        print(f"Mounted {name} at {self.path}")


class Hardware(Resource):
    def __init__(self, path: str, target: str = "sim"):
        super().__init__(path)
        self.target = target


class PipelineValue:
    def __init__(self, value):
        self.value = value

    def __or__(self, operation):
        if callable(operation):
            result = operation(self.value)
            if isinstance(result, (Resource, PipelineValue)):
                return result
            return PipelineValue(result)
        return self


class Model(Resource):
    def __init__(self, path: str):
        super().__init__(path)
        self.energy_fn: Callable[[np.ndarray], float] = lambda s: float(np.sum(s ** 2))

    def energy(self, states: np.ndarray) -> float:
        return float(self.energy_fn(states))

    def __or__(self, operation):
        result = operation(self) if callable(operation) else self
        if isinstance(result, (Resource, PipelineValue)):
            return result
        return PipelineValue(result)


class Forge:
    def __init__(self):
        self.cluster = Hardware("/cluster/default", "sim")
        self.models: Dict[str, Model] = {}
        self.proc: Dict[str, Dict[str, float]] = {}

    def mount(self, path: str, resource: Resource):
        print(f"Forge: Mounted {type(resource).__name__} at {path}")
        if isinstance(resource, Model):
            self.models[path] = resource

    def train(self, epochs: int = 100, **kwargs):
        def _train(model):
            if isinstance(model, Model):
                print(f"Training {model.path} for {epochs} epochs")
            else:
                print("Train: received non-model object, skipping")
            return model

        return _train

    def sample(self, steps: int = 1000, block_size: int = 32, **kwargs):
        def _sample(model):
            if isinstance(model, IsingModel):
                print(f"Sampling {steps} steps from {model.path}")
                return model.sample_states(steps)
            if isinstance(model, Model):
                print(f"Sampling from model {model.path} is not implemented, returning zeros")
                return np.zeros((steps, 1), dtype=float)
            print("Sample: received non-model object, passing through")
            return model

        return _sample

    def evolve(self, generations: int = 10, mutation_rate: float = 0.05):
        def _evolve(model):
            if isinstance(model, Model):
                print(f"Evolving {model.path} over {generations} generations")
                if hasattr(model, "J"):
                    model.J *= 1.0 + random.uniform(-mutation_rate, mutation_rate)
            else:
                print("Evolve: received non-model object, passing through")
            return model

        return _evolve

    def monitor(self, stat: str):
        def _monitor(item):
            key = f"{stat}_{id(item)}"
            self.proc[key] = {"energy": 0.0, "magnetization": 0.0, "entropy": 0.0}
            print(f"/proc/{stat}: {self.proc[key]}")
            return item

        return _monitor


class IsingModel(Model):
    def __init__(self, path: str, size: int = 64, J: float = -1.0):
        super().__init__(path)
        self.size = size
        self.J = J
        self.state = np.random.choice([-1, 1], size=(size,))

    def energy(self, state: Optional[np.ndarray] = None) -> float:
        if state is None:
            state = self.state
        energy = -self.J * np.sum(state * np.roll(state, -1))
        return float(energy)

    def sample_states(self, n_steps: int = 1000) -> np.ndarray:
        states = np.empty((n_steps, self.size), dtype=int)
        current = self.state.copy()
        for step in range(n_steps):
            i = random.randrange(self.size)
            candidate = current.copy()
            candidate[i] *= -1
            delta = self._delta_energy(current, candidate)
            if delta < 0 or random.random() < np.exp(-delta):
                current = candidate
            states[step] = current
        self.state = current
        return states

    def _delta_energy(self, current: np.ndarray, candidate: np.ndarray) -> float:
        return self.energy(candidate) - self.energy(current)


def observe(stat: str, output_dir: Path):
    def _observe(model_or_samples):
        print(f"Observing {stat}")
        if isinstance(model_or_samples, np.ndarray):
            magnetization = np.mean(model_or_samples, axis=1)
            plt.figure(figsize=(8, 4))
            plt.plot(magnetization, label="Magnetization")
            plt.xlabel("Sample step")
            plt.ylabel("Mean magnetization")
            plt.title("Magnetization over Sampling")
            plt.legend()
            plt.tight_layout()
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / "ising_magnetization.png"
            data_path = output_dir / "magnetization.dat"
            plt.savefig(plot_path)
            plt.close()
            np.savetxt(data_path, magnetization)
            print(f"Saved magnetization plot to {plot_path}")
            print(f"Saved magnetization data to {data_path}")
        return model_or_samples

    return _observe


def save_samples(path: Path):
    def _save(samples):
        if isinstance(samples, np.ndarray):
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, samples)
            print(f"Saved sample states to {path}")
        else:
            print("Save samples: no samples to save")
        return samples

    return _save


def analyze(output_dir: Path):
    def _analyze(samples):
        if isinstance(samples, np.ndarray):
            mean_mag = float(np.mean(np.mean(samples, axis=1)))
            std_mag = float(np.std(np.mean(samples, axis=1)))
            model = IsingModel("/tmp", size=samples.shape[1])
            energies = np.array([model.energy(samples[i]) for i in range(samples.shape[0])])
            stats_path = output_dir / "ising_sample_stats.txt"
            output_dir.mkdir(parents=True, exist_ok=True)
            stats_text = (
                f"Sample summary:\n"
                f"- steps: {samples.shape[0]}\n"
                f"- chain length: {samples.shape[1]}\n"
                f"- mean magnetization: {mean_mag:.5f}\n"
                f"- std magnetization: {std_mag:.5f}\n"
                f"- mean energy: {float(np.mean(energies)):.5f}\n"
                f"- energy std: {float(np.std(energies)):.5f}\n"
            )
            stats_path.write_text(stats_text)
            print(f"Wrote sample statistics to {stats_path}")
        return samples

    return _analyze


def deploy(version: str = "v1.0"):
    def _deploy(model):
        if isinstance(model, Model):
            print(f"Deployed {model.path} as version {version} on {forge.cluster.target}")
        else:
            print("Deploy: received non-model object, skipping deployment")
        return model

    return _deploy


def create_model_from_reso(reso_meta: Dict[str, object]) -> Model:
    if reso_meta.get("type") == "ising":
        return IsingModel(
            path=reso_meta.get("path", "/models/ising1d"),
            size=int(reso_meta.get("size", 64)),
            J=float(reso_meta.get("J", -1.0)),
        )
    return IsingModel("/models/ising1d", size=64, J=-1.0)


forge = Forge()


def build_results_directory(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the ResoForge thermodynamic demo.")
    parser.add_argument("--size", type=int, default=64, help="Ising chain size")
    parser.add_argument("--steps", type=int, default=2000, help="Number of sampling steps")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--block-size", type=int, default=32, help="Block size for sampling")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--deploy-version", type=str, default="v0.1-alpha", help="Deployment version label")
    parser.add_argument("--reso-file", type=str, default="Reso Files", help="Optional Reso model definition file")
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    build_results_directory(output_dir)

    model_file = args.reso_file if args.reso_file else "Reso Files"
    if os.path.exists(model_file):
        try:
            reso_meta = parse_reso_file(model_file)
            ising = create_model_from_reso(reso_meta)
            print(f"Loaded model from Reso file: {model_file}")
        except (FileNotFoundError, ValueError) as exc:
            print(f"Warning: failed to parse Reso file '{model_file}': {exc}")
            ising = IsingModel("/models/ising1d", size=args.size, J=-1.0)
    else:
        ising = IsingModel("/models/ising1d", size=args.size, J=-1.0)

    forge.mount(ising.path, ising)

    result = (
        ising
        | forge.train(epochs=args.epochs)
        | forge.sample(steps=args.steps, block_size=args.block_size)
        | observe("magnetization", output_dir)
        | save_samples(output_dir / "ising_samples.npy")
        | analyze(output_dir)
        | forge.evolve(generations=20)
        | deploy(version=args.deploy_version)
    )

    final = result.value if isinstance(result, PipelineValue) else result

    print("\nResoForge pipeline completed successfully.")
    print(f"Final pipeline value type: {type(final).__name__}")


if __name__ == "__main__":
    main()
