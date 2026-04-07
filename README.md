# Evolutionary Kuramoto Dilemma on Temporal Proximity Graphs

This repository contains the source code to simulate the Evolutionary Kuramoto Dilemma (EKD) on spatio-temporal networks, specifically Temporal Proximity Graphs (TPG). The model evaluates the coevolution synchronization and cooperation within a population of mobile agents moving on a 2D square with periodic boundary conditions.

## Repository Contents
* `main.py`: The main script containing the model's logic (Kuramoto dynamics, strategy update via Fermi rule, agents' random walk, and network topology management using `networkx`).
* `requirements.txt`: List of Python libraries needed to run the code.

## Installation

Make sure you have Python 3.x installed. You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```
*(Main dependencies include `numpy`, `networkx`, and `matplotlib`).*

## Usage

To start the simulations, simply run the main file:

```bash
python main.py
```

### Execution Features:
* **Multiprocessing:** The code automatically leverages all available CPUs to parallelize simulations across different parameter grids.
* **Resumable:** The script checks for the existence of output files. If a run is interrupted, it will automatically skip previously computed configurations upon restart.

## Output
The results of the individual simulations are saved in `.npz` format inside the `results/` folder (automatically generated). 
Each `.npz` file contains three NumPy arrays representing the time evolution of the system's order parameters:
* `R_G`: Global synchronization (Macroscopic order parameter).
* `R_L`: Local synchronization (Microscopic order parameter).
* `C`: Average level of cooperation.

## Configuration
The global simulation parameters can be modified directly in the `PARAMETERS` block at the beginning of the `main.py` file. These include:
* `N`: Number of agents (default: 1000).
* `alpha_vec`, `coupling_vec`, `mobility_vec`: Control parameter ranges for the simulation.
* `timestep` and `T`: Resolution and total duration for the numerical integration (4th
