# Code for "Tabular and Deep Reinforcement Learning for Gittins Index"

This repository contains the code accompanying the paper "Tabular and Deep Reinforcement Learning for Gittins Index" accepted to the main proceedings of WiOpt 2025.

## Abstract

In the realm of multi-armed bandit problems, the Gittins index policy is known to be optimal in maximizing the expected total discounted reward obtained from pulling the Markovian arms. In most realistic scenarios however, the Markovian state transition probabilities are unknown and therefore the Gittins indices cannot be computed. One can then resort to reinforcement learning (RL) algorithms that explore the state space to learn these indices while exploiting to maximize the reward collected. In this work, we propose novel tabular (QGI) and Deep RL (DGN) algorithms for learning the Gittins index that are based on the retirement formulation for the multi-armed bandit problem. When compared with existing RL algorithms that learn the Gittins index, our algorithms have a lower run time, require less storage space (small Q-table size in QGI and smaller replay buffer in DGN), and illustrate better empirical convergence to the Gittins index. This makes our algorithm well suited for problems with large state spaces and is a viable alternative to existing methods. As a key application, we demonstrate the use of our algorithms in minimizing the mean flowtime in a job scheduling problem when jobs are available in batches and have an unknown service time distribution. We provide comparisons of our Retirement-based algorithms (QGI and DGN) with existing methods like Whittles (QWI) and Restart-in-i.

## Repository Structure

The code is organized into directories corresponding to the figures in the paper:

- `Fig 1 (Toy problem - Tabular)`: Code for reproducing results presented in Figure 1, focusing on tabular methods.
- `Fig 2 (Toy problem - NN)`: Code for reproducing results presented in Figure 2, focusing on neural network-based methods.
- `Fig 3 A (Const HR)`: Code for reproducing results presented in Figure 3A, related to job scheduling with constant arrival rates.
- `Fig 3 B,C (Mono HR)`: Code for reproducing results presented in Figure 3 B and C, related to job scheduling with potentially non-constant arrival rates.
- `Fig 4 and 5`: Code for reproducing results presented in Figures 4 and 5, showing comparisons between DGN, Restart-in-i, and Retirement (QGI).
- `Fig 6`: Additional results and code related to Figure 6.
- `Supplementary example`: Contains supplementary code or examples.

## How to Reproduce Figures

To reproduce the figures from the paper, follow the instructions below:

### Figure 1

Navigate to the `Fig 1 (Toy problem - Tabular)` directory. This figure compares the performance of tabular algorithms: Retirement (QGI), Restart-in-i, and Whittles (QWI).

- For Retirement (QGI), use the `_base_code.py` files related to "Retirement".
- For Restart-in-i, use the `_base_code.py` file related to "Restart-in-i".
- For Whittles (QWI), use the `_base_code.py` files related to "Whittle".

Run the appropriate scripts within this directory to generate the data and plots for Figure 1.

### Figure 2

Navigate to the `Fig 2 (Toy problem - NN)` directory. This figure focuses on neural network-based algorithms, specifically QWINN and DGN.

- Run the `QWINNmain` files to reproduce the QWINN results.
- Run the `DGNmain` files to reproduce the DGN results.

Different suffixes on the `QWINNmain` and `DGNmain` files within this directory correspond to different experimental settings explored in the paper.

### Figure 3

Navigate to the `Fig 3 A (Const HR)` and `Fig 3 B,C (Mono HR)` directories. This figure illustrates the performance of the algorithms on the job scheduling problem.

- **Figure 3A (Constant Arrival Rate):**
    - The `_jobs.ipynb` notebook in `Fig 3 A (Const HR)` contains code to sample from fixed job sizes, which are stored in the `jobsizes_constHR.csv` file.
    - The `jobsize_sampling_constHR.py` script is used to populate `jobsizes_constHR.csv`.
    - Files without the `_jobs` suffix handle dynamic job size sampling for each algorithm during the simulation.
- **Figure 3 B and C (Potentially Non-Constant Arrival Rate):**
    - Similar to Figure 3A, the files in `Fig 3 B,C (Mono HR)` are used to generate the results. Files with `_jobs` suffix (if present) likely use pre-sampled job sizes, while others handle dynamic sampling.

Run the relevant `.ipynb` notebooks or `.py` scripts in these directories to reproduce the results for Figure 3.

### Figures 4 and 5

Navigate to the `Fig 4 and 5` directory. These figures present a comparison between the DGN, Restart-in-i, and Retirement (QGI) algorithms. Run the scripts within this directory to generate the data and plots for these comparison figures.

### Figure 6 and Supplementary Example

Explore the `Fig 6` and `Supplementary example` directories for additional code and results presented in the paper's Figure 6 and supplementary materials.

## Paper Reference

A PDF of the accepted paper, "Learning Gittins Indices via Retirement: Tabular and Deep RL Approaches," should be available in this repository, potentially named `1571114374 paper.pdf`.

## Dependencies

(While specific dependencies were not listed, typical Python libraries for RL and numerical computation would be required. Users may need to install libraries such as `numpy`, `scipy`, `tensorflow` or `pytorch` (for DGN/QWINN), and `matplotlib` for plotting. It's recommended to use a virtual environment.)

```bash
# Example of creating and activating a virtual environment
python -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`

# Example of installing potential dependencies
pip install numpy scipy matplotlib tensorflow # or pytorch