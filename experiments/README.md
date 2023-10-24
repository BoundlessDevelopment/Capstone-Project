# Experiments Repository

This repository focuses on conducting experiments related to drone formations in a 2-dimensional space. Our primary goals include constructing the problem layout, performing experiments, and documenting observations. While we commence with basic Python implementations, as the tasks escalate in complexity, we shift towards more advanced frameworks.

## Overview

- **Scenario**: Three drones in a 2D space attempting to form an equilateral triangle.
- **Objective**: Achieve an equilateral triangular formation with a side length of 10 while staying close to the origin.
- **Scoring**: Based on Pavel's objective function, the final score is an average of each drone's score. This score considers the mean squared cost of each drone's distance from the origin and their relative distances to each other.
- **Algorithm**: A basic greedy algorithm drives the drones, with an adjustable weight for the origin distance cost.

## Decision Functions: Mean vs Median

In the realm of our experiments, decision functions play a pivotal role in guiding the actions of the drones:

- **Mean Decision Function**: This approach aggregates the data from all drones to derive a collective decision by averaging. While it offers a holistic view, its decisions can be influenced by outliers or misleading data, especially from adversarial drones.
- **Median Decision Function**: Serving as a resilient alternative to the mean, the median decision function minimizes the influence of extreme values or anomalous data. By extracting the median value from the available data, it provides a more stable reference for the drones' actions. Notably, employing the median decision function has enabled us to accurately replicate Dian's code results, affirming its reliability and consistency within our experimental setup.
## Key Concepts

### Basic Simulations

- `equil_center.py`: Demonstrates an equilateral triangle centered at the origin. Here, the cost incurred by each drone is around 6.
- `equil_point.py`: Shows the effect of positioning one drone at the origin—resulting in a suboptimal score of 6.66.

#### For the `equil_far` segment:
- `stuck`: Drones remain stationary due to insufficient weight.
- `opt`: An almost optimal scenario where drones move closer to the center.
- `implode`: Overemphasis on the center causes drones to "implode", leading to suboptimal convergence.

### Advanced Simulations

- **Partial Information**: Drones have limited knowledge about their environment.
  - **Observation**: Drones detect others within a specific observation radius.
  - **Communication**: Drones communicate their perceived positions to others within a communication radius.

- **Adversarial Drones**: Simulations introduce drones that share misleading data.
  - **Noise Addition**: Adversarial drones add noise to shared data.
  - **Handling Adversaries**: Drones develop strategies to deal with inaccurate data.

## Progress & Next Steps

- [x] Shift from pure greedy to a robust greedy epsilon or an alternative approach.
- [x] Incorporate simulations with partial information and adversarial factors.
- [x] Enhance the testing framework and execute grid searches on parameters.
- [ ] Migrate framework, algorithm, and data to the PettingZoo libraries.
- [ ] Define clear convergence criteria.
- [ ] Develop comprehensive tests and POCs with focus on partial observability and adversaries.

## Resources & Additional Details

- **Core Setup**: `drone_simulation.py` houses the foundational environment configurations.
- **Grid Tests**: `batch_grid_test.py` runs grid tests for each configuration to determine the best setup.
- **Parameters & Definitions**:
  - **D matrix**: Specifies the desired final drone positions. It should be symmetrical for convergence.
  - **Distance to Origin Weight**: Weight prioritizing a drone's proximity to the origin.
  - **Epsilon**: Chance of a drone making a random instead of a greedy decision.
  - **Verbose**: Activating this provides detailed drone data and visual plots per iteration.
  - **Unit Tests**: Validates drone features like initialization, visibility, communication, and more.
