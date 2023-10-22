# Experiments Repository

In this repository, we focus on:

- Building the layout of the problem
- Conducting new experiments
- Recording observations

All tasks are initially implemented in simple Python before transitioning to more complex tasks using advanced frameworks.

## Current Setup

We've established a scenario with **3 drones in a 2-dimensional space**. These drones aim to form an **equilateral triangular formation** with a side length of 10, staying as close to the origin as possible. The final score is derived from the average of all agent scores. This calculation is based on the objective function from Pavel's paper, considering the mean squared cost of each drone from the origin and their respective distances from one another. Each drone operates using a basic greedy algorithm to determine its direction, and all drones can observe each other.

In our greedy algorithm, there's an adjustable weight for the origin distance cost, rather than keeping the two components of the cost function equivalent. This is crucial because, with a very low weight, drones might avoid moving towards the origin and maintain their triangle formation.

## POC Key Concepts

Starting with a perfect equilateral triangle of length 10:

- `equil_center.py`: Displays a "global optimum" point— an equilateral triangle centered at the origin. Here, each drone has a cost of ~6, the minimum achievable.
- `equil_point.py`: Demonstrates that initializing one drone at the origin causes it to remain stationary, leading to a suboptimal convergence of 6.66.

For `equil_far`:

- `stuck`: Provides an instance where the weight is too low, preventing drone movement.
- `opt`: Shows a near-optimal scenario where a sufficient weight moves the drones towards the center.
- `implode`: Depicts what happens when the weight is overly high, causing the drones to "implode" towards the center, resulting in suboptimal convergence.

## Next Steps

- [ ] Implement a robust greedy epsilon or an alternative algorithm instead of pure greedy.
- [ ] Simulate scenarios with partial information and adversarial settings.
- [ ] Refine the testing framework and conduct grid searches on weight and epsilon values.
- [ ] Migrate the framework, algorithm, and results to PettingZoo libraries.
- [ ] Define explicit convergence criteria.

## Additional Information

The primary source code is located in `drone_simulation.py`, while the testing framework is found in `test_simulation.py`. The `D matrix` outlines the desired final drone configuration. It's essential that this matrix remains symmetric; otherwise, drone pairs might have conflicting objectives and never converge. The "Distance to origin weight" parameter describes its namesake, and "epsilon" determines the likelihood of a drone making a random rather than a greedy move.
The verbose flag prints out all drone information per iteration and plots them. 
