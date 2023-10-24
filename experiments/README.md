# Experiments Repository

In this repository, our objectives are to:

- Construct the layout of the problem.
- Conduct new experiments.
- Record observations.

All tasks begin with simple Python implementations. As the complexity of the tasks increases, we transition to more sophisticated frameworks.

## Current Setup

We've designed a scenario involving **3 drones in a 2-dimensional space**. The drones are tasked with forming an **equilateral triangular formation** with a side length of 10, aiming to remain as proximate to the origin as possible. The final score is determined by averaging all agent scores. We've based this scoring mechanism on the objective function from Pavel's paper, which accounts for the mean squared cost of each drone's distance from the origin and their relative distances from each other. Every drone utilizes a fundamental greedy algorithm to choose its direction, and all drones have mutual visibility.

In our greedy algorithm, we've introduced an adjustable weight for the origin distance cost. This is distinct from an approach where the two components of the cost function are equivalent. Such an adjustable weight is essential because, at low values, drones may prefer staying in their triangle formation over moving closer to the origin.

## POC Key Concepts

Given a starting point of a perfect equilateral triangle of length 10:

- `equil_center.py`: Highlights a "global optimum" point—an equilateral triangle centralized at the origin. In this setup, each drone incurs a cost of approximately 6, the lowest possible.
- `equil_point.py`: Illustrates that positioning one drone at the origin will make it static, leading to a suboptimal convergence score of 6.66.

For the `equil_far` segment:

- `stuck`: Represents a situation where the weight is insufficient, causing the drones to stay stationary.
- `opt`: Depicts a scenario that is almost optimal, where an adequate weight nudges the drones closer to the center.
- `implode`: Reveals the consequences of setting the weight excessively high, making drones "implode" towards the center and leading to suboptimal convergence.

### Partial Information and Adversarial Implementation

In the advanced stages of our experiments, we have introduced more real-world complexities such as partial information sharing and adversarial behavior among drones:

- **Partial Information**: Not every drone has full knowledge of the complete environment or the positions of all other drones. They rely on two primary mechanisms for information gathering:
  - **Observation**: Each drone has an observation radius within which it can directly see and identify the positions of other drones.
  - **Communication**: Drones can also communicate their beliefs (positions of other drones) within a certain communication radius. This allows drones outside of direct observation to gather information about others indirectly.

- **Adversarial Drones**: In some simulations, certain drones behave as adversaries. These adversarial drones share corrupted data, either by generating noise or by deliberately misleading. This introduces challenges for the regular drones to accurately determine the optimal positions.
  - **Noise Addition**: When an adversarial drone shares its beliefs, it adds noise to the data. This means that the data received from such a drone is perturbed and can mislead others.
  - **Handling Adversaries**: Drones need to determine how to best handle such misleading data, whether by filtering it out, considering it with less weight, or using other mechanisms to ensure optimal decision making

## Next Steps

- [x] Implement a sturdy greedy epsilon or an alternative strategy over the pure greedy approach.
- [x] Introduce simulations with partial information and potential adversarial elements.
- [x] Improve the testing framework and carry out grid searches on weight and epsilon parameters.
- [ ] Transfer the framework, algorithm, and resultant data to the PettingZoo libraries.
- [ ] Clearly articulate the criteria for convergence.
- [ ] Write extensive tests and POCs, leveraging all parameters associated with partial observability and adversaries.

## Additional Information

- `drone_simulation.py`: This is where the core environment setup is located.
- `batch_grid_test.py`: Executes grid searches multiple times for each configuration, aiding in discerning the optimal setup.
- **D matrix**: Defines the intended final positions of the drones. To ensure convergence, this matrix should maintain symmetry.
- **Distance to origin weight**: As the name suggests, this parameter specifies the weight given to a drone's distance from the origin.
- **Epsilon**: Dictates the probability of a drone opting for a random decision instead of a decision from the greedy algorithm.
- **Verbose**: When activated, this flag outputs comprehensive drone data for every iteration and presents them visually in plots.
- **Unit Tests**: Ensures the correct functionality of drone initialization, visibility, communication, simulation termination, and belief updates.
