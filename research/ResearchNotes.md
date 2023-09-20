# Documentation of Research

## Index
- Notes of past team's capstone report
- Research on OpenAI Gym


## Notes of past team's capstone report

- At the module level, each agent splits decision-making into three functions: an estimation function, a state update function, and a communication function
- Each agent applied the algorithm separately, whereas there is only one environment for a given problem
- Both algorithms look holistically at each agent's input and take a weighted average from a softmax
- The key intuition of the Cumulative L2 algorithm is that, over successive iterations, adversarial communications differ from the consensus of the group
- The key factor that reduced convergence time for Cumulative L2 algo was skipping the sorting step described in Gadjov et al.
- According to their efforts, NN-based algorithms are slow to train and often don't entail convergence
- In the NN based algorithm they used mean-squared loss instead of cross-entropy loss
- They used a pre-trained AdversaryNet neural network with some transfer learning. 
- They hard to beat the baseline by much in terms of the number of nodes, node and adversarial density (had to tailor very specific examples), and converge speed
- We have room to beat them in terms of the points mentioned above, using an RL-based approach
- They also added multiple animations to showcase the convergence of their algorithms. This is a great addition to our demo meetings.
- They used a generic convex optimizer in the update function as it is a widely accepted assumption that each agent's loss function is convex.
- They employed Gaussian, Laplacian and uniform noise distributions for adding adversarial noise.
- They proposed a conventional statistical algorithm, the main differences to the baseline algorithm are:
  - Initialized a history vector for each agent
  - Take a mean of all incoming communication messages
  - Calculate an error using L2 norm which gives root mean squared error
  - Accumulate this error in this history vector for each agent
  - Calculate truthfulness weights by applying softmax of the history vector with a tunable temperature-scaling hyperparameter
- The algorithm was able to converge on the 21 agents, 11 adversarial case.
- Their medium-sized test case consisted of 21 agents. We might want to relax our specification of having more than 100 agents and make it more conservative to say 40-50 agents.
- They originally experienced with reinforcement learning strategies such as deep Q-networks, but abandoned it due to reward sparsity in most environments.
  - We will be focusing on getting RL strategies to work for this problem.

## OpenAI Gym Research
