import numpy as np
import itertools

# Local imports
from enum import Enum
from utils.config import Config


# Types of agents
class AgentType(Enum):
    TRUTHFUL = 1
    ADVERSARIAL = 2


class Entity:  # physical/external base state of all entities
    def __init__(self):
        # physical position
        self.p_pos = None

        # physical velocity - we can probably add this information when communicating with other agents
        self.p_vel = None


class Agent(Entity):  # properties of agent entities
    # Agent IDs begin from 0
    agent_id = itertools.count(0)

    def __init__(self, type):
        super().__init__()

        # Randomize the position of the agent, p_pos is stored as [x_coor, y_coor] of the agent
        self.p_pos = np.random.randint(low=0, high=Config.size, size=2)

        # agents type is not defined by default
        self.type = type

        # cannot send communication signals
        self.silent = False

        # cannot observe the world
        self.blind = False

        # unique agent id as int
        self.uid = next(Agent.agent_id)

        print("Agent with uid " + str(self.uid) + " has been initialized")
