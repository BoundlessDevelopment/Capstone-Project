import numpy as np
from enum import Enum

# Types of agents
class AgentType(Enum):
    TRUTHFUL = 0
    ADVERSARIAL = 1

class EntityState:  # physical/external base state of all entities
    def __init__(self):
        # physical position
        self.p_pos = None

        # physical velocity
        self.p_vel = None

class Agent(Entity):  # properties of agent entities
    def __init__(self):
        super().__init__()

        # agents type is not defined by default
        self.type = None

        # cannot send communication signals
        self.silent = False

        # cannot observe the world
        self.blind = False

    
