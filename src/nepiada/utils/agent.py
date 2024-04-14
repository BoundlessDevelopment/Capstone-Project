import numpy as np
import itertools

# Local imports
from enum import Enum
from utils.config import Config

from sklearn.cluster import MiniBatchKMeans

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

        # Initialize beliefs as an empty dictionary.
        # This will be populated with the positions that the agent believes itself and other agents to be in.
        self.beliefs = {}

        # This dictionary stores the ideal distance from a drone's neighbour, based on relative_x and relative_y distance
        self.target_neighbour = {}

        self.truthful_weights = {}

        # This is a dict of the form:
        # Key: The agent name that is sending it the information. Call this agent i
        # Value: An array of size k_means_past_buffer_size * number of agents
        #        This is structured such as each k_means_past_buffer_size segment is 
        #        the info agent i sent in the past. The last k_means_past_buffer_size
        #        elements is the most recent info sent by agent i to this agent.
        self.last_messages = {}

        self.model = MiniBatchKMeans(n_clusters=2, random_state=42)
        dummy_inputs = np.array([1, 0])  # Adjust these values as needed
        self.model.partial_fit(dummy_inputs.reshape(-1, 1))

        self.prev_score = 0

        print("Agent INFO: Agent with uid " + str(self.uid) + " has been initialized")

    def set_target_neighbour(self, neighbour_name, distance):
        # The distance is a list of [ relative_x_pos, relative_y_pos ]
        assert len(distance) == 2
        assert distance[0] < Config.size and distance[0] > -(Config.size)
        assert distance[1] < Config.size and distance[1] > -(Config.size)

        self.target_neighbour[neighbour_name] = distance
