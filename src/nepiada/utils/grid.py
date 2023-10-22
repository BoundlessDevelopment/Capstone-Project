import numpy as np
from utils.config import Config

class Grid():
    def __init__(self):
        self.dim = Config.size

        self.state = np.zeros((self.dim, self.dim))

        ## An adjacency list for which agent can communicate with each other
        ## Will need to build it from config data
        self.Gc = {}

        ## An adjacency list for which agent can observe each other
        ## Will need to build it from config data
        self.Go = {}