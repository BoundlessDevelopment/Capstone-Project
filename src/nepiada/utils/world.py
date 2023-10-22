class World():
    def __init__(self):
        self.agents = []

        self.target_x = 0
        self.target_y = 0

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks