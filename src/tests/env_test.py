import pytest
import env.nepiada as nepiada
from tests.test_config import Config

config = Config()
env = nepiada.parallel_env(config = config)

@pytest.fixture
def initialize():
    """
    This runs before all unit tests
    """
    observations, infos = env.reset()

def test_global_rewards():
    pass

def test_local_rewards():
    pass

def test_observation_radius_limit():
    pass
