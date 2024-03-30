from abc import ABC, abstractmethod
import numpy as np

class AdversarialNoiseStrategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of the supported algorithms.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def add_noise(self, data):    
        pass

    ## For debug
    @abstractmethod
    def get_name(self):
        pass


class AdversarialNoiseContext():
    """
    The Context defines the interface of adversarial noise to clients.
    """

    def __init__(self, strategy: AdversarialNoiseStrategy) -> None:
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._strategy = strategy

    @property
    def strategy(self) -> AdversarialNoiseStrategy:
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: AdversarialNoiseStrategy) -> None:
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._strategy = strategy

    def add_noise(self, data) -> None:
        """
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """

        print(f"Context: Adding {self._strategy.get_name()} noise to the data")
        result = self._strategy.add_noise(data)
        return result


##### Different types of adversarial noise supported #####

class UniformNoise(AdversarialNoiseStrategy):
    """
    The algorithm that adds noise to the data from a uniform distribution. 
    In a uniform distribution all values are equally likely. 
    """
    def __init__(self, seed = 0):
        """
        Initializes the random seed for consistency
        """
        self.seed = seed
        np.random.seed(seed)
    
    def add_noise(self, data):
        noisy_data = {}
        for key, value in data.items():
            if value is None:
                # Skip None values or handle them differently if needed
                noisy_data[key] = value
                continue
            if isinstance(value, (np.ndarray, np.generic)):
                # Add noise to each element in the tuple
                noisy_data[key] = np.array([val + np.random.uniform() for val in value if val is not None], dtype=np.float32)
            else:
                # Optionally handle other non-tuple values or raise an error
                raise TypeError(f"Unsupported data type for key {key}: {type(value)}")

        assert len(noisy_data) == len(data), "The noisy data is not of the same length as original data"
        return noisy_data
    
    def get_name(self):
        return "Uniform noise"

class GaussianNoise(AdversarialNoiseStrategy):
    """
    The algorithm that adds noise to the data from a gaussian distribution.
    A Gaussian distribution is symmetric around the mean. It is also called the bell curve or normal distribution. 
    """
    def __init__(self, seed = 0):
        """
        Initializes the random seed for consistency
        """
        self.seed = seed
        np.random.seed(seed)

    def add_noise(self, data):
        noisy_data = {}
        for key, value in data.items():
            if value is None:
                # Skip None values or handle them differently if needed
                noisy_data[key] = value
                continue
            if isinstance(value, (np.ndarray, np.generic)):
                # Add noise to each element in the tuple
                noisy_data[key] = np.array([val + np.random.normal(loc=0, scale=1) for val in value if val is not None], dtype=np.float32)
            else:
                # Optionally handle other non-tuple values or raise an error
                raise TypeError(f"Unsupported data type for key {key}: {type(value)}")

        assert len(noisy_data) == len(data), "The noisy data is not of the same length as original data"
        return noisy_data

    def get_name(self):
        return "Gaussian noise"

class LaplacianNoise(AdversarialNoiseStrategy):
    """
    The algorithm that adds noise to the data from a laplacian distribution.
    A Laplacian distribution is symmetric around the mean, however it is more concentrated near the mean than a gaussian distribution.
    """
    def __init__(self, seed = 0):
        """
        Initializes the random seed for consistency
        """
        self.seed = seed
        np.random.seed(seed)

    def add_noise(self, data):
        noisy_data = {}
        for key, value in data.items():
            if value is None:
                # Skip None values or handle them differently if needed
                noisy_data[key] = value
                continue
            if isinstance(value, (np.ndarray, np.generic)):
                # Add noise to each element in the tuple
                noisy_data[key] = np.array([val + np.random.laplace(loc=0, scale=1) for val in value if val is not None], dtype=np.float32)
            else:
                # Optionally handle other non-tuple values or raise an error
                raise TypeError(f"Unsupported data type for key {key}: {type(value)}")

        assert len(noisy_data) == len(data), "The noisy data is not of the same length as original data"
        return noisy_data

    def get_name(self):
        return "Laplacian noise"

class RandomizeData(AdversarialNoiseStrategy):
    """
    The algorithm replaces values with random datapoints. This is not standard noise as we DO NOT add noise to the 
    values, rather we replace the values with random numbers.
    """
    def __init__(self, max_dim, seed = 0):
        """
        Initializes the random seed and the upper bound of the random data by max_dim
        """
        self.seed = seed
        self._max_dim = max_dim
        np.random.seed(seed)

    def add_noise(self, data):
        noisy_data = {}
        for key, value in data.items():
            if value is None:
                # Skip None values or handle them differently if needed
                noisy_data[key] = value
                continue
            if isinstance(value, (np.ndarray, np.generic)):
                # Add noise to each element in the tuple
                noisy_data[key] = np.array([np.random.randint(0, self._max_dim, size = val.shape) for val in value if val is not None], dtype=np.float32)
            else:
                # Optionally handle other non-tuple values or raise an error
                raise TypeError(f"Unsupported data type for key {key}: {type(value)}")
        
        assert len(noisy_data) == len(data), "The noisy data is not of the same length as original data"
        return noisy_data

    def get_name(self):
        return "Random data"

## Test
if __name__ == "__main__":
    # The client code picks a concrete strategy and passes it to the context.
    # The client should be aware of the differences between strategies in order
    # to make the right choice.
    original_data = {"agent1" : np.array([[1, 2, 3, 4, 5]])}
    print("Original data: ", original_data)

    context = AdversarialNoiseContext(LaplacianNoise())
    print("Client: Strategy is set to laplacian noise.")
    print("Data after noise: ", context.add_noise(original_data))
    print()

    original_data = {"agent1" : np.array([[1, 2, 3, 4, 5]])}
    context.strategy = GaussianNoise()
    print("Client: Strategy is set to gaussian noise.")
    print("Data after noise: ", context.add_noise(original_data))
    print()

    original_data = {"agent1" : np.array([[1, 2, 3, 4, 5]])}
    context.strategy = UniformNoise()
    print("Client: Strategy is set to uniform noise.")
    print("Data after noise: ", context.add_noise(original_data))
    print()

    original_data = {"agent1" : np.array([[1, 2, 3, 4, 5]])}
    context.strategy = RandomizeData()
    print("Client: Strategy is replace with random values.")
    print("Data after noise: ", context.add_noise(original_data))
    print()
