from abc import ABC, abstractmethod
import numpy as np

class adversarial_noise_strategy(ABC):
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

    def __init__(self, strategy: adversarial_noise_strategy) -> None:
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._strategy = strategy

    @property
    def strategy(self) -> adversarial_noise_strategy:
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: adversarial_noise_strategy) -> None:
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

class uniform_noise(adversarial_noise_strategy):
    """
    The algorithm that adds noise to the data from a uniform distribution. 
    In a uniform distribution all values are equally likely. 
    """
    def add_noise(self, data):
        data_dim = data.shape
        return data + np.random.uniform(low=0, high=1, size=data_dim) ## TODO: Get the low and high values from config

    def get_name(self):
        return "Uniform noise"

class gaussian_noise(adversarial_noise_strategy):
    """
    The algorithm that adds noise to the data from a gaussian distribution.
    A Gaussian distribution is symmetric around the mean. It is also called the bell curve or normal distribution. 
    """
    def add_noise(self, data):
        data_dim = data.shape
        return data + np.random.normal(loc=0, scale=1, size=data_dim) ## TODO: Get the loc and scale values from config

    def get_name(self):
        return "Gaussian noise"

class laplacian_noise(adversarial_noise_strategy):
    """
    The algorithm that adds noise to the data from a laplacian distribution.
    A Laplacian distribution is symmetric around the mean, however it is more concentrated near the mean than a gaussian distribution.
    """
    def add_noise(self, data):
        data_dim = data.shape
        return data + np.random.laplace(loc=0, scale=1, size=data_dim) ## TODO: Get the loc and scale values from config

    def get_name(self):
        return "Laplacian noise"


## Test
if __name__ == "__main__":
    # The client code picks a concrete strategy and passes it to the context.
    # The client should be aware of the differences between strategies in order
    # to make the right choice.
    original_data = np.array([1, 2, 3, 4, 5])
    print("Original data: ", original_data)

    context = AdversarialNoiseContext(laplacian_noise())
    print("Client: Strategy is set to laplacian noise.")
    print("Data after noise: ", context.add_noise(original_data))
    print()

    original_data = np.array([[1, 2, 3, 4, 5], [3.2, 4.5, 6, 7.1, 2.4]])
    context.strategy = gaussian_noise()
    print("Client: Strategy is set to gaussian noise.")
    print("Data after noise: ", context.add_noise(original_data))
    print()

    original_data = np.array([[1], [2], [3], [4], [5]])
    context.strategy = uniform_noise()
    print("Client: Strategy is set to uniform noise.")
    print("Data after noise: ", context.add_noise(original_data))
