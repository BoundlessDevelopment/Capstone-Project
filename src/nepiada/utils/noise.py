from abc import ABC, abstractmethod

class adversarial_noise_strategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def add_noise():    
        pass

    ## For debug
    def get_name():
        return "Not initialized"


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

    def add_noise(self) -> None:
        """
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """

        print(f"Context: Adding {self._strategy.get_name()} noise to the data")
        result = self._strategy.add_noise()
        return result


##### Different types of adversarial noise supported #####

class uniform_noise(adversarial_noise_strategy):
    """
    The algorithm that adds noise to the data from a uniform distribution. 
    In a uniform distribution all values are equally likely. 
    """
    def add_noise(self):
        #TODO(@HP)
        pass

    def get_name(self):
        return "Uniform noise"

class gaussian_noise(adversarial_noise_strategy):
    """
    The algorithm that adds noise to the data from a gaussian distribution.
    A Gaussian distribution is symmetric around the mean. It is also called the bell curve or normal distribution. 
    """
    def add_noise(self):
        #TODO(@HP)
        pass

    def get_name(self):
        return "Gaussian noise"

class laplacian_noise(adversarial_noise_strategy):
    """
    The algorithm that adds noise to the data from a laplacian distribution.
    A Laplacian distribution is symmetric around the mean, however it is more concentrated near the mean than a gaussian distribution.
    """
    def add_noise(self):
        #TODO(@HP)
        pass

    def get_name(self):
        return "Laplacian noise"


## Test
if __name__ == "__main__":
    # The client code picks a concrete strategy and passes it to the context.
    # The client should be aware of the differences between strategies in order
    # to make the right choice.

    context = AdversarialNoiseContext(laplacian_noise())
    print("Client: Strategy is set to laplacian noise.")
    context.add_noise()
    print()

    print("Client: Strategy is set to gaussian noise.")
    context.strategy = gaussian_noise()
    context.add_noise()
