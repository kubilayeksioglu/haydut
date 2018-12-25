from algs import MABAlgorithm
from environments import EnvironmentFactory


class Experiment:

    def __init__(self, loop_count:int, factory:EnvironmentFactory):
        self.factory = factory
        self.loop_count = loop_count

    def run(self, algorithm: MABAlgorithm, verbose=False):

        if verbose:
            print ("Running %s on %s" % (algorithm, self.factory))

        results = []
        for i in range(self.loop_count):
            dataset = self.factory.make(i)
            rs, fs = algorithm.run(dataset)
            results.append((rs, fs))
            if verbose:
                print (".", end="")

        if verbose:
            print("\nCompleted %s on %s" % (algorithm, self.factory))

        return results