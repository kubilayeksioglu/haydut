import numpy as np
from constants import BANDIT_FEEDBACK


class Environment:

    def __init__(self, feedback_model, T):
        self.feedback_model = feedback_model
        self.T = T

    def __str__(self):
        raise NotImplementedError()

    @property
    def action_count(self):
        raise NotImplementedError()

    def get_feedback(self, t, response):
        raise NotImplementedError()


class RandomDistributionEnvironment(Environment):

    def __init__(self, random_state, feedback_model, T, dist, *args, **kwargs):
        super().__init__(feedback_model, T)
        # set the random seed (IMPORTANT)
        np.random.seed(random_state)

        # generate T samples
        self.samples = dist(*args, **kwargs, size=self.T)

        # arm_count and sample_count are required
        self._arm_count = self.samples.shape[1]

    @property
    def arm_count(self):
        return self._arm_count

    def get_feedback(self, t, response):
        assert self.feedback_model is BANDIT_FEEDBACK, "RandomDistributionDataset does not support custom feedback models"
        return self.samples[t, response]


class EnvironmentFactory:

    def __init__(self, feedback_model, T):
        self.feedback_model = feedback_model
        self.T = T

    def make(self, random_state):
        raise NotImplementedError()


class RandomDistributionEnvironmentFactory(EnvironmentFactory):

    def __init__(self, feedback_model, T, dist, *args, **kwargs):
        super().__init__(feedback_model, T)
        self.dist = dist
        self.args = args
        self.kwargs = kwargs

    def make(self, random_state):
        return RandomDistributionEnvironment(random_state, self.feedback_model, self.T, self.dist, *self.args, **self.kwargs)
