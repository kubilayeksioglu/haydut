import numpy as np
from runstats import Statistics
from utils import argmax_tiebreak


class MABAlgorithm:

    name = None
    feedback_model = None

    def run(self, environment):
        rfs = []
        self.pre_run(environment)
        for t in range(environment.T):
            action = self.play()
            feedback = environment.get_feedback(t, action)
            self.post_feedback(action, feedback)
            rfs.append((action, feedback))

        responses, feedbacks = zip(*rfs)

        return responses, feedbacks

    def pre_run(self, environment):
        raise NotImplementedError()

    def play(self):
        raise NotImplementedError()

    def post_feedback(self, response, feedback):
        raise NotImplementedError()


class UCB1(MABAlgorithm):

    name = 'UCB1'

    def pre_run(self, environment):
        self.statistics = [Statistics() for i in range(environment.arm_count)]
        self.T = environment.T
        self.t = 1

    def play(self):
        arm_plays = np.array([len(s)+1 for s in self.statistics])
        arm_means = np.nan_to_num(np.array([s.mean() for s in self.statistics]))
        ucbs = arm_means + np.sqrt(2 * np.log(self.t) / arm_plays)
        return argmax_tiebreak(ucbs)

    def post_feedback(self, selection, feedback):
        self.statistics[selection].push(feedback)
        self.t += 1
