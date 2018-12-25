import numpy as np
from constants import BANDIT_FEEDBACK


def average_reward(results, feedback=BANDIT_FEEDBACK):

    loop_count = len(results)
    T = len(results[0][0])

    if feedback == BANDIT_FEEDBACK:
        feedbacks = np.zeros(T)
        for _, fs in results:
            feedbacks += np.array(fs) / loop_count

    return feedbacks