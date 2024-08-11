import numpy as np


class SoftmaxPolicy:
    """ Implement softmax policy for multinimial distribution

    Simple Policy

    - takes action according to the pobability distribution

    """
    def select_action(self, nb_actions, probs):
        """Return the selected action

        # Arguments
            probs (np.ndarray) : Probabilty for each action

        # Returns
            action

        """
        action = np.random.choice(nb_actions, p=probs)
        return action


class EpsGreedyQPolicy:
    """Implement the epsilon greedy policy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """
    def __init__(self, eps=.1):
        self.eps = eps

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)
        return action


class GreedyQPolicy:
    """Implement the greedy policy

    Greedy policy returns the current best action according to q_values
    """
    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        action = np.argmax(q_values)
        return action


class BoltzmannQPolicy:
    """Implement the Boltzmann Q Policy

    Boltzmann Q Policy builds a probability law on q values and returns
    an action selected randomly according to this law.
    """
    def __init__(self, tau=1., clip=(-500., 500.)):
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(nb_actions, p=probs)
        return action


class MaxBoltzmannQPolicy:
    """
    A combination of the eps-greedy and Boltzman q-policy.

    Wiering, M.: Explorations in Efficient Reinforcement Learning.
    PhD thesis, University of Amsterdam, Amsterdam (1999)

    https://pure.uva.nl/ws/files/3153478/8461_UBA003000033.pdf
    """
    def __init__(self, eps=.1, tau=1., clip=(-500., 500.)):
        self.eps = eps
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        """Return the selected action
        The selected action follows the BoltzmannQPolicy with probability epsilon
        or return the Greedy Policy with probability (1 - epsilon)

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
            probs = exp_values / np.sum(exp_values)
            action = np.random.choice(nb_actions, p=probs)
        else:
            action = np.argmax(q_values)
        return action


class BoltzmannGumbelQPolicy:
    """Implements Boltzmann-Gumbel exploration (BGE) adapted for Q learning
    based on the paper Boltzmann Exploration Done Right
    (https://arxiv.org/pdf/1705.10257.pdf).

    BGE is invariant with respect to the mean of the rewards but not their
    variance. The parameter C, which defaults to 1, can be used to correct for
    this, and should be set to the least upper bound on the standard deviation
    of the rewards.

    BGE is only available for training, not testing. For testing purposes, you
    can achieve approximately the same result as BGE after training for N steps
    on K actions with parameter C by using the BoltzmannQPolicy and setting
    tau = C/sqrt(N/K)."""

    def __init__(self, C=1.0):
        assert C > 0, "BoltzmannGumbelQPolicy C parameter must be > 0, not " + repr(C)
        self.C = C
        self.action_counts = None

    def select_action(self, q_values, step_counter):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1, q_values.ndim

        # If we are starting training, we should reset the action_counts.
        # Otherwise, action_counts should already be initialized, since we
        # always do so when we begin training.
        if step_counter == 0:
            self.action_counts = np.ones(q_values.shape)
        assert self.action_counts is not None, self.agent.step
        assert self.action_counts.shape == q_values.shape, (self.action_counts.shape, q_values.shape)

        beta = self.C/np.sqrt(self.action_counts)
        Z = np.random.gumbel(size=q_values.shape)

        perturbation = beta * Z
        perturbed_q_values = q_values + perturbation
        action = np.argmax(perturbed_q_values)

        self.action_counts[action] += 1
        return action


class GumbelSoftmaxPolicy:
    
    def __init__(self, temperature=1.0, eps=0.01):
        self.temperature = temperature
        self.eps = eps
        
    def onehot_from_logits(self, logits):
        argmax_acs = (logits == logits.max(axis=1, keepdims=True)).astype(float)
        rand_acs = np.eye(logits.shape[1])[np.random.choice(range(logits.shape[1]), size=logits.shape[0])]
        return np.array([
            argmax_acs[i] if r > self.eps else rand_acs[i]
            for i, r in enumerate(np.random.rand(logits.shape[0]))
        ])

    def sample_gumbel(self, shape, eps=1e-20):
        U = np.random.uniform(0, 1, shape)
        return -np.log(-np.log(U + eps) + eps)

    def gumbel_softmax_sample(self,logits):
        y = logits + self.sample_gumbel(logits.shape)
        return np.exp(y / self.temperature) / np.sum(np.exp(y / self.temperature), axis=1, keepdims=True)

    def gumbel_softmax(self, logits):
        y = self.gumbel_softmax_sample(logits, self.temperature)
        y_hard = self.onehot_from_logits(y)
        y = (y_hard - y) + y
        return y
