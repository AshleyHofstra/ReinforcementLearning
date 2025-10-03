import gymnasium as gym
import numpy as np
from collections import defaultdict

class QAgent:
    def __init__(
            self,
            env: gym.Env,
            learningRate: float, #alpha
            initialEpsilon: float,
            epsilonDecay: float,
            finalEpsilon: float,
            discountFactor: float = 0.95, #gamma
    ):
        self.env = env
        self.qValues = defaultdict(lambda: np.zeros(env.action_space.n))
        self.learningRate = learningRate
        self.epsilon = initialEpsilon
        self.epsilonDecay = epsilonDecay
        self.finalEpsilon = finalEpsilon
        self.trainingError = []
        self.discountFactor = discountFactor

    def explore(self, obs: tuple[int, int, bool]) -> int:
        # eps-greedy policy
        if np.random.random() < self.epsilon: # with probability 1-epsilon
           # ==> choose a random a'
           return self.env.action_space.sample() 
        else: # with probability epsilon
            # ==> a_t+1 = max(Q(S_t+1, a')) for all a'
            return int(np.argmax(self.qValues[obs])) # obs = S_t
        # a_t = a_t+1

    def solve(self, obs: tuple[int, int, bool]) -> int:
        return int(np.argmax(self.qValues[obs])) # take best action
        
    def learn(
            self,
            obs: tuple[int, int, bool], #S_t
            action: int, #a_t
            reward: float, #r(S_t+1)
            terminated: bool,
            nextObs: tuple[int, int, bool], #S_t+1
    ):
        # NOTE: Q(S, a) initialized to 0's
        # NOTE: S_t --> a_t --> S_t+1

        # Q(S_t+1, a_t+1) = max(Q(S_t+1, a) forall a in a'
        nextQVal = (not terminated) * np.max(self.qValues[nextObs])
        
        # delta = r(t+1) + gamma              * Q(S_t+1, a_t+1) - Q(S_t, a_t)
        delta = (reward + self.discountFactor * nextQVal - self.qValues[obs][action])
        
        # Q(S_t, a_t)             = Q(S_t, a_t)               + alpha             * delta
        self.qValues[obs][action] = self.qValues[obs][action] + self.learningRate * delta
        
        self.trainingError.append(delta)

    def decayEpsilon(self):
        #                                                    lambda
        self.epsilon = max(self.finalEpsilon, self.epsilon - self.epsilonDecay)

class VAgent:
    def __init__(
            self,
            env: gym.Env,
            learningRate: float, #alpha
            initialEpsilon: float,
            epsilonDecay: float,
            finalEpsilon: float,
            discountFactor: float = 0.95, #gamma
    ):
        self.env = env
        self.qValues = defaultdict(lambda: np.zeros(env.action_space.n))
        self.vValues = np.zeros(env.observation_space.n)
        self.learningRate = learningRate
        self.epsilon = initialEpsilon
        self.epsilonDecay = epsilonDecay
        self.finalEpsilon = finalEpsilon
        self.trainingError = []
        self.discountFactor = discountFactor

    def explore(self, obs: tuple[int, int, bool]) -> int:
        # eps-greedy policy
        if np.random.random() < self.epsilon: # with probability 1-epsilon
           # ==> choose a random a'
           return self.env.action_space.sample() 
        else: # with probability epsilon
            # ==> a_t+1 = max(Q(S_t+1, a')) for all a'
            return int(np.argmax(self.qValues[obs])) # obs = S_t
        # a_t = a_t+1

    def solve(self, obs: tuple[int, int, bool]) -> int:
        return int(np.argmax(self.qValues[obs])) # take best action
        
    def learn(
            self,
            obs: tuple[int, int, bool], #S_t
            action: int, #a_t
            reward: float, #r(S_t+1)
            terminated: bool,
            nextObs: tuple[int, int, bool], #S_t+1
    ):

        nextVVal = (not terminated) * self.vValues[nextObs]
        
        # Use V to update values
        delta = (reward + self.discountFactor * nextVVal - self.vValues[obs])
        
        # Update q values
        self.qValues[obs][action] = self.vValues[obs] + self.learningRate * delta
        
        # Update V
        if(obs == 47):
            self.vValues[obs] = 0 # Terminal states stay 0
        else:
            self.vValues[obs] = np.max(self.qValues[obs])

        self.trainingError.append(delta)

    def decayEpsilon(self):
        #                                                    lambda
        self.epsilon = max(self.finalEpsilon, self.epsilon - self.epsilonDecay)

