import gymnasium as gym
import numpy as np

from tqdm import tqdm

from RlAgent import QAgent, VAgent
import plots

# Hyperparameters
learningRate = 0.01
nEpisodes = 10_000 #[3_500, 50_000]
startEpsilon = 1.0
#epsilonDecay = startEpsilon / (nEpisodes / 2)
finalEpsilon = 0.1
# Note: tutorial suggests increasing nEpisodes by 10x & lowering learningRate to ~0.001

# Make qEnviornment
qEnv = gym.make('CliffWalking-v0')
qEnv = gym.wrappers.RecordEpisodeStatistics(qEnv, buffer_length=nEpisodes)

vEnv = gym.make('CliffWalking-v0')
vEnv = gym.wrappers.RecordEpisodeStatistics(vEnv, buffer_length=nEpisodes)


nCols = 12

# Instantiate RL Agents
qAgent = QAgent(
    env =               qEnv,
    learningRate =      0.01,
    initialEpsilon =    1.0,
    epsilonDecay =      startEpsilon / (nEpisodes / 2),
    finalEpsilon =      0.1
)

vAgent = VAgent(
    env             = vEnv,
    learningRate    = 0.01,
    initialEpsilon  = 1.0,
    epsilonDecay    = startEpsilon / (nEpisodes / 2),
    finalEpsilon    = 0.1
)

agents = [qAgent, vAgent]
envs = [qEnv, vEnv]
labels = ["Q-Learning", "Value Function"]

actions = {
    "UP": 0,
    "RIGHT": 1,
    "DOWN": 2,
    "LEFT": 3
}

solutions = []

for i in range (len(agents)):
    agent = agents[i]
    env = envs[i]

    # Training error tracking for debug purposes
    prevErr = None
    curErr = None

    #TRAINING
    # Run episodes N times
    for episode in tqdm(range(nEpisodes)):
        #debug.write(f"STARTING EPISODE {episode}\n")

        obs, info = env.reset()
        done = False

        prevErr = curErr

        steps = 0
        # Run until goal or failure reached
        while not done:
            #a_t = a_t+1 -- Given by current policy (epsilon-greedy)
            action = agent.explore(obs)
            
            #Take action a_t
            #S_t+1   r(t+1)                                          a_t
            nextObs, reward, terminated, truncated, info = env.step(action)

            #             S_t  a_t     r(t+1)              S_t+1
            agent.learn(obs, action, reward, terminated, nextObs)

            #S_t = S_t+1
            obs = nextObs
            steps += 1 # Counting steps taken for debug purposes

            done = terminated or truncated
        
        try: agent.decayEpsilon()
        except: pass

        curErr = agent.trainingError[-1]
        # END EPISODE LOOP

    # END TRAINING LOOP

      
    #RUN AFTER TRAINING
    moves = []
    obs, info = env.reset()
    done = False

    while not done:
        #a_t = a_t+1 -- Given by optimal policy (pi*)
        action = agent.solve(obs)

        #Take action a_t
        #S_t+1   r(t+1)                                          a_t
        nextObs, reward, terminated, truncated, info = env.step(action)
        
        # Keep track of solution data
        moves.append(
            {
                "obs": {obs: (obs%12, np.floor(obs/12))},
                "action": action,
                "nextObs": {nextObs: (nextObs%12, np.floor(nextObs/12))},

            }
        )
        solutions.append(moves)

        #S_t = S_t+1
        obs = nextObs

        done = terminated or truncated

    # END SOLUTION LOOP

    env.close()
# END AGENTS LOOP

plots.plot(agents, envs)

# Print agent solutions to terminal
for i in range (len(agents)):
    moves = solutions[i]

    print(f"{labels[i]} agent reached end in {len(moves)} moves")
    for move in moves:
        print(f"{move['obs']} --> {list(actions.keys())[move['action']]} --> {move['nextObs']}")