from matplotlib import pyplot as plt
import numpy as np

import datetime

def getMovingAvgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode = convolution_mode
    ) / window


def plot(agents, envs):
    rollingLength = 500
    fig, axs = plt.subplots(ncols = 3, figsize = (12, 5))
    
    colors = ["tab:pink", "tab:green"]
    labels = ["Q-Learning", "Value Function"]

    for i in range (len(agents)):
        agent = agents[i]
        env = envs[i]

        # Ep Rewards
        axs[0].set_title("Episode Rewards")
        rewardMovingAvgs = getMovingAvgs(
            env.return_queue,
            rollingLength,
            "valid"
        )
        axs[0].plot(range(len(rewardMovingAvgs)), rewardMovingAvgs, colors[i])

        # Ep Lengths
        axs[1].set_title("Episode Lengths")
        length_moving_average = getMovingAvgs(
            env.length_queue,
            rollingLength,
            "valid"
        )
        axs[1].plot(range(len(length_moving_average)), length_moving_average, color=colors[i])

        #Training Err
        axs[2].set_title("Training Error")
        trainErrMovingAvgs = getMovingAvgs(
            agent.trainingError,
            rollingLength,
            "same"
        )
        axs[2].plot(range(len(trainErrMovingAvgs)), trainErrMovingAvgs, colors[i])
        plt.tight_layout()

    timeStamp = str(datetime.datetime.now()).replace(" ", "T")
    timeStamp = timeStamp[:timeStamp.index(".")]
    plt.legend(labels)

    plt.savefig(f"./plots/fig_{timeStamp}.png")