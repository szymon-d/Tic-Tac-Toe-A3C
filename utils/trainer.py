from agent import Agent
from settings import CPU_CORE_AMOUNT
from torch import multiprocessing as mp
from settings import settings
from torch.multiprocessing import freeze_support
from network import global_network
import torch

def train(games2play: int, number_of_agents = CPU_CORE_AMOUNT):

    #Initial share variable
    games_played = torch.multiprocessing.Value('i', 0)

    #Initial list with agents
    agents = [Agent(games_played=games_played,
                    games2play=games2play) for _ in range(number_of_agents)]

    #Trigger each agent
    for agent in agents:
        agent.start()

    #Wait until each agent does his job
    for agent in agents:
        agent.join()


if __name__ == '__main__':
    freeze_support()
    train(100)

    #Save global model
    global_network.save()





