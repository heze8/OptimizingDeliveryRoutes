"""
Using hyperneat to evolve .
"""

from __future__ import print_function
import os
import neat
import neat.nn
import visualize    
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle
from DeliveryMapAuto import MultiAgentDeliveryEnv
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat import ESNetwork
import random
import heapq

SIZE = 20
input_coordinates = []
for i in range(SIZE):
    x = i * 1.9/(SIZE - 1) - 0.95
    for j in range(SIZE):
        y = j * 1.9/(SIZE - 1) - 0.95
        input_coordinates.append((x, y))

input_coordinates.append((0.0, -1.0)) #distance
input_coordinates.append((1.0, 1.0)) #bias

#include a bias
output_coordinates = [(0.0, 1.0)] #must me this coor or it bugs out
sub = Substrate(input_coordinates, output_coordinates)

params = {"initial_depth": 2,
          "max_depth": 4,
          "variance_threshold": 0.4,
          "band_threshold": 0.3,
          "iteration_level": 1,
          "division_threshold": 0.5,
          "max_weight": 15.0,
          "activation": "sigmoid"}

def convert(list): 
    return tuple(float(i)/2 for i in list) 

# def train(net, network, render, env = MultiAgentDeliveryEnv()):
#     episode_reward = 0 
#     step = 1
#     current_state = env.reset()
#     done = False
#     net.reset()
    
#     while not done and step < 999:
#         numAction = []
#         for action in current_state:
#             action = np.append(action, [1]) #bias
#             action = convert(action)
#             for k in range(network.activations):
#                 o = net.activate(action)
#             numAction.append(o)
#         action = np.argmax(numAction)
#         new_state, reward, done = env.step(action)
#         if render:
#             env.render(500) 
#             print(action)
#         episode_reward += reward

#         current_state = new_state
#         step += 1

#     if render:
#         print(episode_reward)
#     # Append episode reward to a list and log stats (every given number of episodes)
#     return episode_reward

def trainMultiple(genomes, config, render, env):
    step = 0
    current_state = env.reset()
    done = False
    nets = []
    networks = []
    rewards = np.zeros(len(genomes))

    for genome_id, genome in genomes:
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        network = ESNetwork(sub, cppn, params)
        net = network.create_phenotype_network()
        nets.append(net)
        networks.append(network)

    pq = []
    for i in range(len(nets)):    
        heapq.heappush(pq, (0, i))

    while not done and step < 1000:
        time, index = heapq.heappop(pq)
        network = networks[index]
        net = nets[index]
        numAction = []
        for action in current_state:
            action = np.append(action, [1]) #bias
            action = convert(action)
            for k in range(network.activations):
                o = net.activate(action)
            numAction.append(o)
        action = np.argmax(numAction)
        new_state, reward, done, distance = env.stepM(action, index)
        if render:
            env.render(100) 
            print(action)
        rewards[index] += reward

        heapq.heappush(pq, (time + distance, index))
        current_state = new_state
        step += 1

    if render:
        print(rewards)
    # Append episode reward to a list and log stats (every given number of episodes)
    return rewards

# def eval_genome(genome, config):
#     cppn = neat.nn.FeedForwardNetwork.create(genome, config)
#     network = ESNetwork(sub, cppn, params)
#     net = network.create_phenotype_network()
#     episode_reward = 0
#     runs = 10
#     for i in range(runs):
#         episode_reward += train(net, network, False)

#     fitness = episode_reward/runs
#     # Append episode reward to a list and log stats (every given number of episodes)
#     return fitness
 

# def eval_genomes(genomes, config):
#     best_net = (None, None, -9999)
#     runs = 10
#     environments = [MultiAgentDeliveryEnv() for i in range(runs)]

#     for genome_id, genome in genomes:

#         cppn = neat.nn.FeedForwardNetwork.create(genome, config)
#         network = ESNetwork(sub, cppn, params)
#         net = network.create_phenotype_network()
#         episode_reward = 0
#         genome.fitness = 0

#         for i in range(runs):
#             episode_reward += train(net, network, False, environments[i])

#         fitness = episode_reward/runs
#         if fitness > best_net[2]:
#             best_net = (net, network, fitness)
#         # Append episode reward to a list and log stats (every given number of episodes)
#         genome.fitness += fitness
#     for i in range(4):
#         train(best_net[0], best_net[1], True)

def eval_multipleGenomes(genomes, config):
    runs = 10
    environments = [MultiAgentDeliveryEnv(len(genomes)) for i in range(runs)]
    rewards = np.zeros(len(genomes))

    for i in range(runs):
        rewards += trainMultiple(genomes, config, False, environments[i])

    rewards = rewards/runs
    i = 0
    for genome_id, genome in genomes:
        fitness = rewards[i]
        genome.fitness = 0
        genome.fitness += fitness
        i+=1

    trainMultiple(genomes, config, True, environments[0])


def run(config_file):
    # Load configuration.
    config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            config_file)

    # Create the population, which is the top-level object for a NEAT run.
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-15')

    # Add a stdout reporter to show progress in the terminal.
    pop = neat.population.Population(config)
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(999))
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-299')

    # Run for up to 300 generations.
    winner = pop.run(eval_multipleGenomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    cppn = neat.nn.FeedForwardNetwork.create(winner, config)
    network = ESNetwork(sub, cppn, params)
    winner_net = network.create_phenotype_network(filename='es_hyperneat_winner.png') 
    input("Winner is found")
    for i in range(10):
        train(winner_net, network, True)

    draw_net(cppn, filename="es_hyperneat")

    #visualize.draw_net(config, winner, True, node_names=node_names)
    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'Config')
    run(config_path)
    # config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
    #                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
    #                      config_path)
    # p = neat.Population(config)

    # p.add_reporter(neat.StdOutReporter(True))
    # stats = neat.StatisticsReporter()
    # p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-24')
    # winner = p.run(eval_genomes, 1)
    # cppn = neat.nn.RecurrentNetwork.create(winner, config)
    # network = ESNetwork(sub, cppn, params)
    # winner_net = network.create_phenotype_network(filename='es_hyperneat_winner.png') 
    # for i in range(10):
    #     train(winner_net, True)
