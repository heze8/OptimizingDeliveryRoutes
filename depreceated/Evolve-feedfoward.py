"""
Using neat to evolve .
"""

from __future__ import print_function
import os
import neat
import visualize
import numpy as np
from DeliveryMap import MultiAgentDeliveryEnv


def train(net, render):
    episode_reward = 0 
    step = 1
    env = MultiAgentDeliveryEnv()
    current_state = env.reset()
    done = False
    
    
    while not done and step < 999:
        current_state = current_state.flatten()
        action = np.argmax(net.activate(current_state))

        new_state, reward, done = env.step(action)
        if render:
            env.render(100) 
            print(action)
        episode_reward += reward

        current_state = new_state
        step += 1

    if render:
        print(episode_reward)
    # Append episode reward to a list and log stats (every given number of episodes)
    return episode_reward

def eval_genomes(genomes, config):
    best_net = (None, -9999)
    for genome_id, genome in genomes:
        net = neat.nn.RecurrentNetwork.create(genome, config)
        episode_reward = 0
        runs = 3
        genome.fitness = 0

        for i in range(runs):
            net.reset()
            episode_reward += train(net, False)

        fitness = episode_reward/runs
        if fitness > best_net[1]:
            best_net = (net, fitness)
        # Append episode reward to a list and log stats (every given number of episodes)
        genome.fitness += fitness
    
    # train(best_net[0], True)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1000))
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-299')

    # Run for up to 300 generations.
    # pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genomes)
    winner = p.run(eval_genomes, 1000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    input("Winner is found")
    for i in range(10):
        train(winner_net, True)

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
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-290')
    # winner = p.run(eval_genomes, 1)
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # train(winner_net,True)