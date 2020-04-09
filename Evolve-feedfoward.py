"""
Using neat to evolve .
"""

from __future__ import print_function
import os
import neat
import visualize
import numpy as np
from Grid import Grid

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

def train(net, render):
    episode_reward = 0 
    step = 1
    env = Grid()
    current_state = env.reset()
    done = False
    
    
    while not done and step < 999:
        current_state = current_state.flatten()
        action = np.argmax(net.activate(current_state))
        
        new_state, reward, done = env.step(action)
        if render:
            env.render(3)
        episode_reward += reward

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    return episode_reward

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0

        net = neat.nn.FeedForwardNetwork.create(genome, config)

        episode_reward = train(net, False)

        # Append episode reward to a list and log stats (every given number of episodes)
        genome.fitness += episode_reward


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
    p.add_reporter(neat.Checkpointer(10))
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-299')

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    train(winner_net, True)

    node_names = {-1:'A', -2: 'B', 0:'Delivery'}
    #visualize.draw_net(config, winner, True, node_names=node_names)
    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)




if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'Config')
    # run(config_path)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-299')
    winner = p.run(eval_genomes, 10)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    train(winner_net,True)