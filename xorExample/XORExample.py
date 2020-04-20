import neat 
import neat.nn
try:
   import cPickle as pickle
except:
   import pickle
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat import ESNetwork

# Network inputs and expected outputs.
xor_inputs  = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [    (0.0,),     (1.0,),     (1.0,),     (0.0,)]

input_coordinates  = [(-1.0, -1.0),(0.0, -1.0),(1.0, -1.0)]
output_coordinates = [(0.0, 1.0)]

sub = Substrate(input_coordinates, output_coordinates)

# ES-HyperNEAT specific parameters.
params = {"initial_depth": 2, 
          "max_depth": 3, 
          "variance_threshold": 0.03, 
          "band_threshold": 0.3, 
          "iteration_level": 1,
          "division_threshold": 0.5, 
          "max_weight": 5.0, 
          "activation": "sigmoid"}

# Config for CPPN.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn_xor')


def eval_fitness(genomes, config):
    for idx, g in genomes:
        cppn = neat.nn.FeedForwardNetwork.create(g, config)
        network = ESNetwork(sub, cppn, params)
        net = network.create_phenotype_network()
        
        sum_square_error = 0.0
        for inputs, expected in zip(xor_inputs, xor_outputs):
            new_input = inputs + (1.0,)
            net.reset()
            for i in range(network.activations):
                output = net.activate(new_input)

            sum_square_error += ((output[0] - expected[0])**2.0)/4.0
 
        g.fitness = 1 - sum_square_error

from DeliveryMapAuto import MultiAgentDeliveryEnv

def convert(list): 
    return tuple(float(i)/2 for i in list) 

def train(net, network, render, env = MultiAgentDeliveryEnv()):
    episode_reward = 0 
    step = 1
    current_state = env.reset()
    done = False
    net.reset()
    
    while not done and step < 999:
        numAction = []
        for action in current_state:
            action = np.append(action, [1]) #bias
            action = convert(action)
            for k in range(network.activations):
                o = net.activate(action)
            numAction.append(o)
        action = np.argmax(numAction)
        new_state, reward, done = env.step(action)
        if render:
            env.render(500) 
            print(action)
        episode_reward += reward

        current_state = new_state
        step += 1

    if render:
        print(episode_reward)
    # Append episode reward to a list and log stats (every given number of episodes)
    return episode_reward


def eval_genomes(genomes, config):
    best_net = (None, None, -9999)
    runs = 10
    environments = [MultiAgentDeliveryEnv() for i in range(runs)]

    for genome_id, genome in genomes:

        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        network = ESNetwork(sub, cppn, params)
        net = network.create_phenotype_network()
        episode_reward = 0
        genome.fitness = 0

        sum_square_error = 0.0
        for inputs, expected in zip(xor_inputs, xor_outputs):
            new_input = inputs + (1.0,)
            net.reset()
            for i in range(network.activations):
                output = net.activate(new_input)

            sum_square_error += ((output[0] - expected[0])**2.0)/4.0
        fitness = episode_reward/runs
        if fitness > best_net[2]:
            best_net = (net, network, fitness)
        # Append episode reward to a list and log stats (every given number of episodes)
        genome.fitness += fitness
    for i in range(4):
        train(best_net[0], best_net[1], True)

# Create the population and run the XOR task by providing the above fitness function.
def run(gens):
    pop = neat.population.Population(config)
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))

    winner = pop.run(eval_fitness, gens)
    print("es_hyperneat_xor_large done")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    winner = run(300)[0]
    print('\nBest genome:\n{!s}'.format(winner))

    # Verify network output against training data.
    print('\nOutput:')
    cppn = neat.nn.FeedForwardNetwork.create(winner, config)
    network = ESNetwork(sub, cppn, params)
    winner_net = network.create_phenotype_network(filename='es_hyperneat_xor_large_winner.png')  # This will also draw winner_net.
    for inputs, expected in zip(xor_inputs, xor_outputs):
        new_input = inputs + (1.0,)
        winner_net.reset()
        for i in range(network.activations):
            output = winner_net.activate(new_input)
        print("  input {!r}, expected output {!r}, got {!r}".format(inputs, expected, output))

    # Save CPPN if wished reused and draw it to file.
    draw_net(cppn, filename="es_hyperneat_xor_large_cppn")
    with open('es_hyperneat_xor_large_cppn.pkl', 'wb') as output:
        pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)