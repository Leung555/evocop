"""
# *******************************************
# *                                         *
# *         dbAlpha hebbbian                *
# *                                         *
# *******************************************
#  created by: Joachim Winther Pedersen
#  1st modified by: Worasuchad Haomachai
#  2nd modified by: Binggwong Leung
#  contract: haomachai@gmail.com
#  update: 01/05/2023
#  version: 0.1.0


# *******************************************
# *                                         *
# *               description               *
# *                                         *
# *******************************************
#  Agent: dung beetle-like robot (dbAlpha)
#  Nets: FFN
#  Meta-Lerning: ABCD Hebbain, wo/ Hebbian
#  Optimization: OpenES  
"""

"""
An example of how one might use PyRep to create their RL environments.
In this case.
This script contains examples of:
    - RL environment example.
    - Scene manipulation.
    - Environment resets.
    - Setting joint properties (control loop disabled, motor locked at 0 vel)

This code is modified/based on "example_reinforcement_learning_env.ttt" 
"""
# libraries for simulation environment
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.legged_robots.dbAlpha import dbAlpha
from pyrep.objects.shape import Shape
import numpy as np
import time
import random
from multiprocessing import Process

# Lib for Evolutional strategy algorithm
from hebbian_neural_net import HebbianNet
from ES_classes import OpenES

# libraries for Multi-process
import concurrent.futures
import copy

# libraries for array manipulation, math cal, visualization
import pickle
import numpy as np
import yaml
import wandb

# scene file destination
SCENE_FILE = join(dirname(abspath(__file__)),
                  './scenes/scene_dbAlpha_test.ttt')
                #   './scenes/scene_slalom_fixedbody_terrain_real_u2004.ttt')

# ES parameters configuration
EPISODES = 2
EPISODE_LENGTH = 100

# initiate Coppeliasim simulation
# pr = PyRep()
# pr.launch(SCENE_FILE, headless=False)
# agent = dbAlpha()
# PROCESSES = 5

# ARCHITECTURE
inp_size = 18 # joint angles (18 joints)
action_size = 18 # joint angles (18 joints)
hidden_neuron_num1 = 2
hidden_neuron_num2 = 2

ARCHITECTURE = [inp_size, 
                hidden_neuron_num1, 
                hidden_neuron_num2, 
                action_size]

# CPU cores for training 
cpus = 1

# Training parameters
EPOCHS = 1
TASK_PER_IND = 1
EVAL_EVERY = 2
popsize = 2

runs = ['d_']
for run in runs:

    init_net = HebbianNet(ARCHITECTURE)

    init_params = init_net.get_params()

    print('trainable parameters: ', len(init_params))

    with open('log_'+str(run)+'.txt', 'a') as outfile:
        outfile.write('trainable parameters: ' + str(len(init_params)))

    solver = OpenES(len(init_params),
                    popsize=popsize,
                    rank_fitness=True,
                    antithetic=True,
                    learning_rate=0.01,
                    learning_rate_decay=0.9999,
                    sigma_init=0.1,
                    sigma_decay=0.999,
                    learning_rate_limit=0.001,
                    sigma_limit=0.01)
    solver.set_mu(init_params)

    def worker_fn(params):
        mean = 0
        for epi in range(TASK_PER_IND):
            net = HebbianNet(ARCHITECTURE)
            net.set_params(params)
            
            #####################################
            # Test fitness in the simulation    #
            #####################################
            # Do some stuffs
            # time.sleep(2)
            pr = PyRep()
            pr.launch(SCENE_FILE, headless=True)
            # agent = dbAlpha()
            pr.start()
            print('---start simulation---')

            for i in range(EPISODE_LENGTH):
                print("step: ", i)

                # Sensing
                # joint_angles = np.array(agent.get_joint_positions())
                # action = net.forward(joint_angles)
                # action = net.forward(obs)
                # target_pos = []
                # for i in range(18): 
                #     temp = random.gauss(0, 1)
                #     target_pos.append(temp)
                    
                # # Actuation
                # agent.set_joint_positions(action)
                # print("action: ",action )

                # step simulation
                pr.step()

                # replay_buffer.append((state, action, reward, next_state))
                # state = next_state
                # agent.learn(replay_buffer)

            # stop simulation
            print('---stop simulation---')            
            # time.sleep(2)
            pr.stop()
            pr.shutdown()
            mean += 5 # TODO test reward value
            print('mean: ', mean)
            # done = False
            # r_tot = 0
            # while not done:
            #     action = net.forward(obs)
            #     obs, r, done, _ = env.step(action)
            #     r_tot += r
            # env.close()
            # return r_tot
            # mean += fitness(net, ENV_NAME) # FIXME ############
            
            ####################################3
        return mean/TASK_PER_IND
    
    pop_mean_curve = np.zeros(EPOCHS)
    best_sol_curve = np.zeros(EPOCHS)
    eval_curve = np.zeros(EPOCHS)

    for epoch in range(EPOCHS):

        solutions = solver.ask()

        with concurrent.futures.ProcessPoolExecutor(cpus) as executor:
            fitlist = executor.map(worker_fn, [params for params in solutions])

        # fitlist = list(fitlist)
        # solver.tell(fitlist)

        # fit_arr = np.array(fitlist)

        # print('epoch', epoch, 'mean', fit_arr.mean(), "best", fit_arr.max(), )
        # with open('log_'+str(run)+'.txt', 'a') as outfile:
        #     outfile.write('epoch: ' + str(epoch)
        #             + ' mean: ' + str(fit_arr.mean())
        #             + ' best: ' + str(fit_arr.max())
        #             + ' worst: ' + str(fit_arr.min())
        #             + ' std.: ' + str(fit_arr.std()) + '\n')
            
        # pop_mean_curve[epoch] = fit_arr.mean()
        # best_sol_curve[epoch] = fit_arr.max()

        # #'''
        # if (epoch + 1) % 500 == 0:
        #     print('saving..')
        #     pickle.dump((
        #         solver,
        #         copy.deepcopy(init_net),
        #         pop_mean_curve,
        #         best_sol_curve,
        #         ), open(str(run)+'_' + str(len(init_params)) + str(epoch) + '_' + str(pop_mean_curve[epoch]) + '.pickle', 'wb'))

        #'''

    # ------------- end training epoch --------------#

    
     
# ------------- end training episode --------------#


# # Do some stuff
# for e in range(EPISODES):
#     # start simulation
#     pr.start()

#     print('Starting episode %d' % e)

#     for i in range(EPISODE_LENGTH):
#         print("episode: {}", e)
#         print("step:    {}", i)
#         pr.step()
#         # replay_buffer.append((state, action, reward, next_state))
#         # state = next_state
#         # agent.learn(replay_buffer)

#     # stop simulation
#     pr.stop()

# pr.stop()
# pr.shutdown()