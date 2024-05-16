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
from os import listdir
from pyrep import PyRep
from pyrep.robots.legged_robot.B1 import B1
from pyrep.objects.shape import Shape
import numpy as np
import timeit
import random
from multiprocessing import Process
from multiprocessing import Pool

# Lib for Evolutional strategy algorithm
# from hebbian_neural_net import HebbianNet
# from feedforward_neural_net import FeedForwardNet
# from rbf_neural_net import RBFNet
from simple_cpg import CPGNet
from ES_classes import OpenES
from rollout_B1 import fitness

# libraries for Multi-process
import concurrent.futures
import copy

# libraries for array manipulation, math cal, visualization
import pickle
import numpy as np
import yaml
import wandb


# open config files for reading prams
with open('config.yml', 'r') as file:
    configs = yaml.safe_load(file)

# CPU cores for training 
cpus = configs['Device']['CPU_NUM']
TASK_PER_IND = configs['Device']['TASK_PER_IND']

# ES parameters configuration
POPSIZE             = configs['ES_params']['POPSIZE']
EPISODE_LENGTH      = configs['ES_params']['EPISODE_LENGTH']
REWARD_FUNCTION     = configs['ES_params']['REWARD_FUNC']
RANK_FITNESS        = configs['ES_params']['rank_fitness']
ANTITHETIC          = configs['ES_params']['antithetic']
LEARNING_RATE       = configs['ES_params']['learning_rate']
LEARNING_RATE_DECAY = configs['ES_params']['learning_rate_decay']
SIGMA_INIT          = configs['ES_params']['sigma_init']
SIGMA_DECAY         = configs['ES_params']['sigma_decay']
LEARNING_RATE_LIMIT = configs['ES_params']['learning_rate_limit']
SIGMA_LIMIT         = configs['ES_params']['sigma_limit']

# # Training parameters
EPOCHS = configs['Train_params']['EPOCH']
# EVAL_EVERY = configs['Train_params']['EVAL_EVERY']
SAVE_EVERY = configs['Train_params']['SAVE_EVERY']
# USE_TRAIN_WEIGHT = configs['Model']['USE_TRAIN_WEIGHT']

# # Model
# ARCHITECTURE_NAME = configs['Model']['TYPE']
# ARCHITECTURE = configs['ARCHITECTURE']['size']

# # scene file destination
ENV_NAME = configs['ENV']['NAME']

# # WanDB Log
use_Wandb = configs['Wandb_log']['use_WanDB']
# config_wandb = configs

# if use_Wandb:
#     wandb.init(project='dbAlpha_wandb_log', 
#            config=config_wandb)

# print("CPU_num: ", cpus)
# print("POPSIZE: ", POPSIZE)
# print("EPISODE_LENGTH: ", EPISODE_LENGTH)
# print("REWARD_FUNCTION: ", REWARD_FUNCTION)
# print("ARCHITECTURE_NAME: ", ARCHITECTURE_NAME)
# print("ARCHITECTURE_size: ", ARCHITECTURE)
# print("ENV_NAME: ", ENV_NAME)

# initial_time = timeit.default_timer()
# print("initial_time", initial_time)

runs = ['d_']
for run in runs:

    # Initialize Selected Model
    # if ARCHITECTURE_NAME == 'FEEDFORWARD':
    #     dir_path = './data/model/FF/'
    #     init_net = FeedForwardNet(ARCHITECTURE)
    # elif ARCHITECTURE_NAME == 'HEBBIAN':
    #     dir_path = './data/model/HEBB/'
    #     init_net = HebbianNet(ARCHITECTURE)
    # elif ARCHITECTURE_NAME == 'RBF':
    #     dir_path = './data/model/RBF/'
    #     init_net = RBFNet([20, 18])
    dir_path = './data/B1/'

#     # Using weight result from previous training
#     if USE_TRAIN_WEIGHT:
#         res = listdir(dir_path)
#         trained_data = pickle.load(open(dir_path+res[-1], 'rb'))
#         open_es_data = trained_data[0]
#         init_params = open_es_data.mu
#         init_net.set_params(init_params)
#     else:
#         init_params = init_net.get_params()

#     print('trainable parameters: ', len(init_params))

#     with open('log_'+str(run)+'.txt', 'a') as outfile:
#         outfile.write('trainable parameters: ' + str(len(init_params)))

    init_net = CPGNet()
    init_params = init_net.get_params()
    print('init_params: ', init_params)

    solver = OpenES(len(init_params),
                    popsize=POPSIZE,
                    rank_fitness=RANK_FITNESS,
                    antithetic=ANTITHETIC,
                    learning_rate=LEARNING_RATE,
                    learning_rate_decay=LEARNING_RATE_DECAY,
                    sigma_init=SIGMA_INIT,
                    sigma_decay=SIGMA_DECAY,
                    learning_rate_limit=LEARNING_RATE_LIMIT,
                    sigma_limit=SIGMA_LIMIT)
    solver.set_mu(init_params)

    def worker_fn(params):
        mean = 0
        for epi in range(TASK_PER_IND):
            net = CPGNet()
            init_net.set_params(params)
            mean += fitness(net, ENV_NAME, EPISODE_LENGTH, REWARD_FUNCTION) 
        return mean/TASK_PER_IND
    
    pop_mean_curve = np.zeros(EPOCHS)
    best_sol_curve = np.zeros(EPOCHS)
    eval_curve = np.zeros(EPOCHS)


    for epoch in range(EPOCHS):
        start_time = timeit.default_timer()
        # print("start_time", start_time)

        solutions = solver.ask()
        print('TEST')
        print(solutions)

        with concurrent.futures.ProcessPoolExecutor(cpus) as executor:
            fitlist = executor.map(worker_fn, [params for params in solutions])
        # with Pool(cpus) as p:
        #     fitlist = p.map(worker_fn, [params for params in solutions])

        # processes = [Process(target=worker_fn, args=[params for params in solutions]) for i in solutions]
        # [p.start() for p in processes]
        # [p.join() for p in processes]
        
        fitlist = list(fitlist)
        solver.tell(fitlist)

        fit_arr = np.array(fitlist)

        print('epoch', epoch, 'mean', fit_arr.mean(), "best", fit_arr.max(), )
        with open('log_'+str(run)+'.txt', 'a') as outfile:
            outfile.write('epoch: ' + str(epoch)
                    + ' mean: ' + str(fit_arr.mean())
                    + ' best: ' + str(fit_arr.max())
                    + ' worst: ' + str(fit_arr.min())
                    + ' std.: ' + str(fit_arr.std()) + '\n')
            
        pop_mean_curve[epoch] = fit_arr.mean()
        best_sol_curve[epoch] = fit_arr.max()

        # # WanDB Log data -------------------------------
        # if use_Wandb:
        #     wandb.log({"epoch": epoch,
        #                 "mean" : np.mean(fitlist),
        #                 "best" : np.max(fitlist),
        #                 "worst": np.min(fitlist),
        #                 "std"  : np.std(fitlist),
        #                 })
        # -----------------------------------------------
        # '''
        if (epoch + 1) % SAVE_EVERY == 0:
            print('saving..')
            pickle.dump((
                solver,
                copy.deepcopy(init_net),
                pop_mean_curve,
                best_sol_curve,
                ), open(dir_path+str(run)+'_' + str(len(init_params)) + str(epoch) + '_' + str(pop_mean_curve[epoch]) + '.pickle', 'wb'))
        
        # stop_time = timeit.default_timer()
        # print("running time per epoch: ", stop_time-start_time)

        # '''

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