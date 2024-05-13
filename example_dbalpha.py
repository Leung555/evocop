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
from dbAlphaEnv import dbAlphaEnv
from pyrep.objects.shape import Shape
import numpy as np
import timeit
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
import matplotlib.pyplot as plt


def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
     
        return [roll_x, pitch_y, yaw_z] # in radians

# scene file destination
SCENE_FILE = join(dirname(abspath(__file__)),
                  './scenes/scene_dbAlpha_test.ttt')
                #   './scenes/scene_slalom_fixedbody_terrain_real_u2004.ttt')

# ES parameters configuration
EPISODES = 2
EPISODE_LENGTH = 50

# initiate Coppeliasim simulation
# pr = PyRep()
# pr.launch(SCENE_FILE, headless=False)
# agent = dbAlpha()
# PROCESSES = 5

# ARCHITECTURE
inp_size = 2 # joint angles (18 joints)
action_size = 1 # joint angles (18 joints)
hidden_neuron_num1 = 4
hidden_neuron_num2 = 2

ARCHITECTURE = [inp_size, 
                hidden_neuron_num1, 
                hidden_neuron_num2, 
                action_size]

# CPU cores for training 
cpus = 6

# Training parameters
EPOCHS = 1
TASK_PER_IND = 1
EVAL_EVERY = 2
popsize = 2

runs = ['d_']

initial_time = timeit.default_timer()
print("initial_time", initial_time)

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
            # Test simulation speed by printing simulation step
            # pr = PyRep()
            # pr.launch(SCENE_FILE, headless=True)
            # pr.start()
            # print('---start simulation---')

            # for i in range(EPISODE_LENGTH):
            #     print("step: ", i)

            #     # step simulation
            #     pr.step()

            # # stop simulation
            # print('---stop simulation---')            
            # # time.sleep(2)
            # pr.stop()
            # pr.shutdown()
            # mean += 5 # TODO test reward value
            # print('mean: ', mean)
            #######################################

            ###################################
            # Test function in simulation
            pr = PyRep()
            pr.launch(SCENE_FILE, headless=False)
            # print("launch sim")
            pr.start()
            agent = dbAlpha()
            # print('---start simulation---')
            done = False
            r_tot = 0
            counter = 0
            tilt_penalty = 0
            robot_quartenion_series = np.empty(shape=[EPISODE_LENGTH, 4])
            robot_euler_series = np.empty(shape=[EPISODE_LENGTH, 3])
            robot_position_series = np.empty(shape=[EPISODE_LENGTH, 3])
            # print("robot_quartenion_series: ", robot_quartenion_series)
            while not done:
                print("step: ", counter)
                if counter+1 > EPISODE_LENGTH:
                    done = True
                    break
                
                # Sensing
                # joint_angles = np.array(env.agent.get_joint_positions())
                # # print("joint_angles: ", joint_angles)
                robot_position = agent.get_position()
                robot_orient = agent.get_quaternion()
                robot_leg_torq = agent.get_leg_joint_forces()
                print("robot_leg_torq: ", robot_leg_torq)
                # robot_quartenion_series[counter] = robot_orient
                x = robot_orient[0]
                y = robot_orient[1]
                z = robot_orient[2]
                w = robot_orient[3]
                robot_euler_series[counter] = euler_from_quaternion(x,y,z,w)
                roll    = robot_euler_series[counter][0]
                pitch   = robot_euler_series[counter][1]
                yaw     = robot_euler_series[counter][2]                # # print(test)
                # # print("action: ", action)
                # print("robot_position: ", robot_position_series[counter])
                # print("robot_orientation: ", robot_orient)

                # obs, r, done, _ = env.step(action)
                pr.step()
                
                # Calculate reward from sensor
                tilt_lim_rad = 0.35 # (-20,+20) degrees
                if abs(roll) > tilt_lim_rad or abs(pitch) > tilt_lim_rad:
                    tilt_penalty -= 0.4

                counter += 1
            # print("robot_quartenion_series: ", robot_quartenion_series)
            # print("robot_euler_series: ", robot_euler_series)

            r_tot += robot_position[1]+tilt_penalty # y axis distance
            mean += r_tot
            print('r_total', r_tot)
            print('robot_position', robot_position[1])
            print('tilt_penalty', tilt_penalty)
            # env.stop_simulation()
            pr.stop()
            pr.shutdown()
            # print('---shutdown simulation---')

            # Visualize data
            # print(robot_position_series)
            plt.figure()
            plt.plot(robot_euler_series[:,0])
            plt.plot(robot_euler_series[:,1])
            plt.plot(robot_euler_series[:,2])
            plt.show()
            ####################################

        return mean/TASK_PER_IND
    
    pop_mean_curve = np.zeros(EPOCHS)
    best_sol_curve = np.zeros(EPOCHS)
    eval_curve = np.zeros(EPOCHS)

    for epoch in range(EPOCHS):
        start_time = timeit.default_timer()
        print("start_time", start_time)

        solutions = solver.ask()

        reward = worker_fn(solutions[0])
        # with concurrent.futures.ProcessPoolExecutor(cpus) as executor:
        #     fitlist = executor.map(worker_fn, [params for params in solutions])

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
        stop_time = timeit.default_timer()
        print("running time per epoch: ", stop_time-start_time)

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

