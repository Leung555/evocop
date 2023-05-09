from pyrep import PyRep
from pyrep.robots.legged_robots.dbAlpha import dbAlpha
import numpy as np
from hebbian_neural_net import HebbianNet
from os.path import dirname, join, abspath
from dbAlphaEnv import dbAlphaEnv

def fitness(net: HebbianNet, env_name: str, 
            episode_length: int, reward_function: str) -> float:
    # print(env_name)
    env = dbAlphaEnv(env_name)
    env.start()
    # print('---start simulation---')
    obs = env.reset()
    done = False
    r_tot = 0
    counter = 0
    while not done:
        # print("step: ", counter)
        
        # Sensing
        # joint_angles = np.array(env.agent.get_joint_positions())
        # # print("joint_angles: ", joint_angles)
        action = net.forward(obs)
        robot_pos = env.get_robot_position()
        # print("action: ", action)
        # print("robot_pos: ", robot_pos)

        # obs, r, done, _ = env.step(action)
        r, obs = env.step(action)
        # r_tot += r

        counter += 1
        if counter > episode_length:
            done = True
    if reward_function == 'abs_y_distance':
        r_tot = robot_pos[1]
    # env.stop_simulation()
    env.shutdown()
    # print('---shutdown simulation---')
    return r_tot