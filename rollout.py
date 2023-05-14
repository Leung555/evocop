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
    tilt_penalty = 0
    tilt_lim_rad = 0.35 # (-20,+20) degrees

    while not done:
        print("step: ", counter)
        if counter+1 > episode_length:
            done = True 
            break       

        # Sensing
        # joint_angles = np.array(env.agent.get_joint_positions())
        # # print("joint_angles: ", joint_angles)
        action = net.forward(obs)
        robot_position = env.get_robot_position()
        robot_euler = env.get_robot_euler()
        roll    = robot_euler[0]
        pitch   = robot_euler[1]
        yaw     = robot_euler[2]      
        # print("action: ", action)
        # print("robot_pos: ", robot_pos)

        # obs, r, done, _ = env.step(action)
        r, obs = env.step(action)
        # r_tot += r

        if abs(roll) > tilt_lim_rad or abs(pitch) > tilt_lim_rad:
            tilt_penalty -= 0.1

        counter += 1
    r_tot += robot_position[0]+tilt_penalty # y axis distance
    
    # env.stop_simulation()
    env.shutdown()
    # print('---shutdown simulation---')
    return r_tot