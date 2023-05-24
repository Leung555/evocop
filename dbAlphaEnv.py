"""
An example of how one might use PyRep to create their RL environments.
In this case, the Franka Panda must reach a randomly placed target.
This script contains examples of:
    - RL environment example.
    - Scene manipulation.
    - Environment resets.
    - Setting joint properties (control loop disabled, motor locked at 0 vel)

This code is modified/based on "example_reinforcement_learning_env.ttt" 
"""
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.legged_robots.dbAlpha import dbAlpha
from pyrep.objects.shape import Shape
import numpy as np
import time
import yaml
from  utils.imu import *


# open config files for reading prams
with open('config.yml', 'r') as file:
    configs = yaml.safe_load(file)
class dbAlphaEnv(object):

    def __init__(self, scenefile):
        self.pr = PyRep()
        SCENE_FILE = join(dirname(abspath(__file__)),
                './scenes/'+scenefile+'.ttt')        # print(self.scene_file)
        self.pr.launch(SCENE_FILE, headless=configs['ENV']['HEADLESS'])
        # print("launch sim")
        self.pr.start()
        self.agent = dbAlpha()
        self.orientation = self.get_robot_euler()
        # self.agent.set_control_loop_enabled(False)
        # self.agent.set_motor_locked_at_zero_velocity(True)
        # self.target = Shape('target')
        # self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_leg_joint_positions()

    def _get_state(self):
        # Return state containing arm joint angles position
        return np.concatenate([self.agent.get_leg_joint_positions(),
                               self.agent.get_leg_joint_forces(),
                               self.get_robot_euler()])
    
        # This code includes joint velocities
        # return np.concatenate([self.agent.get_joint_positions(),
        #                        self.agent.get_joint_velocities()])

    def get_robot_position(self):
        return self.agent.get_position()
    
    def get_robot_euler(self):
        # get quaternion of the robot (q)
        # Then tranfrom quaternion-->euler and return euler orientation  
        q = self.agent.get_quaternion()
        robot_euler = euler_from_quaternion(q[0], q[1], q[2], q[3])
        return robot_euler

    def reset(self):
        # Get a random position within a cuboid and set the target position
        # pos = list(np.random.uniform(POS_MIN, POS_MAX))
        # self.target.set_position(pos)
        self.agent.set_leg_joint_positions(self.initial_joint_positions)
        return self._get_state()

    def step(self, action):
        self.agent.set_leg_joint_target_positions(action)  # Execute action on arm
        self.pr.step()  # Step the physics simulation
        # ax, ay, az = self.agent_ee_tip.get_position()
        # tx, ty, tz = self.target.get_position()
        # Reward is negative distance to target
        # reward = -np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        reward = 5
        return reward, self._get_state()
        # return reward, self._get_state()

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def start(self):
        self.pr.start()

    def stop(self):
        self.pr.stop()

class Agent(object):

    def act(self, state):
        del state
        return list(np.random.uniform(-1.0, 1.0, size=(7,)))

    def learn(self, replay_buffer):
        del replay_buffer
        pass

# env = dbAlphaEnv()
# agent = Agent()
# replay_buffer = []

# # Do some stuff
# for e in range(EPISODES):

#     print('Starting episode %d' % e)
#     state = env.reset()
#     for i in range(EPISODE_LENGTH):
#         print("episode: {}", e)
#         print("step:    {}", i)
#         action = agent.act(state)
#         reward, next_state = env.step(action)
#         # replay_buffer.append((state, action, reward, next_state))
#         # state = next_state
#         # agent.learn(replay_buffer)

# print('Done!')
# env.shutdown()
