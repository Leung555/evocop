import numpy as np
import torch
from collections import deque
import math


class CPGNet:
    def __init__(self):
        
        self.jointNum = 12
        
        # CPG
        MI = 0.05
        self.w11, self.w22 = 1.4, 1.4
        self.w12 =  0.18 + MI
        self.w21 = -0.18 - MI
        
        self.o1 = 0.01
        self.o2 = 0.01
                
        # robot constant
        # Walking
        #self.h_b = 0.3
        #self.t_b = 0.5
        #self.k_b = -1
        
        #self.h_g = 1.0
        #self.t_g = 0.3
        #self.k_g = 0.1
        
        # bias
        # hip joint
        self.h_b = 0.3
        # thigh joint
        self.Ft_b = 0.5
        self.Rt_b = 0.5
        # knee joint
        self.Fk_b = -1
        self.Rk_b = -1
        
        # gain
        # hip joint
        self.h_g = 1.0
        # thigh joint
        self.Ft_g = 0.3
        self.Rt_g = 0.3
        # knee joint
        self.Fk_g = 0.1
        self.Rk_g = 0.1  
        self.params = [self.h_b, self.Ft_b, self.Rt_b, self.Fk_b, self.Rk_b, 
                       self.h_g, self.Ft_g, self.Rt_g, self.Fk_g, self.Rk_g]
    
    
        cpg_delay = 100
        self.d1 = deque([0]*cpg_delay)
        self.d2 = deque([0]*cpg_delay)
        
        for i in range(100):
            self.o1 = math.tanh(self.w11*self.o1 + self.w12*self.o2)
            self.o2 = math.tanh(self.w22*self.o2 + self.w21*self.o1)
            self.d1.append(self.o1)
            o1_d = self.d1.popleft()

            self.d2.append(self.o2)
            o2_d = self.d2.popleft()
            
        # parameters

    def forward(self, pre):

        # put your actuation code here
        self.o1 = math.tanh(self.w11*self.o1 + self.w12*self.o2)
        self.o2 = math.tanh(self.w22*self.o2 + self.w21*self.o1)
        
        o1 = self.o1
        o2 = self.o2
        
        self.d1.append(o1)
        o1_d = self.d1.popleft()

        self.d2.append(o2)
        o2_d = self.d2.popleft()
        #print(self.d)
        #print(d1)
                    
        #Joint Command
        jointPosTarget = [0]*self.jointNum

        for i in range(0, 12, 3):
            #jointPosTarget[i] = self.h_b
            if i == 0 : # FL thigh
                jointPosTarget[i] = o1*self.Ft_g+self.h_b
            if i == 9: # RL thigh
                jointPosTarget[i] = o1*self.Rt_g+self.h_b
            if i == 3: # RR thigh
                jointPosTarget[i] = -o1_d*self.Rt_g+self.h_b
            if i == 6: # FR thigh
                jointPosTarget[i] = -o1_d*self.Ft_g+self.h_b
            
        # command to thight joint
        for i in range(1, 12, 3):
            if i == 1 : # FL thigh
                jointPosTarget[i] = o1*self.Ft_g+self.Ft_b
            if i == 10: # RL thigh
                jointPosTarget[i] = o1*self.Rt_g+self.Rt_b
            if i == 4: # RR thigh
                jointPosTarget[i] = o1_d*self.Rt_g+self.Rt_b
            if i == 7: # FR thigh
                jointPosTarget[i] = o1_d*self.Ft_g+self.Ft_b

        # command to knee joint
        for i in range(2, 12, 3):
            if i == 2: # FL knee 
                jointPosTarget[i] = o2*self.Fk_g+self.Fk_b
            if i == 11: # RL knee 
                jointPosTarget[i] = o2*(self.Rk_g)+self.Rk_b
            if i == 5 : # RR knee 
                jointPosTarget[i] = o2_d*(self.Rk_g)+self.Rk_b
            if i == 8: # FR knee 
                jointPosTarget[i] = o2_d*(self.Fk_g)+self.Fk_b

        return jointPosTarget

    def get_params(self):
        return self.params


    def set_params(self, flat_params):
        self.params = flat_params
        self.h_b, self.Ft_b, self.Rt_b, self.Fk_b, self.Rk_b = self.params[:5]
        self.h_g, self.Ft_g, self.Rt_g, self.Fk_g, self.Rk_g = self.params[5:]

    def get_weights(self):
        return [w for w in self.params]
