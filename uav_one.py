import logging
import numpy as np

logger = logging.getLogger(__name__)
S = 40
N = 1
M = np.zeros([S,S],dtype=int)-1
M[24,15] = 10


class PathEnv1():
    def __init__(self, render : bool = False):
        self._render = render
        # 定义动作空间
        self.action = np.zeros([N,5],dtype=int)

        # 定义状态空间
        self.state = np.zeros([N,2],dtype=int)

        # 定义回报
        self.r = 0

        # 计数器
        self.step_num = 0
    
    def __apply_action(self, action):
        transMatrix = np.array([[0,1],[0,-1],[-1,0],[1,0],[0,0]])
        self.state = self.state + np.dot(action,transMatrix)
        for s in self.state:
            if min(s[0],s[1])<0 or max(s[0],s[1])>S-1:
                s[0] = np.clip(s[0],0,S-1)
                s[1] = np.clip(s[1],0,S-1)
                self.r += -10
    

    def reset(self, initState):
        self.state = initState
        self.r = 0
        self.step_num = 0
        return self.state

    
    def reward(self):
        r = 0
        for s in self.state:
            r += M[s[0],s[1]]
        self.r = r

    def judge(self):
        if self.step_num < 2000:
            if self.j1(np.array([24,15]),self.state):
                print("success")
                return True
            return False
        return True

    def j1(self, goal, state):
        state = state - goal
        for s in state:
            if s[0] == 0 and s[1] == 0:
                return True
    
    def step(self, action):
        self.__apply_action(action)
        self.step_num += 1
        state = self.state
        self.reward()
        if self.judge():
            done = True
        else:
            done = False
        info = {}
        return state, self.r, done, info
    
    def seed(self, seed=None):
        pass

    def render(self, mode='human'):
        pass