import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering

class Agent:
    def __init__(self):
        self.pos = np.random.random(size=(2))
        self.vel = np.array([0.,0.])
    
    def update(self, a, dt):
        self.vel += a*dt
        self.pos += self.vel*dt

class WaypointsEnv():
    def __init__(self, n=1):
        self.dt = 0.1
        self.max_o = np.array([1.,1.,1.,1.])
        self.max_a = np.array([1.,1.])
        self.n = n
        self.i_step = 0
        self.max_step = 200
        self.viewer = None

        self.action_space = [spaces.Box(low=-self.max_a, high=self.max_a, shape=(2,)) for i in range(n)]
        self.observation_space = [spaces.Box(low=np.array([0.,0.,0.,0.]), high=self.max_o, shape=(4,)) for i in range(n)]       
        self.agents = [Agent() for agent in range(n)]

    def step(self, a: list):
        self.i_step += 1
        rewards = [0. for i in range(self.n)]
        observations = []
        
        for i, agent in enumerate(self.agents):
            agent.update(a[i], self.dt)
            rewards[i] += -np.linalg.norm(agent.pos)
            if agent.pos[0] > self.max_o[0] or agent.pos[0]<0.:
                agent.vel[0] *= -1
                rewards[i] -= 1.0
                agent.pos[0] = max(min(agent.pos[0], self.max_o[0]), 0.)

            if agent.pos[1] > self.max_o[1] or agent.pos[1]<0.:
                agent.vel[1] *= -1
                rewards[i] -= 1.0
                agent.pos[1] = max(min(agent.pos[1], self.max_o[1]), 0.)

        return self._get_obs(), rewards, [self.i_step > self.max_step], None

    def reset(self): 
        self.agents = [Agent() for agent in range(self.n)]
        self.i_step = 0
        return self._get_obs()

    def _get_obs(self):
        return [np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1]]) for agent in self.agents]

    def render(self):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(0.,1.,0.,1.)
            self.transforms = []
            for agent in self.agents:
                circle = rendering.make_circle(0.05)
                circle.set_color(0,0,0)
                self.viewer.add_geom(circle)
                transform = rendering.Transform()
                self.transforms.append(transform)
                circle.add_attr(transform)
        for agent, transform in zip(self.agents, self.transforms):
            transform.set_translation(agent.pos[0], agent.pos[1])
        return self.viewer.render()


    def close(self):
        pass
