import numpy as np
from gym import spaces
# from gym.envs.classic_control import rendering


class Agent:
    def __init__(self, pos=None):
        if pos is not None:
            self.pos = pos
        else:
            self.pos = np.random.random(size=(2))
        self.vel = np.array([0., 0.])
        self.target = np.random.randint(2)

    def update(self, a, dt):
        self.vel = a
        self.pos += self.vel*dt

class ContinuousCrossingEnv():
    def __init__(self, n=3):
        self.dt = 0.05
        self.n_closest = 2
        self.max_o = np.ones(3 + 3*self.n_closest)
        self.max_a = np.ones(2)
        self.n = n
        self.i_step = 0
        self.max_step = 200
        self.viewer = None
        self.action_space = [spaces.Box(low=-self.max_a, high=self.max_a, shape =self.max_a.shape) for i in range(n)]
        self.observation_space = [spaces.Box(low=self.max_o*0, high=self.max_o, shape=self.max_o.shape) for i in range(n)]
        self.agents = [Agent() for i in range(self.n)]
        self.reward_scaling = 10.

    def step(self, a: list):
        self.i_step += 1
        rewards = [0. for i in range(self.n)]
        
        for i, agent in enumerate(self.agents):
            agent.update(a[i], self.dt)
            # Colissions with wall
            if agent.pos[0] > self.max_o[0] or agent.pos[0]<0.:
                rewards[i] -= 0.1 * self.reward_scaling
                agent.pos[0] = max(min(agent.pos[0], self.max_o[0]), 0.)

            if agent.pos[1] > self.max_o[1] or agent.pos[1]<0.:
                rewards[i] -= .1 * self.reward_scaling
                agent.pos[1] = max(min(agent.pos[1], self.max_o[1]), 0.)

            # Reaching target
            if np.linalg.norm(agent.pos[0] - agent.target) < 0.25:
                rewards[i] += 1.0 * self.reward_scaling
                agent.target = (agent.target + 1.0) % 2.0

            # Collissions
            for agent2 in self.agents:
                if agent2 is not agent and np.linalg.norm(agent2.pos - agent.pos)<0.20:
                    rewards[i] -= 0.25 * self.reward_scaling

        return self._get_obs(), rewards, [self.i_step > self.max_step], None

    def reset(self): 
        self.agents = [Agent() for i in range(self.n)]#pos=np.array([float((i+1)%2), float(i)/(self.n)])) for i in range(self.n)]
        self.i_step = 0
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for agent in self.agents:
            sorted_agents = sorted(self.agents, key=lambda x: np.linalg.norm(agent.pos - x.pos))
            obs_i = []
            # Own agent position and target
            obs_i += agent.pos[0], agent.pos[1], agent.target
            for i in range(self.n_closest):
                obs_i += sorted_agents[i+1].pos[0] - sorted_agents[0].pos[0], sorted_agents[i+1].pos[1] - sorted_agents[0].pos[1], sorted_agents[i+1].target
            # obs_i += target.pos[0], target.pos[1]
            obs.append(obs_i)
        return obs

    def render(self):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(0.,1.,0.,1.)
            self.transforms = []
            col = 0
            for agent in self.agents:
                circle = rendering.make_circle(0.10)
                circle.set_color(0,col+0,0)
                self.viewer.add_geom(circle)
                transform = rendering.Transform()
                self.transforms.append(transform)
                circle.add_attr(transform)
                col+=0.5
                col = col%1.0
            for x in [0.25, 1-0.25 ]:
                line = rendering.make_polyline([[x, 0.], [x, 1.0]])
                self.viewer.add_geom(line)

        for agent, transform in zip(self.agents, self.transforms):
            transform.set_translation(agent.pos[0], agent.pos[1])

        return self.viewer.render()

    def close(self):
        pass