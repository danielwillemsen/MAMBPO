import numpy as np
from gym import spaces
try:
    from gym.envs.classic_control import rendering
except:
    pass

class Agent:
    def __init__(self):
        self.pos = np.random.random(size=(2))
        self.vel = np.array([0., 0.])

    def update(self, a, dt):
        self.vel += a*0.5
        self.vel *= 0.9
        self.pos += self.vel*dt

class Target:
    def __init__(self):
        self.pos = np.random.random(size=(2))

    def reset(self):
        self.pos = np.random.random(size=(2))

class WaypointsEnv():
    def __init__(self, n_agents=2):
        self.dt = 0.1
        self.max_a = np.ones(2)
        self.n = n_agents
        self.i_step = 0
        self.max_step = 1000
        self.viewer = None
        self.use_gradient_only = True
        self.n_closest = min(2, self.n-1)
        self.max_o = np.ones(4*self.n_closest + 6)

        self.action_space = [spaces.Box(low=-self.max_a, high=self.max_a, shape = (2,)) for i in range(n_agents)]
        self.observation_space = [spaces.Box(low=self.max_o*0, high=self.max_o, shape=(4*self.n_closest+6,)) for i in range(n_agents)]
        self.agents = [Agent() for agent in range(n_agents)]
        self.targets = [Target() for target in range(n_agents)]

    def step(self, a: list):
        self.i_step += 1
        rewards = [0. for i in range(self.n)]
        
        for i, agent in enumerate(self.agents):
            agent.update(a[i], self.dt)
            # rewards[i] += np.linalg.norm(agent.vel)
            if agent.pos[0] > self.max_o[0] or agent.pos[0]<0.:
                agent.vel[0] *= 0.
                rewards[i] -= .0
                agent.pos[0] = max(min(agent.pos[0], self.max_o[0]), 0.)

            if agent.pos[1] > self.max_o[1] or agent.pos[1]<0.:
                agent.vel[1] *= 0.
                rewards[i] -= .0
                agent.pos[1] = max(min(agent.pos[1], self.max_o[1]), 0.)
            # Targets
            if not self.use_gradient_only:
                if np.linalg.norm(agent.pos - self.targets[i].pos) < -0.25:
                    rewards[i] += 1.0
                    self.targets[i].reset()
                else:
                    rewards[i] += -0.2 * np.linalg.norm(agent.pos - self.targets[i].pos)
            else:
                rewards[i] += -1.0 * np.linalg.norm(agent.pos - self.targets[i].pos)

            # Collissions
            for agent2 in self.agents:
                if agent2 is not agent and np.linalg.norm(agent2.pos - agent.pos)<0.10:
                    rewards[i] -= 0.0
                    # print("Collision!")
            if self.i_step % 50 == 0:
                for target in self.targets:
                    target.reset()
        return self._get_obs(), rewards, [self.i_step > self.max_step], None

    def reset(self): 
        self.agents = [Agent() for agent in range(self.n)]
        self.targets = [Target() for target in range(self.n)]
        self.i_step = 0
        return self._get_obs()

    # def _get_obs(self):
    #     obs = []
    #     for agent, target in zip(self.agents, self.targets):
    #         obs_i = []
    #         # Own position, velocity, target
    #         obs_i += agent.pos[0], agent.pos[1]
    #         obs_i += agent.vel[0], agent.vel[1]
    #         obs_i += target.pos[0], target.pos[1]
    #
    #         for agent2 in self.agents:
    #             if agent2 is not agent:
    #                 obs_i += agent2.pos[0], agent2.pos[1]
    #         obs.append(obs_i)
    #     return obs

    def _get_obs(self):
        obs = []
        for agent, target in zip(self.agents, self.targets):
            sorted_agents = sorted(self.agents, key=lambda x: np.linalg.norm(agent.pos - x.pos))
            obs_i = []
            obs_i += agent.pos[0], agent.pos[1]
            obs_i += agent.vel[0], agent.vel[1]
            obs_i += target.pos[0], target.pos[1]

            for i in range(self.n_closest):
                obs_i += sorted_agents[i+1].pos[0] - sorted_agents[0].pos[0], sorted_agents[i+1].pos[1] - sorted_agents[0].pos[1]
                obs_i += sorted_agents[i+1].vel[0] - sorted_agents[0].vel[0], sorted_agents[i+1].vel[1] - sorted_agents[0].vel[1]
            # obs_i += target.pos[0], target.pos[1]
            obs.append(obs_i)
        return obs

    def seed(self, **kwargs):
        pass
    try:
        def render(self):
            if self.viewer is None:
                self.viewer = rendering.Viewer(500, 500)
                self.viewer.set_bounds(0.,1.,0.,1.)
                self.transforms = []
                self.target_transforms = []
                col = 0
                for agent, target in zip(self.agents, self.targets):
                    circle = rendering.make_circle(0.05)
                    circle.set_color(0,col+0,0)
                    self.viewer.add_geom(circle)
                    transform = rendering.Transform()
                    self.transforms.append(transform)
                    circle.add_attr(transform)
                    circle = rendering.make_circle(0.15)
                    circle.set_color(0.5,col+0.5,0.5)
                    self.viewer.add_geom(circle)
                    transform = rendering.Transform()
                    self.target_transforms.append(transform)
                    circle.add_attr(transform)
                    col+=0.3
            for target, transform in zip(self.targets, self.target_transforms):
                transform.set_translation(target.pos[0], target.pos[1])

            for agent, transform in zip(self.agents, self.transforms):
                transform.set_translation(agent.pos[0], agent.pos[1])

            return self.viewer.render()
    except:
        pass

    def close(self):
        pass
