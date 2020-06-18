import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering


class Agent:
    def __init__(self, pos=None):
        if pos is not None:
            self.pos = pos
        else:
            self.pos = np.random.random(size=(2))
        self.vel = np.array([0., 0.])

    def update(self, a, dt):
        self.vel = a
        self.pos += self.vel*dt

class Target:
    def __init__(self):
        self.pos = np.random.random(size=(2))

    def reset(self):
        self.pos = np.random.random(size=(2))


class CrossingEnv():
    def __init__(self, n=6):
        self.dt = 0.05
        self.n_closest = 2
        self.max_o = np.ones(2 + 2*self.n_closest)
        self.max_a = np.ones(2)
        self.n = n
        self.i_step = 0
        self.max_step = 50
        self.viewer = None
        self.action_space = [spaces.Box(low=-self.max_a, high=self.max_a, shape = (2,)) for i in range(n)]
        self.observation_space = [spaces.Box(low=self.max_o*0, high=self.max_o, shape=(2 + 2*self.n_closest,)) for i in range(n)]       
        self.agents = [Agent(pos=np.array([float((i+1)%2), float(i)/(self.n)])) for i in range(self.n)]
        self.targets = [i % 2 for i in range(n)]

    def step(self, a: list):
        self.i_step += 1
        rewards = [0. for i in range(self.n)]
        
        for i, agent in enumerate(self.agents):
            agent.update(a[i], self.dt)
            # rewards[i] += np.linalg.norm(agent.vel)
            if agent.pos[0] > self.max_o[0] or agent.pos[0]<0.:
                agent.vel[0] *= -1
                rewards[i] -= .0
                agent.pos[0] = max(min(agent.pos[0], self.max_o[0]), 0.)

            if agent.pos[1] > self.max_o[1] or agent.pos[1]<0.:
                agent.vel[1] *= -1
                rewards[i] -= .0
                agent.pos[1] = max(min(agent.pos[1], self.max_o[1]), 0.)
            # Targets
            rewards[i] += -0.5 * np.linalg.norm(agent.pos[0] - self.targets[i])
            # Collissions
            for agent2 in self.agents:
                if agent2 is not agent and np.linalg.norm(agent2.pos - agent.pos)<0.20:
                    rewards[i] -= 1.0
                    # agent.pos -= self.dt * agent.vel
        return self._get_obs(), rewards, [self.i_step > self.max_step], None

    def reset(self): 
        self.agents = [Agent(pos=np.array([float((i+1)%2), float(i)/(self.n)])) for i in range(self.n)]
        self.targets = [float(i%2) for i in range(self.n)]
        self.i_step = 0
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for agent, target in zip(self.agents, self.targets):
            sorted_agents = sorted(self.agents, key=lambda x: np.linalg.norm(agent.pos - x.pos))
            obs_i = []
            obs_i += agent.pos[0], agent.pos[1]
            for i in range(self.n_closest):
                obs_i += sorted_agents[i+1].pos[0] - sorted_agents[0].pos[0], sorted_agents[i+1].pos[1] - sorted_agents[0].pos[1]
            # obs_i += target.pos[0], target.pos[1]
            obs.append(obs_i)
        return obs

    def render(self):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(0.,1.,0.,1.)
            self.transforms = []
            self.target_transforms = [] 
            col = 0
            for agent, target in zip(self.agents, self.targets):
                circle = rendering.make_circle(0.10)
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
                col+=0.5
                col = col%1.0
        for agent, transform in zip(self.agents, self.transforms):
            transform.set_translation(agent.pos[0], agent.pos[1])
        for target, transform in zip(self.targets, self.target_transforms):
            transform.set_translation(target, 0.5)

        return self.viewer.render()

    def close(self):
        pass


class ContinuousCrossingEnv():
    def __init__(self, n=4):
        self.dt = 0.05
        self.n_closest = 2
        self.max_o = np.ones(1 + 2*self.n_closest)
        self.max_a = np.ones(2)
        self.n = n
        self.i_step = 0
        self.max_step = 150
        self.viewer = None
        self.action_space = [spaces.Box(low=-self.max_a, high=self.max_a, shape = (2,)) for i in range(n)]
        self.observation_space = [spaces.Box(low=self.max_o*0, high=self.max_o, shape=(1 + 2*self.n_closest,)) for i in range(n)]       
        self.agents = [Agent(pos=np.array([float((i+1)%2), float(i)/(self.n)])) for i in range(self.n)]
        self.targets = [i % 2 for i in range(n)]

    def step(self, a: list):
        self.i_step += 1
        rewards = [0. for i in range(self.n)]
        
        for i, agent in enumerate(self.agents):
            agent.update(a[i], self.dt)

            if agent.pos[1] > self.max_o[1] or agent.pos[1]<0.:
                agent.vel[1] *= -1
                rewards[i] -= .0
                agent.pos[1] = max(min(agent.pos[1], self.max_o[1]), 0.)
            # Targets
            rewards[i] += agent.vel[0]*self.dt if i%2==1 else -agent.vel[0]*self.dt 
            # Collissions
            for agent2 in self.agents:
                if agent2 is not agent and np.linalg.norm(agent2.pos - agent.pos)<0.20:
                    rewards[i] -= 1.0
                    # agent.pos -= self.dt * agent.vel
            if agent.pos[0]>1.0:
                agent.pos[0] = 0.
            elif agent.pos[0] < 0.00:
                agent.pos[0] = 1.
        return self._get_obs(), rewards, [self.i_step > self.max_step], None

    def reset(self): 
        self.agents = [Agent(pos=np.array([float((i+1)%2), float(i)/(self.n)])) for i in range(self.n)]
        self.targets = [float(i%2) for i in range(self.n)]
        self.i_step = 0
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for agent, target in zip(self.agents, self.targets):
            sorted_agents = sorted(self.agents, key=lambda x: np.linalg.norm(agent.pos - x.pos))
            obs_i = []
            obs_i.append(agent.pos[1])
            for i in range(self.n_closest):
                difa = sorted_agents[i+1].pos[0] - agent.pos[0]
                difb = sorted_agents[i+1].pos[1] - agent.pos[1]
                if difa < 0.0:
                    difa += 1.0

                obs_i += difa, difb 
            # obs_i += target.pos[0], target.pos[1]
            obs.append(obs_i)
        return obs

    def render(self):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(0.,1.,0.,1.)
            self.transforms = []
            self.target_transforms = [] 
            col = 0
            for agent, target in zip(self.agents, self.targets):
                circle = rendering.make_circle(0.10)
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
                col+=0.5
                col = col%1.0
        for agent, transform in zip(self.agents, self.transforms):
            transform.set_translation(agent.pos[0], agent.pos[1])
        for target, transform in zip(self.targets, self.target_transforms):
            transform.set_translation(target, 0.5)

        return self.viewer.render()

    def close(self):
        pass
