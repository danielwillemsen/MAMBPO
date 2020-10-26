import numpy as np
from gym import spaces
from numpy import random
import random

try:
    from gym.envs.classic_control import rendering
except:
    pass


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def limit_pi(angle):
    if angle > np.pi:
        angle = angle - 2*np.pi
    if angle < -np.pi:
        angle = angle + 2*np.pi
    return angle

class Agent:
    def __init__(self, angle, randomized=False):
        if randomized:
            self.pos = (np.random.random(2)-0.5)*2
            self.heading = limit_pi((random.random()-0.5)*2*np.pi)
        else:
            self.pos = np.array(pol2cart(0.5,angle))
            self.heading = limit_pi(angle+np.pi)
        self.vel = 0.

    def update(self, v, rot, dt):
        self.heading = limit_pi(self.heading + dt*rot*10)
        self.vel = v
        self.pos += np.array(pol2cart(v*dt, self.heading))

class Target:
    def __init__(self, angle, randomized=False):
        if randomized:
            self.pos = (np.random.random(2)-0.5)*2
        else:
            self.pos = np.array(pol2cart(0.5,angle))

    def reset(self):
        pass

class CircleEnv():
    def __init__(self, n_agents=2, randomized=False):
        self.runobs = []
        self.dt = 0.05
        self.max_a = np.ones(2)
        self.n = n_agents
        self.i_step = 0
        self.max_step = 100
        self.viewer = None
        self.use_gradient_only = True
        self.randomized = randomized
        self.n_closest = min(2, self.n-1)
        self.max_o = np.ones(3*self.n_closest + 3)*1

        self.action_space = [spaces.Box(low=-self.max_a, high=self.max_a, shape = (2,)) for i in range(n_agents)]
        self.observation_space = [spaces.Box(low=self.max_o*0, high=self.max_o, shape=(3*self.n_closest+3,)) for i in range(n_agents)]
        self.agents = [Agent(angle, randomized=randomized) for angle in np.linspace(0, 2*np.pi, num=self.n, endpoint=False)]
        self.targets = [Target(angle+np.pi, randomized=randomized) for angle in np.linspace(0, 2*np.pi, num=self.n, endpoint=False)]

    def step(self, a: list):
        self.i_step += 1
        rewards = [0. for i in range(self.n)]
        rewards_collision = [0. for i in range(self.n)]
        rewards_target = [0. for i in range(self.n)]

        for i, agent in enumerate(self.agents):
            agent.update(a[i][0]*0.5+0.5, a[i][1], self.dt)
            # rewards[i] += np.linalg.norm(agent.vel)
            if agent.pos[0] > self.max_o[0] or agent.pos[0]<-self.max_o[0]:
                agent.vel *= 0.
                rewards[i] -= .0
                agent.pos[0] = max(min(agent.pos[0], self.max_o[0]), -self.max_o[0])

            if agent.pos[1] > self.max_o[1] or agent.pos[1]<self.max_o[1]:
                agent.vel *= 0.
                rewards[i] -= .0
                agent.pos[1] = max(min(agent.pos[1], self.max_o[1]), -self.max_o[1])
            # Targets
            if not self.use_gradient_only:
                if np.linalg.norm(agent.pos - self.targets[i].pos) < -0.25:
                    rewards[i] += 1.0
                    self.targets[i].reset()
                else:
                    rewards[i] += -0.2 * np.linalg.norm(agent.pos - self.targets[i].pos)
            else:
                if np.linalg.norm(self.targets[i].pos - agent.pos)<0.10:
                    rewards[i] += 1.0
                    rewards_target[i] += 1.0

                else:
                    rewards[i] += 0.1/np.linalg.norm(self.targets[i].pos - agent.pos)
                    rewards_target[i] += 0.1/np.linalg.norm(self.targets[i].pos - agent.pos)

                    # rewards[i] -= np.linalg.norm(self.targets[i].pos - agent.pos)**(1./2.)

            # Collissions
            for agent2 in self.agents:
                if agent2 is not agent:
                    if np.linalg.norm(agent2.pos - agent.pos)<0.1:
                        rewards[i] -= 1.
                        rewards_collision[i] -= 1.

                        # print("Collision!")
                    # elif np.linalg.norm(agent2.pos - agent.pos)<0.4:
                    #     rewards[i] -= 0.5
                    #     rewards_collision[i] -= 0.5
                    # else:
                    #     rewards[i] -= 0.05/np.linalg.norm(agent2.pos - agent.pos)
                    #     rewards_collision[i] -= 0.05/np.linalg.norm(agent2.pos - agent.pos)

                        # print("Close!")
                        #rewards[i] -= 0.1/np.linalg.norm(agent2.pos - agent.pos)
                    # rewards[i] += 0#np.linalg.norm(agent2.pos - agent.pos)**(1./2.)
                    # print("Collision!")
            if self.i_step % 50 == 0:
                for target in self.targets:
                    target.reset()
        return self._get_obs(), rewards, [self.i_step > self.max_step], None, {"rewards_target": rewards_target, "rewards_collision": rewards_collision}

    def reset(self):
        self.agents = [Agent(angle, randomized=self.randomized) for angle in np.linspace(0, 2*np.pi, num=self.n, endpoint=False)]
        self.targets = [Target(angle+np.pi, randomized=self.randomized) for angle in np.linspace(0, 2*np.pi, num=self.n, endpoint=False)]
        self.i_step = 0
        if self.viewer is not None:
            self.runobs = []
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
            #obs_i += agent.pos[0], agent.pos[1]
            obs_i.append(agent.vel)#, agent.heading
            dist, head = cart2pol(*(target.pos - agent.pos))
            head = limit_pi(head - agent.heading)
            obs_i += dist, head
            # obs_i += target.pos[0], target.pos[1]

            for i in range(self.n_closest):
                dist, head = cart2pol(*(sorted_agents[i+1].pos - agent.pos))
                head = limit_pi(head - agent.heading)
                rel_head = limit_pi(sorted_agents[i+1].heading - agent.heading)

                obs_i += dist, head, rel_head
                # obs_i += sorted_agents[i+1].vel - sorted_agents[0].vel, sorted_agents[i+1].vel - sorted_agents[0].vel
            # obs_i += target.pos[0], target.pos[1]
            obs.append(obs_i)
        return obs

    def seed(self, **kwargs):
        pass
    try:
        def render(self):
            colors = [(0., 0., 0.), (0.5,0.0,0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)]
            if self.viewer is None:
                self.viewer = rendering.Viewer(500, 500)
                self.viewer.set_bounds(-1.,1.,-1.,1.)
                self.transforms = []
                self.target_transforms = []
                col = 0
                for agent, target in zip(self.agents, self.targets):
                    circle = rendering.make_circle(0.05)
                    circle.set_color(*colors[col])
                    self.viewer.add_geom(circle)
                    transform = rendering.Transform()
                    self.transforms.append(transform)
                    circle.add_attr(transform)
                    circle = rendering.make_circle(0.15, filled=False)
                    circle.set_color(*colors[col])
                    self.viewer.add_geom(circle)
                    transform = rendering.Transform()
                    self.target_transforms.append(transform)
                    circle.add_attr(transform)
                    col += 1
                    if col == len(colors):
                        col = 0
            for target, transform in zip(self.targets, self.target_transforms):
                transform.set_translation(target.pos[0], target.pos[1])

            col = 0
            for obs in self.runobs:
                self.viewer.add_onetime(obs)
            for agent, transform in zip(self.agents, self.transforms):
                circle = rendering.make_circle(0.005)
                circle.set_color(*colors[col])
                transform2 = rendering.Transform()
                self.runobs.append(circle)
                circle.add_attr(transform2)
                transform2.set_translation(agent.pos[0], agent.pos[1])

                self.viewer.add_onetime(circle)
                transform.set_translation(agent.pos[0], agent.pos[1])
                col += 1
                if col == len(colors):
                    col = 0
            return self.viewer.render()
    except:
        pass

    def close(self):
        pass
