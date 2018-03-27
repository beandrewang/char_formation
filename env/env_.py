import numpy as np
import collections


class environment(object):
    Loc = collections.namedtuple('Loc', 'x y')
    def __init__(self, width, height, targets_loc):
        self._width = width
        self._height = height
        self._action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self._action_space)
        self.n_freedom = len(targets_loc)
        self.FDs = []
        self._targets = targets_loc
        self.middle = environment.Loc(x = width / 2 - 1, y = height / 2 - 1)
        self._build_field()

    def _build_field(self):
        # create fds
        for n in range(self.n_freedom):
            x = np.random.randint(0, self._width)
            y = np.random.randint(0, self._height)
            loc = environment.Loc(x = x, y = y)
            self.FDs.append(loc)

    def update_env(self, targets_loc):
        self.FDs = []
        self._targets = targets_loc
        self.n_freedom = len(targets_loc)

        # create fds
        for n in range(self.n_freedom):
            x = np.random.randint(0, self._width)
            y = np.random.randint(0, self._height)
            loc = environment.Loc(x = x, y = y)
            
            self.FDs.append(loc)


    def reset_freedom(self, n):
        obsv = np.array([])
   
        x = np.random.randint(0, self._width)
        y = np.random.randint(0, self._height)
        loc = environment.Loc(x = x, y = y)
        
        self.FDs[n] = loc
        obsv = self.obsv(n)

        return obsv

    def reward(self, loc):
        r, c = [loc.y, loc.x]
        r = 100 - np.sqrt((c - self.middle.x) ** 2 + (r - self.middle.y) ** 2)
        return int(r)

    def distance(self, coords):
        r, c = [loc.y, loc.x]
        d = np.sqrt((c - self.middle.x) ** 2 + (r - self.middle.y) ** 2)
        return d

    def obsv(self, n):
        obsv = np.zeros([1, self._height, self._width, 3])

        loc = self.FDs[n]
        r, c = [loc.y, loc.x]
        obsv[0, r, c, 2] = 1

        for fd in self.FDs:
            loc = fd
            r, c = [loc.y, loc.x]
            obsv[0, r, c, 1] += 1

        for target in self._targets:
            loc = target
            r, c = [loc.y, loc.x]
            obsv[0, r, c, 0] += 1

        #obsv = obsv.reshape(obsv.shape[0], self._height, self._width, 4)
        return obsv - np.mean(obsv)

    def render(self):
        pass

    def step_freedom(self, n, action):
        fd = self.FDs[n]
        r, c = [fd.y, fd.x]
        reward = 0
        base_action = np.array([0, 0])
        if action == 0:   # up
            if r > 0:
                base_action[1] -= 1
            else:
                reward -= 100
        elif action == 1:   # down
            if r < (self._height - 1):
                base_action[1] += 1
            else:
                reward -= 100
        elif action == 2:   # right
            if c < (self._width - 1):
                base_action[0] += 1
            else:
                reward -= 100
        elif action == 3:   # left
            if c > 0:
                base_action[0] -= 1
            else:
                reward -= 100
        elif action == 4: # wait
            reward -= 1
        else:
            reward -= 100

        loc = environment.Loc(x = c + base_action[0], y = r + base_action[1])
        #print action, self.FDs[n], loc
        self.FDs[n] = loc
        fd = self.FDs[n]
        
        done = False
        collision = False
        overlap = 0
        if loc in self._targets:
            for agent in self.FDs:
                if agent == fd:
                  overlap += 1
                  if overlap > 1:
                    collision = True
                    break
            if collision == False:
                #reward += self.reward(loc)
                reward += 100
                '''
                if s in [self.canvas.coords(target) for target in self._targets]:
                    reward -= 10
                '''
                done = True
            else:
                reward -= 100
        else:
            for agent in self.FDs:
                if agent == fd:
                  overlap += 1
                  if overlap > 1:
                    collision = True
                    break
            if collision == False:
                #reward -= self.distance(next_coords) - 1
                reward -= 1
            else:
                reward -= 100
        obsv_ = self.obsv(n)
        return obsv_, reward, done, base_action

import Tkinter as tk
class env_ui(tk.Tk, environment):
    def __init__(self, width, height, targets_loc):
        self._unit = 40
        self._targets_ui = []
        self._agents_ui = []
        self.Texts = []
        tk.Tk.__init__(self)
        environment.__init__(self, width, height, targets_loc)
        self.title('CopterFieldSimulator')
        self.geometry('{0}x{1}'.format(self._width * self._unit, self._height * self._unit))

    def _build_field(self):
        super(env_ui, self)._build_field()
        self.canvas = tk.Canvas(self, bg = 'white', height = self._height * self._unit, width = self._width * self._unit)
        
        '''
        # create the grid
        for c in range(0, self._width * self._unit, self._unit):
            x0, y0, x1, y1 = c, 0, c, self._height * self._unit
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self._height * self._unit, self._unit):
            x0, y0, x1, y1 = 0, r, self._width * self._unit, r
            self.canvas.create_line(x0, y0, x1, y1)
        '''

        # create targets
        for loc in self._targets:
            x, y = loc.x * self._unit + 20, loc.y * self._unit + 20
            target = self.canvas.create_rectangle(x - 15, y - 15, x  + 15, y + 15, fill = 'black')
            self._targets_ui.append(target)

        for n, loc in enumerate(self.FDs):
            x, y = loc.x * self._unit + 20, loc.y * self._unit + 20
            color = "#%06x" % np.random.randint(0x100000, 0xEFFFFF)
            agent = self.canvas.create_rectangle(x - 15, y - 15, x  + 15, y + 15, fill = color)
            text = self.canvas.create_text(x, y, font=("Purisa", 12), text = str(n))
            self._agents_ui.append(agent)
            self.Texts.append(text)

        self.canvas.pack()
        

    def reset_freedom(self, n):
        obsv = super(env_ui, self).reset_freedom(n)

        self.update()

        self.canvas.delete(self._agents_ui[n])
        self.canvas.delete(self.Texts[n])
        loc = self.FDs[n]
        x, y = loc.x * self._unit + 20, loc.y * self._unit + 20
        color = "#%06x" % np.random.randint(0x100000, 0xEFFFFF)
        agent = self.canvas.create_oval(x - 15, y - 15, x + 15, y + 15, fill = color)
        text = self.canvas.create_text(x, y, font=("Purisa", 12), text = str(n))
        self._agents_ui[n] = agent
        self.Texts[n] = text
        
        return obsv

    def update_env(self, targets_loc):
        super(env_ui, self).update_env(targets_loc)
        
        for n in range(len(self.Texts)):
            self.canvas.delete(self._agents_ui[n])
            self.canvas.delete(self.Texts[n])
            self.canvas.delete(self._targets_ui[n])

        self._agents_ui = []
        self.Texts = []
        self._targets_ui = []

        # create targets
        for loc in self._targets:
            x, y = loc.x * self._unit + 20, loc.y * self._unit + 20
            target = self.canvas.create_rectangle(x - 15, y - 15, x  + 15, y + 15, fill = 'black')
            self._targets_ui.append(target)

        for n, loc in enumerate(self.FDs):
            x, y = loc.x * self._unit + 20, loc.y * self._unit + 20
            color = "#%06x" % np.random.randint(0x100000, 0xEFFFFF)
            agent = self.canvas.create_rectangle(x - 15, y - 15, x  + 15, y + 15, fill = color)
            text = self.canvas.create_text(x, y, font=("Purisa", 12), text = str(n))
            self._agents_ui.append(agent)
            self.Texts.append(text)
        

    def step_freedom(self, n, action):
        obsv_, reward, done, base_action = super(env_ui, self).step_freedom(n, action)
        
        base_action *= self._unit
        self.canvas.move(self._agents_ui[n], base_action[0], base_action[1])
        self.canvas.move(self.Texts[n], base_action[0], base_action[1])
        
        return obsv_, reward, done

    def render(self):
        self.update()