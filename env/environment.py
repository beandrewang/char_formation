import numpy as np
import Tkinter as tk
import collections

'''
class env:
    Loc = collections.namedtuple('Loc', 'x y')
    def __init__(self, width, height, n_freedom, targets_loc):
        self._width = width
        self._height = height
        self.n_freedom = n_freedom
        self._targets_loc = targets_loc

    def reset_freedom(self, n):
'''

class environment(tk.Tk, object):
    Loc = collections.namedtuple('Loc', 'x y')
    def __init__(self, width, height, targets_loc, rendering = True):
        super(environment, self).__init__()
        self._width = width
        self._height = height
        self._unit = 40
        self._action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self._action_space)
        self.n_freedom = len(targets_loc)
        self._targets_loc = targets_loc
        self.FDs = []
        self.Texts = []
        self._targets = []
        self.rendering = rendering
        self.middle = environment.Loc(x = width / 2 - 1, y = height / 2 - 1)
        self.title('CopterFieldSimulator')
        self.geometry('{0}x{1}'.format(self._width * self._unit, self._height * self._unit))
        self._build_field()

    def _build_field(self):
        self.canvas = tk.Canvas(self, bg = 'white', height = self._height * self._unit, width = self._width * self._unit)

        # create the grid
        for c in range(0, self._width * self._unit, self._unit):
            x0, y0, x1, y1 = c, 0, c, self._height * self._unit
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self._height * self._unit, self._unit):
            x0, y0, x1, y1 = 0, r, self._width * self._unit, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create targets
        for loc in self._targets_loc:
            x, y = loc.x * self._unit + 20, loc.y * self._unit + 20
            target = self.canvas.create_rectangle(x - 15, y - 15, x  + 15, y + 15, fill = 'black')
            self._targets.append(target)

        # create fds
        for n in range(self.n_freedom):
            x = np.random.randint(0, self._width)
            y = np.random.randint(0, self._height)
            loc = environment.Loc(x = x, y = y)
            x, y = loc.x * self._unit + 20, loc.y * self._unit + 20
            color = "#%06x" % np.random.randint(0x100000, 0xEFFFFF)
            fd = self.canvas.create_oval(x - 15, y - 15, x + 15, y + 15, fill = color)
            text = self.canvas.create_text(x, y, font=("Purisa", 12), text = str(n))
            self.FDs.append(fd)
            self.Texts.append(text)

        self.canvas.pack()

    def update_env(self, targets_loc):
        for n in range(self.n_freedom):
            self.canvas.delete(self.FDs[n])
            self.canvas.delete(self.Texts[n])
            self.canvas.delete(self._targets[n])

        self.FDs = []
        self.Texts = []
        self._targets = []
        self.n_freedom = len(targets_loc)
        self._targets_loc = targets_loc

        # create targets
        for loc in self._targets_loc:
            x, y = loc.x * self._unit + 20, loc.y * self._unit + 20
            target = self.canvas.create_rectangle(x - 15, y - 15, x  + 15, y + 15, fill = 'black')
            self._targets.append(target)

        # create fds
        for n in range(self.n_freedom):
            x = np.random.randint(0, self._width)
            y = np.random.randint(0, self._height)
            loc = environment.Loc(x = x, y = y)
            x, y = loc.x * self._unit + 20, loc.y * self._unit + 20
            color = "#%06x" % np.random.randint(0x100000, 0xEFFFFF)
            fd = self.canvas.create_oval(x - 15, y - 15, x + 15, y + 15, fill = color)
            text = self.canvas.create_text(x, y, font=("Purisa", 12), text = str(n))
            self.FDs.append(fd)
            self.Texts.append(text)

    def coords2unit(self, coords):
        x = int(coords[0] / self._unit)
        y = int(coords[1] / self._unit)
        return int(y), int(x)


    def reset_freedom(self, n):
        if self.rendering:
            self.update()
        obsv = np.array([])
        self.canvas.delete(self.FDs[n])
        self.canvas.delete(self.Texts[n])
        x = np.random.randint(0, self._width)
        y = np.random.randint(0, self._height)
        loc = environment.Loc(x = x, y = y)
        x, y = loc.x * self._unit + 20, loc.y * self._unit + 20
        color = "#%06x" % np.random.randint(0x100000, 0xEFFFFF)
        fd = self.canvas.create_oval(x - 15, y - 15, x + 15, y + 15, fill = color)
        text = self.canvas.create_text(x, y, font=("Purisa", 12), text = str(n))
        self.FDs[n] = fd
        self.Texts[n] = text
        obsv = self.obsv(n)

        return obsv

    def render(self):
        if self.rendering:
            self.update()

    def reward(self, coords):
        r, c = self.coords2unit(coords)
        r = 100 - np.sqrt((c - self.middle.x) ** 2 + (r - self.middle.y) ** 2)
        return int(r)

    def distance(self, coords):
        r, c = self.coords2unit(coords)
        d = np.sqrt((c - self.middle.x) ** 2 + (r - self.middle.y) ** 2)
        return d

    def obsv(self, n):
        obsv = np.zeros([1, self._height, self._width, 3])

        fd = self.FDs[n]
        coords = self.canvas.coords(fd)
        r, c = self.coords2unit(coords)
        obsv[0, r, c, 2] = 1

        for fd in self.FDs:
            coords = self.canvas.coords(fd)
            r, c = self.coords2unit(coords)
            obsv[0, r, c, 1] += 1

        for target in self._targets:
            coords = self.canvas.coords(target)
            r, c = self.coords2unit(coords)
            obsv[0, r, c, 0] += 1

        #obsv = obsv.reshape(obsv.shape[0], self._height, self._width, 4)
        return obsv - np.mean(obsv)

    def step_freedom(self, n, action):
        fd = self.FDs[n]
        s = self.canvas.coords(fd)
        reward = 0
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > self._unit:
                base_action[1] -= self._unit
            else:
                reward -= 100
        elif action == 1:   # down
            if s[1] < (self._height - 1) * self._unit:
                base_action[1] += self._unit
            else:
                reward -= 100
        elif action == 2:   # right
            if s[0] < (self._width - 1) * self._unit:
                base_action[0] += self._unit
            else:
                reward -= 100
        elif action == 3:   # left
            if s[0] > self._unit:
                base_action[0] -= self._unit
            else:
                reward -= 100
        elif action == 4: # wait
            reward -= 1
        else:
            reward -= 100

        #print reward
        
        self.canvas.move(fd, base_action[0], base_action[1])
        self.canvas.move(self.Texts[n], base_action[0], base_action[1])
        next_coords = self.canvas.coords(fd)  # next state
        done = False
        collision = False
        if next_coords in [self.canvas.coords(target) for target in self._targets]:
            for agent in self.FDs:
                if agent != fd:
                    if next_coords == self.canvas.coords(agent):
                        collision = True
                        break
            if collision == False:
                reward += self.reward(next_coords)
                #reward += 100
                '''
                if s in [self.canvas.coords(target) for target in self._targets]:
                    reward -= 10
                '''
                done = True
            else:
                reward -= 100
        else:
            for agent in self.FDs:
                if agent != fd:
                    if next_coords == self.canvas.coords(agent):
                        collision = True
                        break
            if collision == False:
                #reward -= self.distance(next_coords) - 1
                reward -= 1
            else:
                reward -= 100

        obsv_ = self.obsv(n)

        return obsv_, reward, done


def update():
    for t in range(10):
        s = env.reset()

        while True:
            env.render()
            a = [1, 2, 3]
            s, r, done = env.step(a)
            print s
            if done:
                break

if __name__ == '__main__':
    nFDs = 3
    n_targets = 3
    width = 9
    height = 9
    fds_origin_loc = []
    targets_loc = (environment.Loc(x = 3, y = 3),
                   environment.Loc(x = 4, y = 4),
                   environment.Loc(x = 5, y = 5))
    env = environment(8, 8, 40, nFDs, targets_loc)
    #env.after(100, update)
    print env.obsv(0)
    env.mainloop()
