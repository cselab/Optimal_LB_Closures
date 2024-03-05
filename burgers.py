import numpy as np
import matplotlib.pyplot as plt
import lib.environments.burgers

ep_len = 500
env = lib.environments.burgers.BurgersEnvironment(ep_len=ep_len,
                                                  train=True,
                                                  num_points_cgs=50,
                                                  subsample=3)
action_space = env.action_space
seed = 123456
observation, info = env.reset(seed)
action = np.zeros(action_space.shape, action_space.dtype)
i = 0
while True:
    for name, field in ("u", env.u_fgs), ("v", env.v_fgs):
        plt.imshow(field, origin="upper")
        plt.savefig("%s.%06d.png" % (name, i))
        plt.close()
        nx, ny = np.shape(field)
        plt.plot(field[:, ny // 2], '-')
        plt.plot(field[ny // 2, :], '-')
        plt.axis([None, None, -env.max_velocity, env.max_velocity])
        plt.savefig("%s.slice.%06d.png" % (name, i))
        plt.close()
    i += 1
    if i == ep_len:
        break
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
