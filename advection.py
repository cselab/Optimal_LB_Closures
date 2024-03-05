import numpy as np
import matplotlib.pyplot as plt
import lib.environments.burgers

env = lib.environments.advection.AdvectionEnvironment(
    img_size=64,
    train=False,
    subsample=4,
    ep_len=2000,
    dataset_name="mnist",
    velocity_field_type="train")
seed = 123456
observation, info = env.reset(seed)
action = np.zeros(env.action_space.shape, env.action_space.dtype)
i = 0
while True:
    for name, field in [("gt_state", env.gt_state)]:
        plt.imshow(field)
        plt.savefig("%s.%06d.png" % (name, i))
        plt.close()
        nx, ny = np.shape(field)
        plt.plot(field[:, ny // 2], '-')
        plt.plot(field[ny // 2, :], '-')
        plt.axis([None, None, -0.2, 1.2])
        plt.savefig("%s.slice.%06d.png" % (name, i))
        plt.close()
    i += 1
    observation, reward, terminated, truncated, info = env.step(
        action[None, :])
    if terminated:
        break
