import numpy as np
import random


class VelocityFieldGenerator:
    def __init__(self, velocity_field_type, max_velocity, img_size):
        self.velocity_field_type = velocity_field_type
        self.max_velocity = max_velocity
        self.img_size = img_size
        x = np.linspace(0, 1, img_size)
        y = np.linspace(0, 1, img_size)
        self.dx, self.dy = x[1] - x[0], y[1] - y[0]
        self.X, self.Y = np.meshgrid(x, y)

    def generate(self):
        """
        Returns random field with max velocity lower than self.max_velocity
        """
        if self.velocity_field_type == 'vortex':

            # sample random sign
            sign = np.random.choice([-1, 1])

            # sample from uniform [0.5, 1]
            abs_velocity = (np.random.rand(1) + 1) / 2 * self.max_velocity
            c_x = np.sin(np.pi * self.X) ** 2 * np.sin(2 * np.pi * self.Y)
            c_y = - np.sin(np.pi * self.Y) ** 2 * np.sin(2 * np.pi * self.X)
            return np.float32(sign * abs_velocity * c_x), np.float32(sign * abs_velocity * c_y)

        elif self.velocity_field_type == "train":
            # create linear combination of Taylor-Green vortices
            num_ks = random.randint(1, 4)
            ks = random.sample(list(range(1, 6)), num_ks)
            # translation component
            u, v = 2. * random.random() - 1., 2 * random.random() - 1.
            for k in ks:
                sign = np.random.choice([-1, 1])
                u += sign * np.sin(np.pi * k * self.X) * np.cos(np.pi * k * self.Y)
                v += - sign * np.cos(np.pi * k * self.X) * np.sin(np.pi * k * self.Y)
            return u / (num_ks + 1), v / (num_ks + 1)

        elif self.velocity_field_type == "train2":
            # Same to "train" but with different range of k and no translation component
            # create linear combination of Taylor-Green vortices
            num_ks = random.randint(2, 4)
            ks = random.sample([2, 4, 6, 8], num_ks)  # use only even numbers to ensure periodic boundary conditions

            u, v = 0, 0
            for k in ks:
                sign = np.random.choice([-1, 1])
                u += sign * np.sin(np.pi * k * self.X) * np.cos(np.pi * k * self.Y)
                v += - sign * np.cos(np.pi * k * self.X) * np.sin(np.pi * k * self.Y)
            return u / (num_ks + 1), v / (num_ks + 1)
        else:
            raise NotImplementedError(f'Unknown velocity field type {self.velocity_field_type}')

