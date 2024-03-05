from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import wandb

from gymnasium import spaces
from tianshou.data import Batch
from torch.utils.data import DataLoader
from scipy.interpolate import griddata
from typing import Tuple

from lib.utils import sample_from_dataloader
from lib.environments.base import BaseEnvironment
from lib.environments.velocity_generation import VelocityFieldGenerator
from lib.datasets import get_train_val_test_initial_conditions_dataset
from lib.models.wrappers import MarlModel


class AdvectionEnvironment(BaseEnvironment, ABC):
    def __init__(self,
                 ep_len: int = 5,
                 train: bool = True,
                 img_size: int = 64,
                 subsample: int = 8,
                 dataset_name: str = 'mnist',
                 velocity_field_type: str = 'translational'):
        super().__init__()
        self.dataset_name = dataset_name
        self.train = train
        self.img_size = img_size

        # Settings for FD simulation
        self.ep_len = ep_len
        self.max_ep_len = 100
        self.mse_threshold = 0.015
        self.dx, self.dy = 1 / img_size, 1 / img_size

        # Setting for DNS FD simulation
        self.subsample = subsample
        self.dns_img_size = img_size * subsample
        self.dns_dx, self.dns_dy = 1 / self.dns_img_size, 1 / self.dns_img_size

        # Needed for running FD simulation
        self.max_velocity = 1.0
        self.c_x, self.c_y = None, None  # get reset randomly for each episode
        self.c_x_dns, self.c_y_dns = None, None  # get reset randomly for each episode
        self.initial_condition = None  # gets set to initial condition in reset (keeps track of analytical solution)
        self.gt_state = None  # label to calculate supervised loss for each step in testing
        self.rl_adjusted_state = None
        self.velocity_field_generator = VelocityFieldGenerator(velocity_field_type,
                                                               self.max_velocity,
                                                               self.dns_img_size)

        #self.dt = 0.9 / ((self.max_velocity / self.dx) + (self.max_velocity / self.dy))  # dt fulfills CFL condition
        self.dt = 0.25 * min(self.dx, self.dy)
        self.T = self.ep_len * self.dt
        self.dns_dt = self.dt / subsample
        self.dns_nt = int(self.T / self.dns_dt)

        # limit action space to [-1, 1] to avoid too large changes
        self.action_space = spaces.Box(low=-1.,
                                       high=1.,
                                       shape=(self.img_size, self.img_size),
                                       dtype=np.float32)

        # data is zero mean and unit variance, choose observation space in range [-3, 3] to account for outliers
        self.observation_space = spaces.Box(low=-3.,
                                            high=3.,
                                            shape=(self.img_size, self.img_size),
                                            dtype=np.float32)

        # Set up to load data for initial conditions of field
        _train_dataset, _, _test_dataset = get_train_val_test_initial_conditions_dataset(dataset_name=self.dataset_name,
                                                                                         img_size=self.dns_img_size)
        self.dataset = _train_dataset if train else _test_dataset
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=1,
                                      shuffle=True)
        self.data_iter = iter(self.data_loader)

    def reset(self, *args, **kwargs):
        # reset tracking
        self.step_count = 0

        # reset state to new image from dataset
        dns_initial_condition, self.data_iter = sample_from_dataloader(self.data_iter, self.data_loader)

        # remove batch dimension and add small constant for numerical stability
        dns_initial_condition = dns_initial_condition[0, 0].numpy() + 1e-6
        self.initial_condition = dns_initial_condition[::self.subsample, ::self.subsample]
        assert self.observation_space.contains(self.initial_condition), \
            f"State is not in observation space, " \
            f"min(state): {np.min(self.initial_condition)}, max(state): {np.max(self.initial_condition)}"

        # Sample new velocity vector
        self.c_x_dns, self.c_y_dns = self.velocity_field_generator.generate()
        self.c_x, self.c_y = self.c_x_dns[::self.subsample, ::self.subsample], self.c_y_dns[::self.subsample,
                                                                               ::self.subsample]

        # Assert that parameters are chosen such that simulation is numerically stable
        assert self._check_cfl_conditions(self.dt, self.dx, self.dy, self.c_x, self.c_y), \
            "CFL conditions not fulfilled"
        assert self._check_cfl_conditions(self.dns_dt, self.dns_dx, self.dns_dy, self.c_x_dns, self.c_y_dns), \
            "CFL conditions in DNS not fulfilled"

        # First state is after having done first simulation step
        self.state = self.upwind_scheme_2d_step(self.initial_condition, self.dt, self.dx, self.dy, self.c_x, self.c_y)
        self.gt_state = dns_initial_condition
        for _ in range(self.subsample):
            self.gt_state = self.rk4_step(self.gt_state, self.dns_dt, self.dns_dx, self.dns_dy,
                                                       self.c_x_dns, self.c_y_dns)

        _info = self._get_info()
        _obs = self._get_obs()
        return _obs, _info

    def step(self, action):
        if not (np.any(self.action_space.low <= action.min()) and np.any(action.max() <= self.action_space.high)):
            action = np.clip(action, self.action_space.low, self.action_space.high)
            print("WARNING: Action is not in action space")

        # load in action and get rid of channel dimension
        action = action[0]
        assert action.shape == self.action_space.shape

        # Clip new state to still be in observation space
        self.rl_adjusted_state = np.clip(self.state - action, self.observation_space.low, self.observation_space.high)
        assert self.observation_space.contains(self.rl_adjusted_state), \
            f"State is not in observation space, " \
            f"min(state): {np.min(self.rl_adjusted_state)}, max(state): {np.max(self.rl_adjusted_state)}"

        # Reward: "How much closer we got to the ground truth  solution"
        _reward = (
                (self.state - self.gt_state[::self.subsample, ::self.subsample]) ** 2
                - (self.rl_adjusted_state - self.gt_state[::self.subsample, ::self.subsample]) ** 2
        )

        # Update state and gt_state
        self.state = self.upwind_scheme_2d_step(self.rl_adjusted_state, self.dt, self.dx, self.dy, self.c_x, self.c_y)
        for _ in range(self.subsample):
            self.gt_state = self.rk4_step(self.gt_state, self.dns_dt, self.dns_dx, self.dns_dy,
                                                       self.c_x_dns, self.c_y_dns)
        self.step_count += 1

        # Termination logic: Stop when we reached the end of the simulation
        _terminated = self._do_terminate() #True if self.step_count == self.ep_len else False
        _truncated = False
        # _terminated = self._do_terminate() if self.train else _truncated
        _obs = self._get_obs()
        _info = self._get_info()
        return _obs, _reward, _terminated, _truncated, _info

    def _do_terminate(self):
        err = self.mean_agent_abs_error(self.state, self.gt_state)
        if err >= self.mse_threshold and self.step_count >= self.ep_len or self.step_count >= self.max_ep_len:
            return True
        else:
            return False

    def mean_agent_abs_error(self, state, gt_state):
        return np.mean(np.abs(state - gt_state[::self.subsample, ::self.subsample]))

    def energ_spec_error(self,
                         pred, label):
        return np.abs((pred - label) / label)

    @staticmethod
    def _check_cfl_conditions(dt, dx, dy, c_x, c_y):
        cfl = (np.max(np.abs(c_x)) * dt / dx) + (np.max(np.abs(c_y)) * dt / dy)
        if cfl <= 1:
            return True
        else:
            print(f"CFL condition is not satisfied: {cfl:.3f} > 1")
            return False

    @staticmethod
    def upwind_scheme_2d_step(u, dt, dx, dy, c_x, c_y):
        u_new = np.zeros_like(u)

        # X direction
        positive_c_x = c_x > 0
        negative_c_x = np.logical_not(positive_c_x)
        # periodic boundary condition in x
        u_new[:, 1:] += (u[:, 1:] - c_x[:, 1:] * dt / dx * (u[:, 1:] - u[:, :-1])) * positive_c_x[:, 1:]
        u_new[:, 0] += (u[:, 0] - c_x[:, 0] * dt / dx * (u[:, 0] - u[:, -1])) * positive_c_x[:, 0]
        # periodic boundary condition in x
        u_new[:, :-1] += (u[:, :-1] - c_x[:, :-1] * dt / dx * (u[:, 1:] - u[:, :-1])) * negative_c_x[:, :-1]
        u_new[:, -1] += (u[:, -1] - c_x[:, -1] * dt / dx * (u[:, 0] - u[:, -2])) * negative_c_x[:, -1]

        # Y direction
        positive_c_y = c_y > 0
        negative_c_y = np.logical_not(positive_c_y)
        u_new[1:, :] += (- c_y[1:, :] * dt / dy * (u[1:, :] - u[:-1, :])) * positive_c_y[1:, :]
        # periodic boundary condition in y
        u_new[0, :] += (- c_y[0, :] * dt / dy * (u[0, :] - u[-1, :])) * positive_c_y[0, :]
        u_new[:-1, :] += (- c_y[:-1, :] * dt / dy * (u[1:, :] - u[:-1, :])) * negative_c_y[:-1, :]
        # periodic boundary condition in y
        u_new[-1, :] += (- c_y[-1, :] * dt / dy * (u[0, :] - u[-2, :])) * negative_c_y[-1, :]

        return u_new

    @staticmethod
    def advection_central(u, dx, dy, c_x, c_y):
        dudx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dx)
        dudy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dy)
        return - c_x * dudx - c_y * dudy

    @staticmethod
    def advection_first_order_upwind(u, dx, dy, c_x, c_y):
        # Initialize the arrays for first-order derivatives
        dudx = np.zeros_like(u)
        dudy = np.zeros_like(u)

        # Compute the x-component using first-order upwind
        dudx_positive_c = u - np.roll(u, 1, axis=1)
        dudx_negative_c = np.roll(u, -1, axis=1) - u
        dudx = np.where(c_x > 0, dudx_positive_c / dx, dudx_negative_c / dx)

        # Compute the y-component using first-order upwind
        dudy_positive_c = u - np.roll(u, 1, axis=0)
        dudy_negative_c = np.roll(u, -1, axis=0) - u
        dudy = np.where(c_y > 0, dudy_positive_c / dy, dudy_negative_c / dy)

        # Combine to get the advection term
        return - c_x * dudx - c_y * dudy

    @staticmethod
    def advection_second_order_upwind(u, dx, dy, c_x, c_y):
        # Compute the x-component using second-order upwind
        dudx_positive_c = 3 * u - 4 * np.roll(u, 1, axis=1) + np.roll(u, 2, axis=1)
        dudx_negative_c = -np.roll(u, -2, axis=1) + 4 * np.roll(u, -1, axis=1) - 3 * u
        dudx = np.where(c_x > 0, dudx_positive_c / (2.0 * dx), dudx_negative_c / (2.0 * dx))

        # Compute the y-component using second-order upwind
        dudy_positive_c = 3 * u - 4 * np.roll(u, 1, axis=0) + np.roll(u, 2, axis=0)
        dudy_negative_c = -np.roll(u, -2, axis=0) + 4 * np.roll(u, -1, axis=0) - 3 * u
        dudy = np.where(c_y > 0, dudy_positive_c / (2.0 * dy), dudy_negative_c / (2.0 * dy))

        # Combine to get the advection term
        return - c_x * dudx - c_y * dudy

    def get_space_disc(self, space_disc):
        if space_disc == 'central':
            return self.advection_central
        elif space_disc == 'first_order_upwind':
            return self.advection_first_order_upwind
        elif space_disc == 'second_order_upwind':
            return self.advection_second_order_upwind
        else:
            raise NotImplementedError(f"Space discretization {space_disc} not implemented")

    def rk4_step(self, u, dt, dx, dy, c_x, c_y, space_disc='central'):
        space_disc = self.get_space_disc(space_disc)
        k1 = dt * space_disc(u, dx, dy, c_x, c_y)
        k2 = dt * space_disc(u + 0.5 * k1, dx, dy, c_x, c_y)
        k3 = dt * space_disc(u + 0.5 * k2, dx, dy, c_x, c_y)
        k4 = dt * space_disc(u + k3, dx, dy, c_x, c_y)
        u_new = u + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return u_new

    def mde_upwind_diffusion_term(self, u, c_x, c_y, dx, dy, dt):
        # Compute first derivative
        du_dx, du_dy = np.gradient(u, dx, dy)

        # Compute second derivative
        d2u_dx2, _ = np.gradient(du_dx, dx, dy)
        _, d2u_dy2 = np.gradient(du_dy, dx, dy)

        # Return diffusion term
        return np.abs(c_x) * dx / 2 * (1. - np.abs(c_x) * dt / dx) * d2u_dx2 \
            + np.abs(c_y) * dy / 2 * (1. - np.abs(c_y) * dt / dy) * d2u_dy2

    def render(self, title: str = 'state'):
        x = np.linspace(0, 1, self.img_size)
        y = np.linspace(0, 1, self.img_size)
        X, Y = np.meshgrid(x, y)
        fig = plt.figure(figsize=(6, 6))
        plt.pcolormesh(X, Y, self.state, shading='auto')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def _play_and_get_eval_metrics(self,
                                   actor: MarlModel,
                                   ep_len: int = 21) -> Batch:

        # Play episode with and without RL
        _total_reward = 0.
        _obs, _ = self.reset()
        _no_rl_state = self.initial_condition.copy()

        # Calculation of true physical invariants:
        _total_mass = self.compute_total_mass(self.initial_condition)
        _total_momentum = self.compute_total_momentum(self.initial_condition, self.c_x, self.c_y)
        _total_kinetic_energy = self.compute_total_kinetic_energy(self.initial_condition, self.c_x, self.c_y)

        for i in range(ep_len):
            # Simulation with MARL in loop
            act_mean = actor.get_action_mean(_obs).clip(self.action_space.low.min(), self.action_space.high.max())

            # Compute RL adjusted update
            _obs, reward, _terminated, _, _ = self.step(act_mean.detach().cpu().numpy()[0])
            _total_reward += reward
            _no_rl_state = self.upwind_scheme_2d_step(_no_rl_state, self.dt, self.dx, self.dy, self.c_x, self.c_y)

        # Storing evaluation metrics in batch
        eval_metrics = Batch()
        eval_metrics['rews'] = _total_reward

        # Evaluation metric: Loss between RL adjusted state and ground truth at end of episode
        eval_metrics['mae_error'] = self.mean_agent_abs_error(self.state, self.gt_state)
        eval_metrics['mae_error_no_rl'] = self.mean_agent_abs_error(_no_rl_state, self.gt_state)

        # Calculation of physical invariants after simulation:
        _total_mass_rl = self.compute_total_mass(self.rl_adjusted_state)
        _total_momentum_rl = self.compute_total_momentum(self.rl_adjusted_state, self.c_x, self.c_y)
        _total_kinetic_energy_rl = self.compute_total_kinetic_energy(self.rl_adjusted_state, self.c_x, self.c_y)

        # Calculation of frequency spectra after simulation
        _, rl_energy_spectrum = np.log(self.compute_energy_spectrum(self.rl_adjusted_state))
        _, no_rl_energy_spectrum = np.log(self.compute_energy_spectrum(_no_rl_state))
        _, gt_energy_spectrum = np.log(self.compute_energy_spectrum(self.gt_state[::self.subsample, ::self.subsample]))

        # with rl errors
        eval_metrics['total_mass_error'] = np.abs((_total_mass_rl - _total_mass) / _total_mass)
        eval_metrics['total_momentum_error'] = np.abs((_total_momentum_rl - _total_momentum) / _total_momentum)
        eval_metrics['total_energy_error'] = np.abs(
            (_total_kinetic_energy_rl - _total_kinetic_energy) / _total_kinetic_energy
        )
        eval_metrics['energy_spectrum_error'] = self.energ_spec_error(rl_energy_spectrum, gt_energy_spectrum)
        eval_metrics['energy_spectrum_error_no_rl'] = self.energ_spec_error(no_rl_energy_spectrum, gt_energy_spectrum)
        # no rl errors
        eval_metrics['total_mass_error_no_rl'] = np.abs(
            (self.compute_total_mass(_no_rl_state) - _total_mass) / _total_mass
        )
        eval_metrics['total_momentum_error_no_rl'] = np.abs(
            (self.compute_total_momentum(_no_rl_state, self.c_x, self.c_y) - _total_momentum) / _total_momentum
        )
        eval_metrics['total_energy_error_no_rl'] = np.abs(
            (self.compute_total_kinetic_energy(_no_rl_state, self.c_x,
                                               self.c_y) - _total_kinetic_energy) / _total_kinetic_energy
        )
        return eval_metrics

    def play_episode_and_log_to_wandb(self,
                                      actor: MarlModel,
                                      step: int,
                                      ep_len: int) -> np.array:
        _min_duration = 9
        assert ep_len > _min_duration, f"test_sim_len must be > {_min_duration}"

        _num_pics = 4
        _plotting_freq = int(ep_len / _num_pics)  # we want to get `_num_pics` pics in total
        _obs, _ = self.reset()
        _no_rl_state = self.initial_condition.copy()

        _log_rl_imgs = []
        _log_no_rl_imgs = []
        _log_dns_imgs = []
        _log_action_imgs = []
        _log_target_acts = []

        # Log initial state
        _log_rl_imgs.append(wandb.Image(self.initial_condition.clip(0, 1), caption=f'state@{self.step_count}'))
        _log_no_rl_imgs.append(wandb.Image(_no_rl_state.clip(0, 1), caption=f'state@{self.step_count}'))
        _log_dns_imgs.append(wandb.Image(self.gt_state[::self.subsample, ::self.subsample].clip(0, 1),
                                         caption=f'dns_state@{self.step_count}'))

        for i in range(ep_len):
            # Simulation with MARL in loop
            act_mean = actor.get_action_mean(_obs).clip(self.action_space.low.min(), self.action_space.high.max())

            # Compute RL adjusted update
            _obs, reward, _terminated, _, _ = self.step(act_mean.detach().cpu().numpy()[0])
            _obs['domain'] = np.expand_dims(_obs['domain'], axis=0)  # need to create fake batch dimension
            _obs['velocity_field'] = np.expand_dims(_obs['velocity_field'], axis=0)

            _no_rl_state = self.upwind_scheme_2d_step(_no_rl_state, self.dt, self.dx, self.dy, self.c_x, self.c_y)

            # Pure numerical simulation -> will lead to numerical errors
            if self.step_count % _plotting_freq == 0:
                _log_rl_imgs.append(wandb.Image(self.rl_adjusted_state.clip(0, 1),
                                                caption=f'state@{self.step_count}'))
                _log_no_rl_imgs.append(wandb.Image(_no_rl_state.clip(0, 1),
                                                   caption=f'state@{self.step_count}'))
                _log_action_imgs.append(wandb.Image(act_mean[0].detach().cpu().numpy()[0],
                                                    caption=f'action@{self.step_count}'))
                _log_dns_imgs.append(wandb.Image(self.gt_state[::self.subsample, ::self.subsample].clip(0, 1),
                                                 caption=f'dns_state@{self.step_count}'))
                _log_target_acts.append(wandb.Image(self.mde_upwind_diffusion_term(self.state, self.c_x, self.c_y, self.dx, self.dy, self.dt),
                                                    caption=f'target_action@{self.step_count}'))

        # Log images of episode to wandb
        _log_rl_imgs.append(wandb.Image(self.rl_adjusted_state.clip(0, 1),
                                        caption='target'))
        _log_no_rl_imgs.append(wandb.Image(_no_rl_state.clip(0, 1),
                                           caption='target'))
        _log_dns_imgs.append(wandb.Image(self.gt_state[::self.subsample, ::self.subsample].clip(0, 1),
                                         caption='target'))

        rl_log_dict = {f'{self.dataset_name} RL in loop': _log_rl_imgs}
        no_rl_log_dict = {f'{self.dataset_name} no RL': _log_no_rl_imgs}
        _log_action_dict = {f'{self.dataset_name} actions': _log_action_imgs}
        _log_target_acts_dict = {f'{self.dataset_name} target actions': _log_target_acts}
        _log_dns_dict = {f'{self.dataset_name} DNS': _log_dns_imgs}
        wandb.log(rl_log_dict, step=step)
        wandb.log(no_rl_log_dict, step=step)
        wandb.log(_log_action_dict, step=step)
        wandb.log(_log_target_acts_dict, step=step)
        wandb.log(_log_dns_dict, step=step)
        return None

    def _get_obs(self):
        # return with extra channel dimension for torch NN
        return {
            'domain': np.expand_dims(self.state, axis=0),
            'velocity_field': np.array([self.c_x, self.c_y], dtype=np.float32)
        }

    @staticmethod
    def translational_analytical_solution(u0, t, c_x, c_y):
        c_x = c_x[0, 0]
        c_y = c_y[0, 0]

        # Get the dimensions of the domain
        ny, nx = u0.shape

        # Create coordinates for the domain
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        # Calculate the shifted coordinates based on advection velocities and time
        x_shifted = np.mod(X - c_x * t, 1)
        y_shifted = np.mod(Y - c_y * t, 1)

        # Interpolate the initial condition at the shifted coordinates
        coords = np.array([X.flatten(), Y.flatten()]).T
        # linear interpolation needed for iterative evaluation of this function
        u_t = np.float32(griddata(coords, u0.flatten(), (x_shifted, y_shifted), method='nearest'))
        return u_t

    @staticmethod
    def compute_total_mass(u: np.ndarray) -> np.ndarray:
        return np.sum(u)

    @staticmethod
    def compute_total_momentum(u: np.ndarray, c_x: np.ndarray, c_y: np.ndarray) -> np.ndarray:
        return np.sum(u * c_x) + np.sum(u * c_y)

    @staticmethod
    def compute_total_kinetic_energy(u: np.ndarray, c_x: np.ndarray, c_y: np.ndarray) -> np.ndarray:
        return 0.5 * np.sum(u * (c_x ** 2 + c_y ** 2))

    @staticmethod
    def compute_energy_spectrum(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The compute_energy_spectrum function calculates the energy spectrum of a two-dimensional input signal using the
        discrete Fourier transform. It selects only the positive frequencies and computes the power spectrum within a
        given frequency range around each frequency value. The energy across both the x and y directions is summed and
        then doubled to obtain the total energy in the input signal.

        :param u: 2D numpy array that the energy spectrum is computed for
        :return: positive frequencies and energy spectrum each as 1D numpy array of len u.shape[0] / 2
        """
        freqs_x = np.fft.fftfreq(u.shape[1])
        freqs_y = np.fft.fftfreq(u.shape[0])

        # Only consider positive frequencies
        idx_x = np.where(freqs_x >= 0)
        idx_y = np.where(freqs_y >= 0)
        freqs_x = freqs_x[idx_x]
        freqs_y = freqs_y[idx_y]

        power_ua = np.abs(np.fft.fft2(u)) ** 2
        power_ua = power_ua[idx_y[0], :][:, idx_x[0]]
        freq_range = 0.1  # frequency range around each frequency value
        energy = np.zeros_like(power_ua)

        # Create boolean masks for selecting Fourier coefficients within the frequency range
        freqs_x_mask = np.logical_and(freqs_x[:, None] >= freqs_x - freq_range / 2,
                                      freqs_x[:, None] <= freqs_x + freq_range / 2)
        freqs_y_mask = np.logical_and(freqs_y >= freqs_y[:, None] - freq_range / 2,
                                      freqs_y <= freqs_y[:, None] + freq_range / 2)

        # Use boolean indexing to select the Fourier coefficients within the frequency range
        energy = power_ua[np.ix_(freqs_y_mask.any(axis=1), freqs_x_mask.any(axis=0))]

        # Double the energy values to account for symmetric negative frequencies
        energy = energy * 2

        # Sum the energy across all frequencies in each direction
        energy_x = np.sum(energy, axis=0)
        energy_y = np.sum(energy, axis=1)
        energy_total = (energy_x + energy_y) * 2
        return freqs_x, energy_total




def main():
    env = AdvectionEnvironment()
    env.reset()
    env.render(title='IC')
    no_action = np.zeros((env.img_size, env.img_size))
    random_action = 0.1 * np.random.rand(env.img_size, env.img_size)

    for i in range(5):
        _obs, _reward, _terminated, _truncated, _info = env.step(random_action)
        print(_reward)
        if _terminated:
            print('DONE at ', i)
            break
        env.render(title=f'state {i}')
    env.step(random_action)
    env.render(title='random action')


if __name__ == "__main__":
    main()
