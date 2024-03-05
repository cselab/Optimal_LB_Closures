import wandb
import numpy as np

from abc import ABC
from gymnasium import spaces
from copy import deepcopy

from lib.environments.base import BaseEnvironment
from lib.environments.velocity_generation import VelocityFieldGenerator
from lib.models import MarlModel
from scipy.signal import convolve2d
from tianshou.data import Batch


class BurgersEnvironment(BaseEnvironment, ABC):
    def __init__(self,
                 ep_len: int = 5,
                 train: bool = True,
                 num_points_cgs: int = 30,
                 subsample: int = 8,
                 velocity_field_type: str = 'train2'):
        super().__init__()
        self.train = train
        self.num_burn_in_steps = 0

        # Physical parameters
        self.viscosity = 0.003
        self.max_velocity = 1.0

        # Settings for CGS
        self.step_count = 0
        self.ep_len = ep_len
        self.num_points_cgs = num_points_cgs
        self.max_ep_len = 200
        self.mae_threshold = 1. #1.5
        self.dx_cgs, self.dy_cgs = 1 / self.num_points_cgs, 1 / self.num_points_cgs

        # Setting for DNS FGS
        self.subsample_fac = subsample
        self.filter_width = subsample
        self.num_points_fgs = num_points_cgs * subsample
        self.dx_fgs, self.dy_fgs = 1 / self.num_points_fgs, 1 / self.num_points_fgs

        # Velocity variables
        self.u_cgs, self.v_cgs = None, None  # get reset randomly for each episode
        self.u_fgs, self.v_fgs = None, None  # get reset randomly for each episode
        self.velocity_field_generator = VelocityFieldGenerator(velocity_field_type,
                                                               self.max_velocity,
                                                               self.num_points_fgs)
        self.dt_cgs = 0.9 * min(self.dx_cgs, self.dy_cgs)
        self.T = self.ep_len * self.dt_cgs
        self.num_fine_steps_per_coarse_step = 2 * self.subsample_fac
        self.dt_fgs = self.dt_cgs / self.num_fine_steps_per_coarse_step
        self.nt_fgs = int(self.T / self.dt_fgs)

        # limit action space to [-1, 1] to avoid too large changes
        self.action_space = spaces.Box(low=-1.,
                                       high=1.,
                                       shape=(2, self.num_points_cgs, self.num_points_cgs),
                                       dtype=np.float32)

        # data is zero mean and unit variance, choose observation space in range [-3, 3] to account for outliers
        self.observation_space = spaces.Box(low=-3.,
                                            high=3.,
                                            shape=(self.num_points_cgs, self.num_points_cgs),  # TODO should we not have channel dim here?
                                            dtype=np.float32)

    def reset(self, *args, **kwargs):
        # reset tracking
        self.step_count = 0

        # initialize FGS velocity fields
        self.u_fgs, self.v_fgs = self.velocity_field_generator.generate()
        for _ in range(self.num_burn_in_steps):
            self.fgs_step()

        # initialize CGS velocity fields (filter and then subsample)
        u_bar_fgs, v_bar_fgs = self.get_coarse_grid_repr_of_fgs_velocities()
        self.u_cgs, self.v_cgs = deepcopy(u_bar_fgs), deepcopy(v_bar_fgs)

        # advance FGS and CGS simulations
        self.fgs_step()
        self.cgs_step()

        info = self._get_info()
        obs = self._get_obs()
        return obs, info

    def step(self, action):
        if not (np.any(self.action_space.low <= action.min()) and np.any(action.max() <= self.action_space.high)):
            action = np.clip(action, self.action_space.low, self.action_space.high)
            print("WARNING: Action is not in action space")

        assert action.shape == self.action_space.shape
        # Clip new state to still be in observation space
        u_cgs_rl_adj = np.clip(self.u_cgs - self.dt_cgs * action[0], self.observation_space.low, self.observation_space.high)
        v_cgs_rl_adj = np.clip(self.v_cgs - self.dt_cgs * action[1], self.observation_space.low, self.observation_space.high)

        # reward wrt to both velocity components
        u_bar_fgs, v_bar_fgs = self.get_coarse_grid_repr_of_fgs_velocities()
        _reward = (
                (self.u_cgs - u_bar_fgs) ** 2
                - (u_cgs_rl_adj - u_bar_fgs) ** 2
        )
        _reward += (
                (self.v_cgs - v_bar_fgs) ** 2
                - (v_cgs_rl_adj - v_bar_fgs) ** 2
        )
        normalizer = np.mean(np.abs(u_bar_fgs) + np.abs(v_bar_fgs))
        _reward /= normalizer

        # update CGS velocities
        self.u_cgs, self.v_cgs = u_cgs_rl_adj, v_cgs_rl_adj

        self.fgs_step()
        self.cgs_step()
        self.step_count += 1

        # Termination logic: Stop when we reached the end of the simulation
        _terminated = self._do_terminate()
        _truncated = False
        _obs = self._get_obs()
        _info = self._get_info()
        return _obs, _reward, _terminated, _truncated, _info

    def _do_terminate(self):
        err = self.compute_normalized_mae_wrt_fgs(self.u_cgs, self.v_cgs)
        if err >= self.mae_threshold and self.step_count >= self.ep_len or self.step_count >= self.max_ep_len:
            return True
        else:
            return False

    def _get_obs(self):
        # return with extra channel dimension for torch NN
        return {
            "velocity_field": np.array([self.u_cgs, self.v_cgs], dtype=np.float32),  # return velocities in channel - dims: (2, H, W)
        }

    def _get_info(self):
        return {}

    def fgs_step(self):
        for _ in range(self.num_fine_steps_per_coarse_step):
            self.u_fgs, self.v_fgs = self.burgers_time_step(self.u_fgs, self.v_fgs, self.viscosity, self.dx_fgs, self.dy_fgs, self.dt_fgs)

    def cgs_step(self):
        self.u_cgs, self.v_cgs = self.burgers_time_step(self.u_cgs, self.v_cgs, self.viscosity, self.dx_cgs, self.dy_cgs, self.dt_cgs)

    def burgers_time_step(self, u, v, viscosity, dx, dy, dt):
        conv_x, conv_y = self.convection_term_1st_order_upwind(u, v, dx, dy)
        diff_x, diff_y = self.diffusion_term_2nd_order_central(u, v, viscosity, dx, dy)
        u = self.forward_euler(u, conv_x + diff_x, dt)
        v = self.forward_euler(v, conv_y + diff_y, dt)
        return u, v

    # Discretization schemes
    @staticmethod
    def upwind_derivative(u, axis, increment):
        dudx_positive_u = u - np.roll(u, 1, axis=axis)
        dudx_negative_u = np.roll(u, -1, axis=axis) - u
        dudx = np.where(u >= 0, dudx_positive_u / increment, dudx_negative_u / increment)
        return dudx

    def convection_term_1st_order_upwind(self, u, v, dx, dy):
        # Compute the x-component using first-order upwind
        dudx = self.upwind_derivative(u, axis=1, increment=dx)
        # Compute the y-component using first-order upwind
        dudy = self.upwind_derivative(u, axis=0, increment=dy)

        # Compute the x-component using first-order upwind
        dvdx = self.upwind_derivative(v, axis=1, increment=dx)
        # Compute the y-component using first-order upwind
        dvdy = self.upwind_derivative(v, axis=0, increment=dy)

        # Combine to get the advection term
        return - u * dudx - v * dudy, - u * dvdx - v * dvdy

    @staticmethod
    def central_difference_2nd_order(u, dx, axis):
        d2udx2 = (np.roll(u, -1, axis=axis) - 2 * u + np.roll(u, 1, axis=axis)) / dx ** 2
        return d2udx2

    def diffusion_term_2nd_order_central(self, u, v, nu, dx, dy):
        # Compute the second-order derivatives
        d2udx2 = self.central_difference_2nd_order(u, dx, axis=1)
        d2udy2 = self.central_difference_2nd_order(u, dy, axis=0)

        d2vdx2 = self.central_difference_2nd_order(v, dx, axis=1)
        d2vdy2 = self.central_difference_2nd_order(v, dy, axis=0)

        # Combine to get the diffusion term
        u_diff = nu * (d2udx2 + d2udy2)
        v_diff = nu * (d2vdx2 + d2vdy2)

        return u_diff, v_diff

    @staticmethod
    def forward_euler(u, rhs, dt):
        return u + dt * rhs

    # Connection between CGS and FGS
    def subsample(self, u):
        shift = self.filter_width // 2
        return u[shift::self.subsample_fac, shift::self.subsample_fac]

    def average_filter(self, u):
        kernel = np.ones((self.filter_width, self.filter_width)) / (self.filter_width * self.filter_width)
        # Apply convolution
        u_bar = convolve2d(u, kernel, mode='same', boundary='wrap')
        return u_bar

    def average_filter_and_subsample(self, u, v):
        u_bar = self.subsample(self.average_filter(u))
        v_bar = self.subsample(self.average_filter(v))
        return u_bar, v_bar

    def get_coarse_grid_repr_of_fgs_velocities(self):
        u_bar_fgs, v_bar_fgs = self.average_filter_and_subsample(self.u_fgs, self.v_fgs)
        return u_bar_fgs, v_bar_fgs

    # Flow quantities
    @staticmethod
    def compute_reynolds_number(u, v, viscosity):
        Re = np.max(np.sqrt(u ** 2 + v ** 2)) / viscosity
        return Re

    def compute_velocity_magnitude(self, u, v):
        return np.sqrt(u ** 2 + v ** 2)

    def current_cgs_velocity_magnitude(self):
        return self.compute_velocity_magnitude(self.u_cgs, self.v_cgs)

    def current_fgs_velocity_magnitude(self):
        u_bar_fgs, v_bar_fgs = self.get_coarse_grid_repr_of_fgs_velocities()
        return self.compute_velocity_magnitude(u_bar_fgs, v_bar_fgs)

    def compute_normalized_mae_wrt_fgs(self, u, v):
        return np.mean(self.compute_local_normalized_mae_wrt_fgs(u, v))

    def compute_local_normalized_mae_wrt_fgs(self, u, v):
        u_bar_fgs, v_bar_fgs = self.get_coarse_grid_repr_of_fgs_velocities()
        normalizer = np.mean(np.abs(u_bar_fgs) + np.abs(v_bar_fgs))
        return (np.abs(u - u_bar_fgs) + np.abs(v - v_bar_fgs)) / normalizer

    def create_local_mae_img(self, u, v, caption):
        max_err = 0.5
        img = self.compute_local_normalized_mae_wrt_fgs(u, v)
        img = np.clip(img, 0, max_err) / max_err
        # convert to ints in range [0, 255]
        img = (img * 255).astype(np.uint8)
        return wandb.Image(img, caption=caption)

    def create_action_imgs(self, action):
        # need to clip quite radically because actions are small
        u_img, v_img = action[0], action[1]
        assert u_img.shape == (self.num_points_cgs, self.num_points_cgs)
        return wandb.Image(u_img, caption=f"u action@{self.step_count}"), wandb.Image(v_img, caption=f"v action@{self.step_count}")

    def play_episode_and_log_to_wandb(self,
                                      actor: MarlModel,
                                      step: int,
                                      ep_len: int) -> None:
        _min_duration = 9
        assert ep_len > _min_duration, f"test_sim_len must be > {_min_duration}"

        _num_pics = 4
        _plotting_freq = int(ep_len / _num_pics)  # we want to get `_num_pics` pics in total
        _obs, _ = self.reset()
        u_cgs_no_rl, v_cgs_no_rl = deepcopy(self.u_cgs), deepcopy(self.v_cgs)

        _log_cgs_errs = []
        _log_cgs_errs_no_rl = []
        _log_u_action_imgs = []
        _log_v_action_imgs = []
        _log_vel_magnitude_no_rl = []
        _log_vel_magnitude_rl = []
        _log_vel_magnitude_fgs = []

        for i in range(ep_len):
            # Simulation with MARL in loop
            act_mean = actor.get_action_mean(_obs).clip(self.action_space.low.min(), self.action_space.high.max())

            # Compute RL adjusted update
            _obs, reward, _terminated, _, _ = self.step(act_mean.detach().cpu().numpy()[0])
            _obs['velocity_field'] = np.expand_dims(_obs['velocity_field'], axis=0) # need to create fake batch dimension

            # Compute no RL update
            u_cgs_no_rl, v_cgs_no_rl = self.burgers_time_step(u_cgs_no_rl, v_cgs_no_rl, self.viscosity,
                                                              self.dx_cgs, self.dy_cgs, self.dt_cgs)

            if self.step_count % _plotting_freq == 0:
                # log MAE wrt FGS
                _log_cgs_errs.append(self.create_local_mae_img(self.u_cgs, self.v_cgs, f"mae@{self.step_count}"))
                _log_cgs_errs_no_rl.append(self.create_local_mae_img(u_cgs_no_rl, v_cgs_no_rl, f"mae_no_rl@{self.step_count}"))
                u_img, v_img = self.create_action_imgs(act_mean.detach().cpu().numpy()[0])

                # log actions
                #_log_u_action_imgs.append(u_img)
                _log_v_action_imgs.append(v_img)

                # log velocity magnitudes
                _log_vel_magnitude_no_rl.append(wandb.Image(self.compute_velocity_magnitude(u_cgs_no_rl, v_cgs_no_rl), caption=f"vel_mag CGS@{self.step_count}"))
                _log_vel_magnitude_rl.append(wandb.Image(self.current_cgs_velocity_magnitude(), caption=f"vel_mag CNN-MARL@{self.step_count}"))
                _log_vel_magnitude_fgs.append(wandb.Image(self.current_fgs_velocity_magnitude(), caption=f"vel_mag FGS@{self.step_count}"))

        cgs_log_dict = {f'CNN-MARL MAE map': _log_cgs_errs}
        cgs_no_rl_log_dict = {f'CGS MAE map': _log_cgs_errs_no_rl}
        #u_action_log_dict = {f'u_actions': _log_u_action_imgs}
        v_action_log_dict = {f'v_actions': _log_v_action_imgs}
        vel_mag_fgs_log_dict = {f'FGS velocity magnitude': _log_vel_magnitude_fgs}
        vel_mag_no_rl_log_dict = {f'CGS velocity magnitude': _log_vel_magnitude_no_rl}
        vel_mag_rl_log_dict = {f'CNN-MARL velocity magnitude': _log_vel_magnitude_rl}
        wandb.log({**cgs_log_dict, **cgs_no_rl_log_dict, **v_action_log_dict,
                   **vel_mag_no_rl_log_dict, **vel_mag_rl_log_dict, **vel_mag_fgs_log_dict})
        return None

    def _play_and_get_eval_metrics(self,
                                   actor: MarlModel,
                                   ep_len: int = 21) -> Batch:
        # Storing evaluation metrics in batch
        eval_metrics = Batch()

        _total_reward = 0.
        _obs, _ = self.reset()
        u_cgs_no_rl, v_cgs_no_rl = deepcopy(self.u_cgs), deepcopy(self.v_cgs)

        for i in range(ep_len):
            # Simulation with MARL in loop
            act_mean = actor.get_action_mean(_obs).clip(self.action_space.low.min(), self.action_space.high.max())
            _obs, reward, _terminated, _, _ = self.step(act_mean.detach().cpu().numpy()[0])
            _total_reward += reward

            # Simulation without MARL in loop
            u_cgs_no_rl, v_cgs_no_rl = self.burgers_time_step(u_cgs_no_rl, v_cgs_no_rl, self.viscosity, self.dx_cgs, self.dy_cgs, self.dt_cgs)

            # Store intermediate metrics
            if i % 10 == 0:
                eval_metrics[f'mae_error@{i}'] = self.compute_normalized_mae_wrt_fgs(self.u_cgs, self.v_cgs)
                eval_metrics[f'mae_error_no_rl@{i}'] = self.compute_normalized_mae_wrt_fgs(u_cgs_no_rl, v_cgs_no_rl)

        # Store final metrics
        eval_metrics['rews'] = _total_reward
        eval_metrics['mae_error'] = self.compute_normalized_mae_wrt_fgs(self.u_cgs, self.v_cgs)
        eval_metrics['mae_error_no_rl'] = self.compute_normalized_mae_wrt_fgs(u_cgs_no_rl, v_cgs_no_rl)
        return eval_metrics

