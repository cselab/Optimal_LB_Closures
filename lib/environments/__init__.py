from lib.environments.advection import AdvectionEnvironment

__all__ = ["get_environment",
           "AdvectionEnvironment",
           "BurgersEnvironment"]


def get_environment(env_name, img_size, dataset, train, subsample, ep_len, velocity_field_type):

    if env_name == "advection":
        return AdvectionEnvironment(img_size=img_size,
                                    train=train,
                                    subsample=subsample,
                                    ep_len=ep_len,
                                    dataset_name=dataset,
                                    velocity_field_type=velocity_field_type)

    else:
        raise ValueError(f"Environment {env_name} not implemented!")
