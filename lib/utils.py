from abc import ABC, abstractmethod
from datetime import datetime
import pickle
import os



class Config(ABC):
    """
    Abstract class for configuration parameters.
    """
    @abstractmethod
    def __init__(self):
        self.architecture = None
        self.iterations = None
        self.gpu = None
        self.img_size = None
        self.max_epochs = None
        self.exp_name = None
        self.dataset = None
        self.batch_size = None
        self.num_epochs = None
        self.algo = None
        self.env = None
        self.discount = None
        self.subsample = None
        self.ep_len = None
        self.vel_type = None
        self.ent_coef = None
        self.eval_data = None
        self.eval_vel_type = None
        self.lr = None
        self.repeat_per_collect = None
        self.eval_ep_len = None
        self.note: str = ""
        self.n_eval_eps = None
        self.eval_env = None
        self.test_ep_len = None
        self.SEED = None

    def as_dict(self):
        return vars(self)

    @abstractmethod
    def model_name(self):
        NotImplementedError()

    def _set_attributes(self, _args):
        # iterate through arguments and create attributes of config object
        for arg_name, arg_val in _args.__dict__.items():
            setattr(self, arg_name, arg_val)

    def model_read_id(self):
        """
        :return: name for model reading. Is compatible with supervised and RL model save names
        such that they can be loaded similarly.
        """
        if self.env == "deconv":
            return f"{self.architecture}_it:{self.iterations}_seed:{self.SEED}_img:{self.img_size}"
        elif self.env == "advection" or self.env == "diffusion" or self.env == "burgers":
            return self.model_name()
        else:
            NotImplementedError(f"model_read_id for environment {self.env} not implemented.")

    def model_save_id(self):
        """
        :return: name for model saving. Same as model_read_id plus timestamp to prevent overwriting.
        """
        return f"{self.model_read_id()}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"


def save_dict_to_file(dict_obj, filename):
    with open(filename, 'w') as f:
        for key, value in dict_obj.items():
            f.write(f'{key}: {value}\n')


def save_batch_to_file(batch_obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(batch_obj, f)


#def log_img_to_wandb(img: torch.Tensor,
#                     input: torch.Tensor,
#                     output: torch.Tensor,
#                     keyword: str):
#    img_sample, input_sample, output_sample = detach_move_tensors_to_cpu(img[0, 0], input[0, 0], output[0, 0])
#    wandb.log({f"{keyword} training correction: target | pred": [create_image_from_tensor(img_sample - input_sample),
#                                                      create_image_from_tensor(output_sample - input_sample)]})
#    wandb.log({f"{keyword} prediction: input | pred | target": [create_image_from_tensor(input_sample),
#                                                              create_image_from_tensor(output_sample),
#                                                              create_image_from_tensor(img_sample)]})
#    wandb.log({f"{keyword} error map:": create_image_from_tensor((img_sample - output_sample) ** 2)})


#def create_image_from_tensor(tensor: torch.Tensor):
#    return wandb.Image(tensor)


#def dict_to_wandb_table(config: dict):
#    table = wandb.Table(columns=["Hyperparameter", "Value"])
#    for key, value in config.items():
#        table.add_data(key, str(value))
#    wandb.log({'Config': table})


#def sample_from_dataloader(dataiter, dataloader):
#    try:
#        _sample = next(dataiter)
#    except StopIteration:
#        dataiter = iter(dataloader)
#        _sample = next(dataiter)
#    return _sample, dataiter


#def detach_move_tensors_to_cpu(*tensors):
#    torchvision.transforms.ToPILImage()
#    return (t.cpu().detach()for t in tensors)


#def str2bool(v):
#    if isinstance(v, bool):
#        return v
#    if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
#        return True
#    elif v.lower() in ('no', 'False',  'false', 'f', 'n', '0'):
#        return False
#    else:
#        raise argparse.ArgumentTypeError('Boolean value expected.')


#def restrict_to_num_threads(num_threads: int):
#    # Specify the number of cores you want to use
#    os.environ["OMP_NUM_THREADS"] = str(num_threads)
#    os.environ["MKL_NUM_THREADS"] = str(num_threads)
#    # Also consider limiting the number of threads PyTorch can use internally for its operations
#    torch.set_num_threads(num_threads)


def model_name(config):
    # creates a name from the relevant hyperparameters
    setup_name = f"{config.environment}_{config.setup}_{config.algorithm}"
    if not os.path.exists("../results/weights/"+setup_name):
        os.makedirs("../results/weights/"+setup_name)

    return "../results/weights/"+setup_name
