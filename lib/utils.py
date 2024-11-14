import torch
import torchvision
import argparse
from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import wandb
import re
import pickle
import os
from tqdm import tqdm
from xlb_flows.utils import vorticity_2d, energy_spectrum_2d


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


#plotting utils 
# Custom sorting key
def custom_sort_key(folder):
    match = re.search(r'N(\d+)_', folder)
    if match:
        N = int(match.group(1))
        return (0 if N < 2048 else 1, N)
    return (1, 0)

# Custom sorting key names
def custom_sort_key_names(folder):
    match = re.search(r'(\d+)_', folder)
    if match:
        N = int(match.group(1))
        return (0 if N < 2048 else 1, N)
    return (1, 0)

# get all filenames in directory
def get_names(path):
    files = os.listdir(path)
    return np.sort(files)


def plot_correlations(corrs, names, N, M):
    t = np.linspace(0, 227, M)
    idex = np.argmin(np.abs(t - 26))
    their_t = t/16
    fig, ax = plt.subplots(figsize=(10,6), dpi=300)
    for i in range(N-1):
        plt.plot(t, corrs[:,i], label=names[i])
    plt.plot(t, corrs[:,N-1], 'k--', label=names[N-1])
    plt.xlabel(r'non-dimensional time $T = (U/L)t$')
    plt.ylabel(r'vorticity correlation with DNS')
    plt.legend(loc='best')
    plt.show()


def plot_spectra(spec_mean, spec_std, k, names, N):
    # plot the averaged spectrum
    plt.figure(figsize=(10,6), dpi=300)
    plt.set_cmap('cool')
    for i in range(N-1):
        plt.loglog(spec_mean[i,...]*k**5, label=names[i])
        plt.fill_between(k,
                         spec_mean[i,...]*k**5 - spec_std[i,...]*k**5,
                         spec_mean[i,...]*k**5 + spec_std[i,...]*k**5,
                         alpha=0.3
                         )
    plt.loglog(spec_mean[-1,...]*k**5,
               'k--',
               label=names[-1]
               )
    plt.fill_between(k,
                     (k**5)*(spec_mean[-1,...]-spec_std[-1,...]),
                     (k**5)*(spec_mean[-1,...] + spec_std[-1,...]), 
                     alpha=0.3,
                     color='k'
                     )
    plt.ylabel(r'Energy spectrum $E(k)k^5$')
    plt.xlabel(r'wavenumber $k$')
    plt.legend()
    plt.show()  

    # plot the averaged spectrum
    plt.figure(figsize=(10,6), dpi=300)
    plt.set_cmap('cool')
    for i in range(N-1):
        plt.loglog(spec_mean[i,...], label=names[i])
        plt.fill_between(k,
                         spec_mean[i,...]-spec_std[i,...],
                         spec_mean[i,...] + spec_std[i,...],
                         alpha=0.3
                         )
    plt.loglog(spec_mean[-1,...],
               'k--',
               label=names[-1]
               )
    plt.fill_between(k,
                     spec_mean[-1,...]-spec_std[-1,...],
                     spec_mean[-1,...] + spec_std[-1,...],
                     alpha=0.3,
                     color='k'
                     )
    #plt.loglog(1e2*k**(-5.), label="k^-5/3", linestyle="--")
    plt.ylabel(r'Energy spectrum $E(k)$')
    plt.xlabel(r'wavenumber $k$')
    plt.legend()
    plt.show()


def create_plots(test_directory):
    # Get all folder names within `test_directory`
    subfolders = [name for name in os.listdir(test_directory) if os.path.isdir(os.path.join(test_directory, name))]
    subfolders = np.sort(subfolders)

    # Sort folders with custom key
    subfolders = sorted(subfolders, key=custom_sort_key)
    
    # Extract names={N}_{model} from each folder name
    names = []
    for folder in subfolders:
        match = re.search(r'N(\d+)_S\d+_U\d+_(\w+)', folder)
        if match:
            N = match.group(1)
            model = match.group(2)
            names.append(f"{N}_{model}")

    names = sorted(names, key=custom_sort_key_names)
    paths = [test_directory+subfolder+"/" for subfolder in subfolders]

    #load all velocity files for all paths, and compute vorticity correlation
    images = [get_names(path) for path in paths]
    N = len(paths)
    M = len(images[0])-1
    corrs = np.zeros((M,N))
    print("computing vorticity correlations:")
    for i in tqdm(range(M)):
        #load all velocities 
        velocities = np.array([np.load(paths[j]+'/'+images[j][i]) for j in range(N)])
        vorticities = np.array([vorticity_2d(velocities[j,...], 2.0*np.pi/128) for j in range(N)])
        corrs[i,...] = np.array([np.corrcoef(vorticities[j,...].flatten(), vorticities[-1,...].flatten())[0, 1] 
                                 for j in range(N)])
    #plot correlations
    plot_correlations(corrs, names, N, M)

    #compute enerty spectrum statistics
    L = int(128/2 - 1)
    a = 1*M//2 #1*M//2
    b = M-1
    spec = np.zeros((N, L, (b-a)))
    k = np.arange(L)
    # loop over all files in files1 between a and b and add to spec
    print("computing energy spectra:")
    for i in tqdm(range(a, b)):
        velocities = np.array([np.load(paths[j]+'/'+images[j][i]) for j in range(N)])
        for j in range(N):
            _, spec[j,:,i-a] = energy_spectrum_2d(velocities[j,...])
    spec_mean=spec.mean(axis=-1)
    spec_std=spec.std(axis=-1)

    # plot spectra
    plot_spectra(spec_mean, spec_std, k, names, N)


    # plot vorticity fields
    f_size = 10

    # Sample time steps for demonstration
    ts = np.array([0, 1/3, 2/3, 1]) * (M-2)
    time_labels = np.array([0, 1/3, 2/3, 1]) * (227)

    # Plot vorticity images at times t1 to t5 for all resolutions
    fig, axs = plt.subplots(N, 4, figsize=(4.2, N+1), dpi=300)

    for i in range(4):
        for j in range(N):
            u = np.load(paths[j] + '/' + images[j][int(ts[i])]) 
            v = vorticity_2d(u, 2.0 * np.pi / 128)       
            axs[j, i].imshow(v, vmin=-10, vmax=10, cmap=sns.cm.icefire)
            axs[j, i].set_xticks([])  # Disable x-ticks
            axs[j, i].set_yticks([])  # Disable y-ticks

    # Add row labels
    for i, ax in enumerate(axs[:, 0]):
        ax.set_ylabel(f"{names[i]}", fontsize=f_size)

    # Add time labels to each column
    for i, ax in enumerate(axs[0, :]):
        ax.set_title(f"t={time_labels[i]:.1f}", fontsize=f_size)

    # Set up the colorbar with continuous color gradient, arrowed edges, and specified ticks
    vmin, vmax = -10, 10
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Adjust to span entire height
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=sns.cm.icefire, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, extend='both', extendfrac=0.07, aspect=10)  # Extend arrows and adjust aspect ratio
    cbar.set_label('Vorticity', fontsize=f_size)
    cbar.ax.tick_params(labelsize=f_size)

    # Set 5 evenly spaced ticks on the colorbar
    cbar.set_ticks(np.linspace(vmin, vmax, 5))

    # Adjust spacing between subplots
    plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.1, wspace=0.06, hspace=0.015)

    plt.show()


