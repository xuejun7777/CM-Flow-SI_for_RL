import copy, math, os
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger

from torch_ema import ExponentialMovingAverage
from agents.stochastic_interpolants import StochasticInterpolants
from agents.model import MLP, Unet, LN_Resnet, InterpolantsMLP
from agents.helpers import EMA, kerras_boundaries
from agents.ql_diffusion import Critic
from bridger.networks.conditional_unet_1D_si import *
from bridger.dataset.load_dataset import set_model_prior


class ExponentialScheduler():
    def __init__(self, v_start=1.5, v_final=0.2, decay_steps=2*10**2):
        """A scheduler for exponential decay.

        :param v_start: starting value of epsilon, default 1. as purely random policy 
        :type v_start: float
        :param v_final: final value of epsilon
        :type v_final: float
        :param decay_steps: number of steps from eps_start to eps_final
        :type decay_steps: int
        """
        self.v_start = v_start
        self.v_final = v_final
        self.decay_steps = decay_steps
        self.value = self.v_start
        self.ini_frame_idx = 0
        self.current_frame_idx = 0

    def reset(self, ):
        """ Reset the scheduler """
        self.ini_frame_idx = self.current_frame_idx

    def step(self):
        """
        The choice of eps_decay:
        ------------------------
        start = 1
        final = 0.01
        decay = 10**6  # the decay steps can be 1/10 over all steps 10000*1000
        final + (start-final)*np.exp(-1*(10**7)/decay)
        
        => 0.01

        """
        self.current_frame_idx += 1
        delta_frame_idx = self.current_frame_idx - self.ini_frame_idx
        self.value = self.v_final + (self.v_start - self.v_final) * math.exp(-1. * delta_frame_idx / self.decay_steps)
        return self.value

    def get_value(self, ):
        return self.value

# https://github.com/Kinyugo/consistency_models/blob/main/consistency_models/consistency_models.py
def timesteps_schedule(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 2,
    final_timesteps: int = 150,
) -> int:
    """Implements the proposed timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.
    """
    num_timesteps = final_timesteps**2 - initial_timesteps**2
    num_timesteps = current_training_step * num_timesteps / total_training_steps
    num_timesteps = math.ceil(math.sqrt(num_timesteps + initial_timesteps**2) - 1)

    return num_timesteps + 1


def improved_timesteps_schedule(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 10,
    final_timesteps: int = 1280,
) -> int:
    """Implements the improved timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    total_training_steps_prime = math.floor(
        total_training_steps
        / (math.log2(math.floor(final_timesteps / initial_timesteps)) + 1)
    )
    num_timesteps = initial_timesteps * math.pow(
        2, math.floor(current_training_step / total_training_steps_prime)
    )
    num_timesteps = min(num_timesteps, final_timesteps) + 1

    return int(num_timesteps)


def lognormal_timestep_distribution(
    num_samples: int,
    sigmas: Tensor,
    mean: float = -1.1,
    std: float = 2.0,
) -> Tensor:
    """Draws timesteps from a lognormal distribution.

    Parameters
    ----------
    num_samples : int
        Number of samples to draw.
    sigmas : Tensor
        Standard deviations of the noise.
    mean : float, default=-1.1
        Mean of the lognormal distribution.
    std : float, default=2.0
        Standard deviation of the lognormal distribution.

    Returns
    -------
    Tensor
        Timesteps drawn from the lognormal distribution.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    pdf = torch.erf((torch.log(sigmas[1:]) - mean) / (std * math.sqrt(2))) - torch.erf(
        (torch.log(sigmas[:-1]) - mean) / (std * math.sqrt(2))
    )
    pdf = pdf / pdf.sum()

    timesteps = torch.multinomial(pdf, num_samples, replacement=True)

    return timesteps

class SI_QL(object):
    def __init__(self,
                 model_args,
                 prior_args,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=2,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 q_norm=False,
                 adaptive_ema=False,
                 steps_per_epoch=1000,
                 ):

        self.device = device
        self.ema_decay = ema_decay
        self.ema = EMA(ema_decay)
        model = StochasticInterpolants(model_args)
        # prior_args = {"env_name": model_args['env_name']}
        # prior_args['device'] = device
        # prior_args['prior_policy'] = model_args['prior_policy']
        # prior_args['seed'] = model_args['seed']
        # prior_args['action_dim'] = model_args['action_dim']

        self.actor = set_model_prior(model, prior_args)
        self.load_model(model_args=model_args, device=device)
        
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm
        self.q_norm = q_norm

        self.step = 0
        self.step_start_ema = step_start_ema

        self.update_ema_every = update_ema_every

        self.critic = Critic(model_args['obs_dim'], model_args['action_dim']).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup
        self.adaptive_ema = adaptive_ema
        self.steps_per_epoch = steps_per_epoch

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_actor, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        loss_ema = None

        for itr in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_actor(next_state_rpt, self.device)
                target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.ema_actor(next_state, self.device)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)

            target_q = (reward + not_done * self.discount * target_q).detach()
            
            # normalize the target_q
            if self.q_norm:
                target_q = (target_q) / (target_q.std() + 1e-6)

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2) # return grad value is before clipped, although the grad for update is clipped
            self.critic_optimizer.step()

            # loss for Q-learning
            new_action = self.actor(state, self.device)
            q1_new_action, q2_new_action = self.critic(state, new_action)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()

            """ Policy Training """

            if len(state) > 0:
                # loss for BC with stochastic interpolants
                bc_loss, detail_loss = self.actor.loss(state, action, device=self.device)
            else:
                bc_loss, detail_loss = torch.zeros((1,), device=self.device), {'v_loss': torch.zeros((1,), device=self.device), 's_loss': torch.zeros((1,), device=self.device), 'b_loss': torch.zeros((1,), device=self.device)}
            
            mean_bc_loss = bc_loss.mean()
            if loss_ema is None:
                loss_ema = mean_bc_loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * mean_bc_loss.item()

            actor_loss = mean_bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()


            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                    log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                log_writer.add_scalar('Actor Loss', actor_loss.item(), self.step)
                log_writer.add_scalar('BC Loss', mean_bc_loss.item(), self.step)
                log_writer.add_scalar('v_Loss', detail_loss['v_loss'].item(), self.step)
                log_writer.add_scalar('s_Loss', detail_loss['s_loss'].item(), self.step)
                log_writer.add_scalar('b_Loss', detail_loss['b_loss'].item(), self.step)
                log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
                log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
                log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(mean_bc_loss.item())
            metric['ql_loss'].append(q_loss.item())
            metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor(state_rpt, self.device)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value), 1)
        return action[idx].cpu().data.numpy().flatten()

    def load_model(self, model_args, device):

        if model_args['net_type'] == 'MLP_si':
            # self.actor.net = InterpolantsConditionalUnet1D(
            #     input_dim=model_args['action_dim'],
            #     global_cond_dim=model_args['obs_dim']
            # )
            self.actor.net = InterpolantsMLP(state_dim=model_args['obs_dim'], action_dim=model_args['action_dim'], device=device)
            # self.actor.ema =  ExponentialMovingAverage(self.actor.net.parameters(), decay=0.75)
        else:
            raise NotImplementedError

        self.ema_actor = copy.deepcopy(self.actor)
        if model_args['pretrain']:
            checkpoint = torch.load(os.path.join(model_args['ckpt_path'], "si_actor.pt"), map_location="cpu")
            self.actor.load_state_dict(checkpoint['actor'])
            self.ema_actor.load_state_dict(checkpoint["ema_actor"])

        self.actor.to(device)
        self.ema_actor.to(device)

    def save_model(self, ckpt_path, itr):
        torch.save({
            "actor": self.actor.state_dict(),
            "ema_actor": self.ema_actor.state_dict(),
        }, os.path.join(ckpt_path, "si_actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(ckpt_path, "critic.pt"))

        torch.save({
            "actor": self.actor.state_dict(),
            "ema_actor": self.ema_actor.state_dict(),
        }, os.path.join(ckpt_path, f"si_actor_{itr}.pt"))
        torch.save(self.critic.state_dict(), os.path.join(ckpt_path, f"critic_{itr}.pt"))
        
