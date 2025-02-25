"""
VM: cpu, cpu, mem, mem, cpu % 16, cpu % 16  (0 is full, 1 is empty)
PM: cpu, cpu, mem, mem, fragment_rate, cpu % 16, fragment_rate, cpu % 16
cpu % 16 = round(normalized_cpu * 88) % 16 / 16
fragment_rate = round(normalized_cpu * 88) % 16 / round(normalized_cpu * 88)
To rescale memory, mem * 368776
"""

import argparse
import os
import random
import time
from distutils.util import strtobool

import pandas as pd
import wandb

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from tqdm import trange

import gym_reschdule_combination.envs.vm_rescheduler_env

import models
import utils
from env_patch import AsyncVectorEnv_Patch
from main import make_env, CategoricalMasked
from eval import parse_args


class Agent(nn.Module):
    def __init__(self, vm_net, pm_net, params, args_model):
        super(Agent, self).__init__()

        self.vm_net = vm_net
        self.pm_net = pm_net
        self.device = params.device
        self.model = args_model
        self.num_vm = params.num_vm
        self.vm_threshold = 0.995
        self.pm_threshold = 0.98

    def get_value(self, obs_info_pm, obs_info_all_vm, obs_info_num_steps, obs_info_num_vms):
        num_vms_mask = torch.arange(self.num_vm, device=obs_info_all_vm.device)[None, :] >= obs_info_num_vms[:, None]
        if self.model == "attn":
            return self.vm_net(obs_info_all_vm, obs_info_num_steps, obs_info_pm, num_vms_mask)[1]
        elif self.model == "mlp":
            return self.vm_net(obs_info_all_vm, obs_info_pm)[1]

    def get_action_and_value(self, envs, obs_info_pm, obs_info_all_vm, obs_info_num_steps, obs_info_num_vms,
                             pm_mask=None, selected_vm=None, selected_pm=None):
        if pm_mask is None:
            assert selected_vm is None and selected_pm is None, \
                'action must be None when action_mask is not given!'
        else:
            assert selected_vm is not None and selected_pm is not None, \
                'action must be given when action_mask is given!'
        num_vms_mask = torch.arange(self.num_vm, device=self.device)[None, :] >= obs_info_num_vms[:, None]

        b_sz = obs_info_pm.shape[0]
        # obs_info_all_vm: torch.Size([8, 2089, 14])
        # obs_info_pm:  torch.Size([8, 279, 8])
        if self.model == "attn":
            vm_logits, critic_score = self.vm_net(obs_info_all_vm, obs_info_num_steps, obs_info_pm, num_vms_mask)
        elif self.model == "mlp":
            vm_logits, critic_score = self.vm_net(obs_info_all_vm, obs_info_pm)
        else:
            raise ValueError(f'self.model={self.model} is not implemented')
        # vm_pred:  torch.Size([8, 2089])
        # critic_score:  torch.Size([8, 1])
        vm_cat_check = CategoricalMasked(logits=vm_logits, masks=num_vms_mask)

        low_prob_vm_mask = []
        vm_cat_check_probs = vm_cat_check.probs.data.cpu().numpy()
        for prob_dim in range(vm_cat_check_probs.shape[0]):
            dim_quantile = np.quantile(vm_cat_check_probs[prob_dim], self.vm_threshold)
            low_prob_vm_mask.append(vm_cat_check.probs[prob_dim] < dim_quantile)
        low_prob_vm_mask = torch.stack(low_prob_vm_mask, dim=0)

        vm_cat = CategoricalMasked(logits=vm_logits, masks=torch.logical_or(num_vms_mask, low_prob_vm_mask))

        if selected_vm is None:
            selected_vm = vm_cat.sample()
        vm_log_prob = vm_cat.log_prob(selected_vm)

        if pm_mask is None:
            pm_mask = torch.tensor(np.array(envs.call_parse('get_pm_mask', vm_id=selected_vm.cpu().tolist())),
                                   dtype=torch.bool, device=self.device)  # pm_mask:  torch.Size([8, 279])

        # obs_info_all_vm:  torch.Size([8, 2089, 14])
        if self.model == "attn":
            pm_logits = self.pm_net(obs_info_all_vm[torch.arange(b_sz), selected_vm].unsqueeze(1), obs_info_num_steps,
                                    obs_info_pm)  # b_sz
        elif self.model == "mlp":
            pm_logits = self.pm_net(obs_info_all_vm[torch.arange(b_sz), selected_vm].unsqueeze(1), obs_info_pm)  # b_sz
        else:
            raise ValueError(f'self.model={self.model} is not implemented')
        # pm_logits:  torch.Size([8, 279])
        pm_cat_check = CategoricalMasked(logits=pm_logits, masks=pm_mask)

        low_prob_mask = []
        pm_cat_check_probs = pm_cat_check.probs.data.cpu().numpy()
        for prob_dim in range(pm_cat_check_probs.shape[0]):
            dim_quantile = np.quantile(pm_cat_check_probs[prob_dim], self.pm_threshold)
            low_prob_mask.append(pm_cat_check.probs[prob_dim] < dim_quantile)
        low_prob_mask = torch.stack(low_prob_mask, dim=0)

        pm_cat = CategoricalMasked(logits=pm_logits, masks=torch.logical_or(pm_mask, low_prob_mask))
        if selected_pm is None:
            selected_pm = pm_cat.sample()  # selected_pm:  torch.Size([8])
        pm_log_prob = pm_cat.log_prob(selected_pm)  # pm_log_prob:  torch.Size([8])
        log_prob = vm_log_prob + pm_log_prob
        entropy = vm_cat.entropy() + pm_cat.entropy()

        return selected_vm, selected_pm, log_prob, entropy, critic_score, pm_mask


if __name__ == "__main__":
    args = parse_args()
    num_train = args.num_train
    dev_idx_start = 6000
    num_dev = 200
    num_test = 200
    num_envs = args.num_envs
    num_steps = args.num_steps
    less_feature = args.less_feature
    run_name = f'{args.restore_name}'
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    torch.backends.cudnn.benchmark = True
    torch.set_default_dtype(torch.float32)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    envs = AsyncVectorEnv_Patch(
        [make_env(args.gym_id, args.seed + i, args.vm_data_size, args.max_steps, num_train,
                  args.normalize, affinity=args.affinity) for i in range(num_envs)]
    )

    params = utils.Params(f'./experiments/pretrain/{args.model}/params.json')
    params.update('./data/params_M.json')
    params.device = device
    params.batch_size = args.num_envs
    if less_feature:
        params.pm_cov = 6
        params.vm_cov = 10
    else:
        params.pm_cov = 8
        params.vm_cov = 14

    envs.call('set_save_json', save_json_flag=args.save_json, save_json_dir=os.path.join('runs', args.restore_name),
              save_json_file_name=args.restore_file)

    # input the vm candidate model
    if args.model == 'attn':
        vm_cand_model = models.VM_Attn_Wrapper(params, args.pretrain).model
        pm_cand_model = models.PM_Attn_Wrapper(params).model
    elif args.model == 'mlp':
        vm_cand_model = models.VM_MLP_Wrapper(params, args.pretrain).model
        pm_cand_model = models.PM_MLP_Wrapper(params).model
    else:
        raise ValueError(f'args.model = {args.model} is not defined!')

    agent = Agent(vm_cand_model, pm_cand_model, params, args.model)
    optim = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    agent.eval()
    global_step = utils.load_checkpoint(args.restore_name, args.restore_file, agent)
    print(f"- Restored file (global step {global_step}) "
          f"from {os.path.join(args.restore_name, args.restore_file + '.pth.tar')}")

    if args.track:
        wandb.watch(agent, log_freq=100)

    # ALGO Logic: Storage setup
    obs_vm = torch.zeros(num_steps, args.num_envs, params.num_vm, params.vm_cov, device=device)
    obs_pm = torch.zeros(num_steps, args.num_envs, params.num_pm, params.pm_cov, device=device)
    obs_num_steps = torch.zeros(num_steps, args.num_envs, 1, 1, device=device)
    obs_num_vms = torch.zeros(num_steps, args.num_envs, dtype=torch.int32, device=device)
    vm_actions = torch.zeros(num_steps, args.num_envs, device=device)
    pm_actions = torch.zeros(num_steps, args.num_envs, device=device)
    logprobs = torch.zeros(num_steps, args.num_envs, device=device)
    rewards = torch.zeros(num_steps, args.num_envs, device=device)
    dones = torch.zeros(num_steps, args.num_envs, device=device)
    values = torch.zeros(num_steps, args.num_envs, device=device)
    # envs.single_action_space.nvec: [2089, 279] (#vm, #pm)
    action_masks = torch.zeros(num_steps, args.num_envs, envs.single_action_space.nvec[1], dtype=torch.bool,
                               device=device)

    if args.debug:
        col_names = ['step']
        for i in range(params.num_vm):
            for j in range(params.vm_cov):
                col_names.append(f'vm_{i}_cov_{j}')

        for i in range(params.num_pm):
            for j in range(params.pm_cov):
                col_names.append(f'pm_{i}_cov_{j}')

        col_names += ['num_steps', 'num_vms', 'vm_action', 'pm_action', 'logprob', 'rewards', 'done']
        col_names += ['values', 'ep_return', 'fragment_rate']
        plot_step = np.tile(np.expand_dims(np.arange(num_steps), -1), 3).reshape((num_steps, 3, 1))

    with torch.no_grad():
        envs.call('set_mode', mode='dev')

        dev_all_frag_rate = np.ones((num_dev, num_steps))
        dev_all_min_frag_rate = np.ones((num_dev, num_steps))
        dev_pbar = trange(0, num_dev, num_envs, desc='Dev')
        for file_index in dev_pbar:
            file_ids = [dev_idx_start + file_index + env_id for env_id in range(num_envs)]
            envs.call_parse('set_current_env', env_id=file_ids)

            current_ep_info = np.zeros((num_steps, args.num_envs, 2)) + 1000  # return, len, fr
            next_obs_dict = envs.reset()
            next_obs_pm = torch.tensor(next_obs_dict['pm_info'], device=device)  # torch.Size([8, 279, 8])
            next_obs_vm = torch.tensor(next_obs_dict['vm_info'], device=device)  # torch.Size([8, 279, 14])
            if less_feature:
                next_obs_pm = torch.cat([next_obs_pm[:, :, :2],
                                         next_obs_pm[:, :, 4:]], dim=-1)
                next_obs_vm = torch.cat([next_obs_vm[:, :, :2], next_obs_vm[:, :, 4:8],
                                         next_obs_vm[:, :, 10:]], dim=-1)
            next_obs_num_steps = torch.Tensor(next_obs_dict['num_steps']).to(device)
            next_obs_num_vms = torch.tensor(next_obs_dict['num_vms'], dtype=torch.int32, device=device)
            next_done = torch.zeros(args.num_envs, device=device)

            for step in range(0, num_steps):
                obs_pm[step] = next_obs_pm
                obs_vm[step] = next_obs_vm
                obs_num_steps[step] = next_obs_num_steps
                obs_num_vms[step] = next_obs_num_vms
                dones[step] = next_done

                vm_action, pm_action, logprob, _, value, action_mask \
                    = agent.get_action_and_value(envs, next_obs_pm, next_obs_vm, next_obs_num_steps, next_obs_num_vms)
                values[step] = value.flatten()  # value:  torch.Size([8, 1])
                action_masks[step] = action_mask
                vm_actions[step] = vm_action
                pm_actions[step] = pm_action
                logprobs[step] = logprob

                # print(f'vm_action: {vm_action.cpu().numpy()}, pm_action: {pm_action.cpu().numpy()}')
                next_obs_dict, reward, done, info = envs.step(torch.stack([vm_action, pm_action],
                                                                          dim=-1).cpu().numpy())
                next_obs_pm = torch.Tensor(next_obs_dict['pm_info']).to(device)
                next_obs_vm = torch.Tensor(next_obs_dict['vm_info']).to(device)
                if less_feature:
                    next_obs_pm = torch.cat([next_obs_pm[:, :, :2],
                                             next_obs_pm[:, :, 4:]], dim=-1)
                    next_obs_vm = torch.cat([next_obs_vm[:, :, :2], next_obs_vm[:, :, 4:8],
                                             next_obs_vm[:, :, 10:]], dim=-1)
                next_obs_num_steps = torch.Tensor(next_obs_dict['num_steps']).to(device)
                next_obs_num_vms = torch.tensor(next_obs_dict['num_vms'], dtype=torch.int32, device=device)
                rewards[step] = torch.tensor(reward, device=device).view(-1)
                next_done = torch.Tensor(done).to(device)

                for env_id, item in enumerate(info):
                    dev_all_frag_rate[file_index + env_id, step] = item['fragment_rate']
                    current_ep_info[step, env_id, 1] = item['fragment_rate']
                    if "episode" in item.keys():
                        current_ep_info[step, env_id, 0] = item["episode"]["r"]
                        dev_all_min_frag_rate[file_index + env_id, step] = item['fragment_rate']

                if args.debug:
                    plot_obs_vm = obs_vm[:, :3].cpu().data.numpy().reshape(num_steps, 3, -1)
                    plot_obs_pm = obs_pm[:, :3].cpu().data.numpy().reshape(num_steps, 3, -1)
                    plot_obs_num_steps = obs_num_steps[:, :3].cpu().data.numpy().reshape(num_steps, 3, -1)
                    plot_obs_num_vms = obs_num_vms[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
                    plot_vm_actions = vm_actions[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
                    plot_pm_actions = pm_actions[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
                    plot_logprobs = logprobs[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
                    plot_rewards = rewards[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
                    plot_dones = dones[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
                    plot_values = values[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
                    plot_ep_info = current_ep_info[:, :3]
                    plot_update_all = np.swapaxes(np.concatenate([plot_step, plot_obs_vm, plot_obs_pm, plot_obs_num_steps,
                                                                  plot_obs_num_vms, plot_vm_actions, plot_pm_actions,
                                                                  plot_logprobs, plot_rewards, plot_dones,
                                                                  plot_values, plot_ep_info], axis=-1), axis1=1, axis2=0)
                    plot_update_all = plot_update_all.reshape((num_steps * 3, -1))
                    episode_df = pd.DataFrame(plot_update_all, columns=col_names)
                    plot_fr_mean = np.mean(plot_ep_info[:, :, 2][plot_ep_info[:, :, 2] != -1000])
                    episode_df.to_pickle(f'runs/{run_name}/dev_{dev_idx_start + file_index}'
                                         f'-{dev_idx_start + file_index + 2}.pkl')

        for i in range(num_dev):
            print(f'dev {i}: {dev_all_min_frag_rate[i][dev_all_min_frag_rate[i] != 1]}')
        current_dev_frag_rate = np.mean(np.amin(dev_all_min_frag_rate, axis=1))
        np.save(os.path.join('runs', args.restore_name, 'dev_all_frag_rate.npy'), dev_all_frag_rate)
        print(f'Dev fragment rate: {current_dev_frag_rate:.4f}')

        envs.call('set_mode', mode='test')

        print('max_steps: ', args.max_steps)
        test_all_min_frag_rate = np.ones((num_test, num_steps))
        test_pbar = trange(0, num_test, num_envs, desc='Test')
        for file_index in test_pbar:
            file_ids = [dev_idx_start + num_dev + file_index + env_id for env_id in range(num_envs)]
            envs.call_parse('set_current_env', env_id=file_ids)

            current_ep_info = np.zeros((num_steps, args.num_envs, 2)) - 1000  # return, len, fr
            next_obs_dict = envs.reset()
            next_obs_pm = torch.tensor(next_obs_dict['pm_info'], device=device)  # torch.Size([8, 279, 8])
            next_obs_vm = torch.tensor(next_obs_dict['vm_info'], device=device)  # torch.Size([8, 279, 14])
            if less_feature:
                next_obs_pm = torch.cat([next_obs_pm[:, :, :2],
                                         next_obs_pm[:, :, 4:]], dim=-1)
                next_obs_vm = torch.cat([next_obs_vm[:, :, :2], next_obs_vm[:, :, 4:8],
                                         next_obs_vm[:, :, 10:]], dim=-1)
            next_obs_num_steps = torch.Tensor(next_obs_dict['num_steps']).to(device)
            next_obs_num_vms = torch.tensor(next_obs_dict['num_vms'], dtype=torch.int32, device=device)
            next_done = torch.zeros(args.num_envs, device=device)

            trial = 0
            for step in range(0, num_steps):
                obs_pm[step] = next_obs_pm
                obs_vm[step] = next_obs_vm
                obs_num_steps[step] = next_obs_num_steps
                obs_num_vms[step] = next_obs_num_vms
                dones[step] = next_done

                vm_action, pm_action, logprob, _, value, action_mask \
                    = agent.get_action_and_value(envs, next_obs_pm, next_obs_vm, next_obs_num_steps, next_obs_num_vms)
                values[step] = value.flatten()  # value:  torch.Size([8, 1])
                action_masks[step] = action_mask
                vm_actions[step] = vm_action
                pm_actions[step] = pm_action
                logprobs[step] = logprob

                # print(f'vm_action: {vm_action.cpu().numpy()}, pm_action: {pm_action.cpu().numpy()}')
                next_obs_dict, reward, done, info = envs.step(torch.stack([vm_action, pm_action],
                                                                          dim=-1).cpu().numpy())
                next_obs_pm = torch.Tensor(next_obs_dict['pm_info']).to(device)
                next_obs_vm = torch.Tensor(next_obs_dict['vm_info']).to(device)
                if less_feature:
                    next_obs_pm = torch.cat([next_obs_pm[:, :, :2],
                                             next_obs_pm[:, :, 4:]], dim=-1)
                    next_obs_vm = torch.cat([next_obs_vm[:, :, :2], next_obs_vm[:, :, 4:8],
                                             next_obs_vm[:, :, 10:]], dim=-1)
                next_obs_num_steps = torch.Tensor(next_obs_dict['num_steps']).to(device)
                next_obs_num_vms = torch.tensor(next_obs_dict['num_vms'], dtype=torch.int32, device=device)
                rewards[step] = torch.tensor(reward, device=device).view(-1)
                next_done = torch.Tensor(done).to(device)

                for env_id, item in enumerate(info):
                    current_ep_info[step, env_id, 1] = item['fragment_rate']
                    if "episode" in item.keys():
                        current_ep_info[step, env_id, 0] = item["episode"]["r"]
                        test_all_min_frag_rate[file_index + env_id, step] = item['fragment_rate']

                if args.debug:
                    plot_obs_vm = obs_vm[:, :3].cpu().data.numpy().reshape(num_steps, 3, -1)
                    plot_obs_pm = obs_pm[:, :3].cpu().data.numpy().reshape(num_steps, 3, -1)
                    plot_obs_num_steps = obs_num_steps[:, :3].cpu().data.numpy().reshape(num_steps, 3, -1)
                    plot_obs_num_vms = obs_num_vms[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
                    plot_vm_actions = vm_actions[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
                    plot_pm_actions = pm_actions[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
                    plot_logprobs = logprobs[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
                    plot_rewards = rewards[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
                    plot_dones = dones[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
                    plot_values = values[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
                    plot_ep_info = current_ep_info[:, :3]
                    plot_update_all = np.swapaxes(np.concatenate([plot_step, plot_obs_vm, plot_obs_pm, plot_obs_num_steps,
                                                                  plot_obs_num_vms, plot_vm_actions, plot_pm_actions,
                                                                  plot_logprobs, plot_rewards, plot_dones,
                                                                  plot_values, plot_ep_info], axis=-1), axis1=1, axis2=0)
                    plot_update_all = plot_update_all.reshape((num_steps * 3, -1))
                    episode_df = pd.DataFrame(plot_update_all, columns=col_names)
                    plot_fr_mean = np.mean(plot_ep_info[:, :, 2][plot_ep_info[:, :, 2] != -1000])
                    episode_df.to_pickle(f'runs/{run_name}/'
                                         f'test_{dev_idx_start + num_dev + file_index}'
                                         f'-{dev_idx_start + num_dev + file_index + 2}.pkl')

        current_test_frag_rate = np.mean(np.amin(test_all_min_frag_rate, axis=1))
        print(f'Test fragment rate: {current_test_frag_rate:.4f}')

        np.save(f"runs/{run_name}/{args.restore_file}_dev_all_min_frag_rate_base_quant.npy", dev_all_min_frag_rate)
        np.save(f"runs/{run_name}/{args.restore_file}_test_all_min_frag_rate_base_quant.npy", test_all_min_frag_rate)

    envs.close()
