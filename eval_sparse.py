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
from distutils.util import strtobool

import pandas as pd
import wandb

import gym
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

import gym_reschdule_combination.envs.vm_rescheduler_env

import models
import utils
from env_patch import AsyncVectorEnv_Patch
from main import make_env
from ultra_attn import Agent


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="attn_graph", help="model architecture")
    parser.add_argument("--restore-name", type=str, required=True, help="restore experiment name")
    parser.add_argument("--restore-file", type=str, required=True, help="restore file name")
    parser.add_argument("--pretrain", action='store_true',
                        help="if toggled, we will restore pretrained weights for vm selection")
    parser.add_argument("--gym-id", type=str, default="graph-v2",
                        help="the id of the gym environment")
    parser.add_argument("--vm-data-size", type=str, default="M", choices=["M", "L", "multi", "M_small", "M_medium"],
                        help="size of the dataset")
    parser.add_argument("--max-steps", type=int, default=50, help="maximum number of redeploy steps")
    parser.add_argument("--num-train", type=int, default=4000, help="number of files used for training")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--normalize", action='store_true',
                        help="if toggled, we will normalize the input features")
    parser.add_argument("--less-feature", action='store_true',
                        help="if toggled, we will use six features for PMs")
    parser.add_argument("--track", action='store_true',
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--debug", action='store_true',
                        help="if toggled, this experiment will save run details")
    parser.add_argument("--no-detail-pm", action='store_true',
                        help="if toggled, we will add pm embeddings and attn score to pm decoder")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=8,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=250,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--save-json", action='store_true',
                        help="save vm-pm mapping result to json if toggled")
    parser.add_argument("--affinity", type=int, default=0, help="number of conflicts per VM")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.detail_pm = not args.no_detail_pm
    return args


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

    # TRY NOT TO MODIFY: seeding
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
    params.update(f'./data/params_{args.vm_data_size}.json')
    params.device = device
    params.batch_size = args.num_envs
    if less_feature:
        params.pm_cov = 6
        params.vm_cov = 10
    else:
        params.pm_cov = 8
        params.vm_cov = 6

    envs.call('set_save_json', save_json_flag=args.save_json, save_json_dir=os.path.join('runs', args.restore_name),
              save_json_file_name=args.restore_file)

    # input the vm candidate model
    vm_cand_model = models.VM_Extra_Sparse_Attn_Wrapper(params, args.pretrain).model
    pm_cand_model = models.PM_Detail_Attn_Wrapper(params).model

    agent = Agent(vm_cand_model, pm_cand_model, params, args.model,
                  args.detail_pm)
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
    obs_edges = torch.zeros(num_steps, num_envs, params.num_vm + params.num_pm, 1, dtype=torch.int32, device=device)
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
            next_obs_edges = torch.tensor(next_obs_dict['edges'], dtype=torch.int32, device=device)
            next_done = torch.zeros(args.num_envs, device=device)

            for step in range(0, num_steps):
                obs_pm[step] = next_obs_pm
                obs_vm[step] = next_obs_vm
                obs_num_steps[step] = next_obs_num_steps
                obs_num_vms[step] = next_obs_num_vms
                obs_edges[step] = next_obs_edges
                dones[step] = next_done

                vm_action, pm_action, logprob, _, value, action_mask \
                    = agent.get_action_and_value(envs, next_obs_pm, next_obs_vm, next_obs_edges,
                                                 next_obs_num_steps, next_obs_num_vms)
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
                next_obs_edges = torch.tensor(next_obs_dict['edges'], dtype=torch.int32, device=device)
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
                                                                  plot_values, plot_ep_info], axies=-1), axis1=1, axis2=0)
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

        test_all_min_frag_rate = np.ones((num_test, num_steps))
        test_pbar = trange(0, num_test, num_envs, desc='Test')
        for file_index in test_pbar:
            file_ids = [dev_idx_start + num_dev + file_index + env_id for env_id in range(num_envs)]
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
            next_obs_edges = torch.tensor(next_obs_dict['edges'], dtype=torch.int32, device=device)
            next_done = torch.zeros(args.num_envs, device=device)

            for step in range(0, num_steps):
                obs_pm[step] = next_obs_pm
                obs_vm[step] = next_obs_vm
                obs_num_steps[step] = next_obs_num_steps
                obs_num_vms[step] = next_obs_num_vms
                obs_edges[step] = next_obs_edges
                dones[step] = next_done

                vm_action, pm_action, logprob, _, value, action_mask \
                    = agent.get_action_and_value(envs, next_obs_pm, next_obs_vm, next_obs_edges,
                                                 next_obs_num_steps, next_obs_num_vms)
                values[step] = value.flatten()  # value:  torch.Size([8, 1])
                action_masks[step] = action_mask
                vm_actions[step] = vm_action
                pm_actions[step] = pm_action
                logprobs[step] = logprob

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
                next_obs_edges = torch.tensor(next_obs_dict['edges'], dtype=torch.int32, device=device)
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
                                                                  plot_values, plot_ep_info], axies=-1), axis1=1, axis2=0)
                    plot_update_all = plot_update_all.reshape((num_steps * 3, -1))
                    episode_df = pd.DataFrame(plot_update_all, columns=col_names)
                    plot_fr_mean = np.mean(plot_ep_info[:, :, 2][plot_ep_info[:, :, 2] != -1000])
                    episode_df.to_pickle(f'runs/{run_name}/'
                                         f'test_{dev_idx_start + num_dev + file_index}'
                                         f'-{dev_idx_start + num_dev + file_index + 2}.pkl')

        current_test_frag_rate = np.mean(np.amin(test_all_min_frag_rate, axis=1))
        print(f'Test fragment rate: {current_test_frag_rate:.4f}')

        np.save(f"runs/{run_name}/{args.restore_file}_dev_all_min_frag_rate.npy", dev_all_min_frag_rate)
        np.save(f"runs/{run_name}/{args.restore_file}_test_all_min_frag_rate.npy", test_all_min_frag_rate)

    envs.close()
