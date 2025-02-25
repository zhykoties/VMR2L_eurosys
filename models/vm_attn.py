import logging
import os
import torch
import torch.nn as nn
from .components.pt_transformer import Transformer


logger = logging.getLogger('VM.attn')


class VM_candidate_model(nn.Module):
    def __init__(self, params):
        super(VM_candidate_model, self).__init__()
        self.device = params.device
        # self.seq_length = params.num_pm
        self.d_hidden = params.d_hidden

        self.pm_encode = nn.Linear(params.pm_cov, params.d_hidden)
        self.vm_encode = nn.Linear(params.vm_cov, params.d_hidden)

        self.transformer = Transformer(d_model=params.d_hidden, nhead=params.num_head,
                                       num_encoder_layers=params.transformer_blocks,
                                       num_decoder_layers=params.transformer_blocks, dim_feedforward=params.d_ff,
                                       activation='gelu', batch_first=True, dropout=params.dropout,
                                       need_attn_weights=True, device=params.device)

        self.output_layer = nn.Linear(params.d_hidden, 1)
        self.critic_layer = nn.Linear(params.d_hidden, 1)
        self.critic_token = -torch.ones(1, 1, params.d_hidden).to(self.device)

    def forward(self, vm_states, num_step_states, pm_states, num_vms_mask=None, return_attns=False):
        b_sz = vm_states.shape[0]
        transformer_output = self.transformer(src=torch.cat([num_step_states.repeat(1, 1, self.d_hidden),
                                                             self.pm_encode(pm_states)], dim=1),
                                              tgt_key_padding_mask=torch.cat([num_vms_mask,
                                                                              torch.zeros(b_sz, 1, dtype=torch.bool,
                                                                                          device=self.device)], dim=1),
                                              tgt=torch.cat([self.vm_encode(vm_states),
                                                             self.critic_token.repeat(b_sz, 1, 1).detach()], dim=1))
        score = torch.squeeze(self.output_layer(transformer_output[0][:, :-1]))
        critic_score = self.critic_layer(transformer_output[0][:, -1])
        if return_attns:
            return score, critic_score, transformer_output[1]
        else:
            return score, critic_score


class VM_Attn_Wrapper(nn.Module):
    def __init__(self, params, pretrain=False):
        super(VM_Attn_Wrapper, self).__init__()
        self.model = VM_candidate_model(params).to(params.device)
        if pretrain:
            model_save_path = './saved_model_weights/best.tar'
            assert os.path.isfile(model_save_path)
            self.model.load_state_dict(torch.load('./saved_model_weights/best.tar')['state_dict'])
            self.model.critic_layer = nn.Linear(params.d_hidden, 1).to(params.device)
