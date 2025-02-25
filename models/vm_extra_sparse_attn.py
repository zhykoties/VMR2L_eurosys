import logging
import os
import torch
import torch.nn as nn
from .components.pt_transformer import TransformerUltraSparseDecoder, TransformerUltraSparseDecoderLayer

logger = logging.getLogger('VM.attn')


class VM_candidate_model(nn.Module):
    def __init__(self, params):
        super(VM_candidate_model, self).__init__()
        self.device = params.device
        self.num_pm = params.num_pm
        self.num_vm = params.num_vm
        self.seq_len = self.num_pm + self.num_vm + 2
        self.num_head = params.num_head

        self.d_hidden = params.d_hidden

        self.pm_encode = nn.Linear(params.pm_cov, params.d_hidden)
        self.vm_encode = nn.Linear(params.vm_cov, params.d_hidden)

        decoder_layer = TransformerUltraSparseDecoderLayer(d_model=params.d_hidden, nhead=params.num_head,
                                                           split_point=self.num_pm + 1,
                                                           dim_feedforward=params.d_ff, dropout=params.dropout,
                                                           activation='gelu', batch_first=True, norm_first=True,
                                                           need_attn_weights=True, device=params.device)
        self.transformer = TransformerUltraSparseDecoder(decoder_layer=decoder_layer,
                                                         num_layers=params.transformer_blocks)

        self.output_layer = nn.Linear(params.d_hidden, 1)
        self.critic_layer = nn.Linear(params.d_hidden, 1)
        self.critic_token = -torch.ones(1, 1, params.d_hidden).to(self.device)

    def forward(self, vm_states, num_step_states, pm_states, vm_pm_relation, num_vms_mask=None, return_attns=False):
        b_sz = vm_states.shape[0]
        local_mask = torch.ones(b_sz, self.seq_len, self.seq_len, dtype=torch.bool, device=self.device)
        local_mask[:, 0, 0] = 0
        local_mask[:, -1, -1] = 0
        local_mask[:, 1:-1, 1:-1] = vm_pm_relation != vm_pm_relation[:, None, :, 0]
        local_mask = local_mask.view(b_sz, 1, self.seq_len, self.seq_len). \
            expand(-1, self.num_head, -1, -1).reshape(b_sz * self.num_head, self.seq_len, self.seq_len)
        # print('local mask: ', (vm_pm_relation != vm_pm_relation[:, None, :, 0])[0])
        tgt_key_pad_mask = torch.zeros(b_sz, self.seq_len, dtype=torch.bool, device=self.device)
        tgt_key_pad_mask[:, 1 + self.num_pm:-1] = num_vms_mask
        transformer_output = self.transformer(src=torch.cat([num_step_states.repeat(1, 1, self.d_hidden),
                                                             self.pm_encode(pm_states)], dim=1),
                                              tgt=torch.cat([self.vm_encode(vm_states),
                                                             self.critic_token.repeat(b_sz, 1, 1).detach()], dim=1),
                                              local_mask=local_mask, tgt_key_padding_mask=tgt_key_pad_mask)
        score = torch.squeeze(self.output_layer(transformer_output[1][:, :-1]))
        critic_score = self.critic_layer(transformer_output[1][:, -1])
        if return_attns:
            return transformer_output, score, critic_score, transformer_output[2]
        else:
            return transformer_output, score, critic_score


class VM_Extra_Sparse_Attn_Wrapper(nn.Module):
    def __init__(self, params, pretrain=False):
        super(VM_Extra_Sparse_Attn_Wrapper, self).__init__()
        self.model = VM_candidate_model(params).to(params.device)
        if pretrain:
            model_save_path = './saved_weights/sparse_attn.tar'
            # './saved_model_weights/sparse_attn.tar'
            assert os.path.isfile(model_save_path)
            self.model.load_state_dict(torch.load(model_save_path)['state_dict'])
