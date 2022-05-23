import logging
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

class Adapter(nn.Module):
    def __init__(self, config, dneck):
        super().__init__()
        nx = config.n_embd if hasattr(config, "n_embd") else config.d_model
        self.ln = nn.LayerNorm(nx, eps=config.layer_norm_epsilon) if hasattr(config, "layer_norm_epsilon") else nn.LayerNorm(nx)
        self.we = nn.Linear(nx, dneck, bias=False)
        self.wd = nn.Linear(dneck, nx, bias=False)

    def forward(self, x):
        a = self.we(self.ln(x))
        m = self.wd(F.relu(a))
        output = x + m
        return output


class MultiAdapter(nn.Module):
    def __init__(self, config, dneck, nk):
        super().__init__()
        nx = config.n_embd if hasattr(config, "n_embd") else config.d_model
        self.nk = nk
        self.ln = nn.ModuleList(nn.LayerNorm(nx, eps=config.layer_norm_epsilon) if hasattr(config, "layer_norm_epsilon") else nn.LayerNorm(nx) for _ in range(self.nk)) 
        
        we = torch.empty(nk, nx, dneck)
        nn.init.normal_(we, std=0.02)
        wd = torch.empty(nk, dneck, nx)
        nn.init.normal_(wd, std=0.02)
        self.we = nn.Parameter(we)
        self.wd = nn.Parameter(wd)
    
    def init_pretraiend_params(self, params):
        """
        params: state dicts [list of parameters for each adapters in one layer]
        [
            {
                "ln.weight": [],
                "ln.bias": [],
                "we.weight": [],
                "wd.weight": [],
            },
        ]
        """
        state_dict = {}
        for i in range(self.nk):
            single_params = params[i]
            state_dict.update({f"ln.{str(i)}.weight": single_params["ln.weight"], f"ln.{str(i)}.bias": single_params["ln.bias"]})

        we_params = torch.stack([params[i]["we.weight"].permute(1, 0) for i in range(self.nk)], dim=0)
        wd_params = torch.stack([params[i]["wd.weight"].permute(1, 0) for i in range(self.nk)], dim=0)
        state_dict.update({"we": we_params, "wd": wd_params})

        self.load_state_dict(state_dict)

    
    def forward(self, x, experts):
        """
        The implementation is inspired by Multi-head Attention from HuggingFace Transformers

        TODO: For LayerNorm part, cannot come up with parallel method
        Linear layers are calculated in parallel way, by expending to one more dimension; bias is ignored.
        """
        ln_x = torch.stack([self.ln[i](x) for i in range(self.nk)], dim=1) # (batch, num_kadapter, seq_len, embed dim)

        # project down
        a = torch.matmul(ln_x, self.we)  # (batch, num_kadapter, seq_len, dneck)
        relu_a = F.relu(a)
        # project up
        m = torch.matmul(relu_a, self.wd)
        output = ln_x + m    # (batch, num_kadapter, seq_len, embed dim)
        
        output_size = output.size()
        size_in = (output_size[0], output_size[1], output_size[2]*output_size[3])
        size_out = (output_size[0], output_size[2], output_size[3])

        weighted_output = torch.bmm(experts.unsqueeze(1), output.view(*size_in)).view(*size_out)
        return weighted_output