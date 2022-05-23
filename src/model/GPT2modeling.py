# Modified from HuggingFace https://github.com/huggingface/transformers/blob/8438bab38e1ea60efca181c92ebc7e4602f91848/src/transformers/modeling_gpt2.py
import logging
import os
import warnings
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import transformers
from transformers import (
    GPT2Config,
    PreTrainedModel,
    GPT2PreTrainedModel,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP, load_tf_weights_in_gpt2
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.file_utils import (
    add_start_docstrings,
)
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from src.model.adapter import Adapter, MultiAdapter

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GPT2Config"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"

def check_load_missing_keys(missing_keys, unexpected_keys, model_path, config, model):
    missing_keys_group = {}
    unexpected_keys_group = {}
    for i in range(config.n_layer):
        missing_keys_group[i] = []
        for key in sorted(missing_keys):
            if key.startswith(f"transformer.h.{str(i)}.kadapter"):
                missing_keys_group[i].append(key)
                missing_keys.remove(key)
        
        unexpected_keys_group[i] = []
        for key in sorted(unexpected_keys):
            if key.startswith(f"transformer.h.{str(i)}.kadapter"):
                unexpected_keys_group[i].append(key)
                unexpected_keys.remove(key)
    
    assert len(missing_keys) == len(unexpected_keys) == 0

    ckpt = torch.load(os.path.join(model_path,"pytorch_model.bin"))

    params = []
    checklist = [("ln", "bias"), ("ln", "weight"), ("we",), ("wd",)]
    for layer_idx, keys in missing_keys_group.items():
        unexpected = unexpected_keys_group[layer_idx]
        
        layer_params = []
        prefix = f"transformer.h.{str(layer_idx)}.kadapter."
        for k in range(config.num_kadapter):
            single_params = {}
            for module in checklist:
                if len(module) == 2:
                    # print(prefix+f"{module[0]}.{k}.{module[1]}", prefix+f"{k}.{module[0]}.{module[1]}")
                    # print("-"*20)
                    if prefix+f"{module[0]}.{k}.{module[1]}" in keys and prefix+f"{k}.{module[0]}.{module[1]}" in unexpected:
                        single_params[f"{module[0]}.{module[1]}"] = ckpt[prefix+f"{k}.{module[0]}.{module[1]}"]
                        keys.remove(prefix+f"{module[0]}.{k}.{module[1]}")
                        unexpected.remove(prefix+f"{k}.{module[0]}.{module[1]}")
                else:
                    # print(prefix+module[0], prefix+f"{k}.{module[0]}.weight")
                    
                    if prefix+module[0] in keys and prefix+f"{k}.{module[0]}.weight" in unexpected:
                        single_params[f"{module[0]}.weight"] = ckpt[prefix+f"{k}.{module[0]}.weight"]
                        if k == config.num_kadapter -1:
                            keys.remove(prefix+module[0])
                        unexpected.remove(prefix+f"{k}.{module[0]}.weight")
            layer_params.append(single_params)
        
        # print("Remainings missing ", keys)
        # print("Remainings unexpect", unexpected)
        # input()
        assert len(keys) == 0
        assert len(unexpected) == 0
        params.append(layer_params)
                        
    model.reset_multiadapter_params_from_params(params)
    return model


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)
        
        self.num_kadapter = config.num_kadapter if hasattr(config, 'num_kadapter') else 1
        self.kadapter_one_hot = config.kadapter_one_hot if hasattr(config, 'kadapter_one_hot') else False
        if config.kadapter:
            if self.num_kadapter == 1:
                self.kadapter = Adapter(config, config.kn_neck) 
            else:
                self.kadapter = nn.ModuleList(Adapter(config, config.kn_neck) for _ in range(config.num_kadapter)) 
                
        else:
            self.kadapter = None
        self.topic_adapter = Adapter(config, config.n_neck) if config.task_adapter else None

    def forward(
        self,
        hidden_states,
        experts=None,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        # residual connection
        hidden_states = attn_output + hidden_states

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attn_outputs = self.crossattention(
                self.ln_cross_attn(hidden_states),
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = hidden_states + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states)) 

        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states

        # TODO debug here
        if self.kadapter is not None:
            if self.num_kadapter == 1:
                hidden_states = self.kadapter(hidden_states)
            else:
                assert experts is not None
                hidden_states_list = torch.stack([self.kadapter[l](hidden_states) for l in range(self.num_kadapter)], dim=1)
                hsl_0, hsl_1, hsl_2, hsl_3 = hidden_states_list.shape
                hidden_states = torch.bmm(experts.unsqueeze(1), hidden_states_list.reshape(hsl_0, hsl_1, hsl_2*hsl_3)).reshape(hsl_0, hsl_2, hsl_3)

        if self.topic_adapter is not None:
            hidden_states = self.topic_adapter(hidden_states)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)



class DualGPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)
        
        self.num_kadapter = config.num_kadapter if hasattr(config, 'num_kadapter') else 1
        self.kadapter_one_hot = config.kadapter_one_hot if hasattr(config, 'kadapter_one_hot') else False
        if config.kadapter:
            if self.num_kadapter == 1:
                self.pre_kadapter = Adapter(config, config.kn_neck) 
                self.post_kadapter = Adapter(config, config.kn_neck) 
            else:
                self.pre_kadapter = nn.ModuleList(Adapter(config, config.kn_neck) for _ in range(config.num_kadapter)) 
                self.post_kadapter = nn.ModuleList(Adapter(config, config.kn_neck) for _ in range(config.num_kadapter)) 
        else:
            self.pre_kadapter = None
            self.post_kadapter = None

        self.topic_adapter = Adapter(config, config.n_neck) if config.task_adapter else None

    def forward(
        self,
        hidden_states,
        experts=None,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        if self.pre_kadapter is not None:
            if self.num_kadapter == 1:
                attn_output = self.pre_kadapter(attn_output)
            else:
                assert experts is not None
                hidden_states_list = torch.stack([self.pre_kadapter[l](attn_output) for l in range(self.num_kadapter)], dim=1)
                hsl_0, hsl_1, hsl_2, hsl_3 = hidden_states_list.shape
                attn_output = torch.bmm(experts.unsqueeze(1), hidden_states_list.reshape(hsl_0, hsl_1, hsl_2*hsl_3)).reshape(hsl_0, hsl_2, hsl_3)

        # residual connection
        hidden_states = attn_output + hidden_states

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attn_outputs = self.crossattention(
                self.ln_cross_attn(hidden_states),
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = hidden_states + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states)) 

        if self.post_kadapter is not None:
            if self.num_kadapter == 1:
                feed_forward_hidden_states = self.post_kadapter(feed_forward_hidden_states)
            else:
                assert experts is not None
                hidden_states_list = torch.stack([self.post_kadapter[l](feed_forward_hidden_states) for l in range(self.num_kadapter)], dim=1)
                hsl_0, hsl_1, hsl_2, hsl_3 = hidden_states_list.shape
                feed_forward_hidden_states = torch.bmm(experts.unsqueeze(1), hidden_states_list.reshape(hsl_0, hsl_1, hsl_2*hsl_3)).reshape(hsl_0, hsl_2, hsl_3)

        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states

        if self.topic_adapter is not None:
            hidden_states = self.topic_adapter(hidden_states)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
)
class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        if config.dual_kadapter:
            self.h = nn.ModuleList([DualGPT2Block(config) for _ in range(config.n_layer)])
        else:
            self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        experts=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    experts,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    experts=experts,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    """The GPT2 Model transformer with a language modeling head on top
    (linear layer with weights tied to the input embeddings). """,
)
class GPT2LMHeadModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        if not hasattr(config, 'task_adapter'):
            config.task_adapter = False
        if hasattr(config, 'topic_adapter'):
            config.task_adapter = True

        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

        self.model_parallel = False

        assert not (config.lm and config.task_adapter)
        assert config.lm or config.kadapter or config.task_adapter

        if not config.lm:
            for p in self.transformer.parameters():
                p.requires_grad=False
        
        if config.kadapter and config.lm:
            if config.dual_kadapter:
                for l in range(len(self.transformer.h)):
                    for p in [
                        *self.transformer.h[l].pre_kadapter.parameters(),
                        *self.transformer.h[l].post_kadapter.parameters(),
                    ]:
                        p.requires_grad=False
            else:
                for l in range(len(self.transformer.h)):
                    for p in [
                        *self.transformer.h[l].kadapter.parameters(),
                    ]:
                        p.requires_grad=False
        
        # The case that kadapters are trained
        # * gpt2+kadapter for lm_finetuning 
        # The case that kadapters are not trained
        # * gpt2+kadapter for task adaptation
        # * gpt2+kadapter+task_adapter for task adaptation
        if config.kadapter and not config.task_adapter and not config.lm:
            if config.dual_kadapter:
                for l in range(len(self.transformer.h)):
                    for p in [
                        *self.transformer.h[l].pre_kadapter.parameters(),
                        *self.transformer.h[l].post_kadapter.parameters(),
                    ]:
                        p.requires_grad=True
            else:
                for l in range(len(self.transformer.h)):
                    for p in [
                        *self.transformer.h[l].kadapter.parameters(),
                    ]:
                        p.requires_grad=True
        
        if config.task_adapter:
            for l in range(len(self.transformer.h)):
                for p in [
                    *self.transformer.h[l].topic_adapter.parameters(),
                ]:
                    p.requires_grad=True
            for p in [
                *self.transformer.wte.parameters(),
                *self.transformer.wpe.parameters(),
                *self.transformer.ln_f.parameters(),
            ]:
                p.requires_grad=True
    
    def reset_task_adapter_params(self, adapter_path):
        checkpoint = torch.load(adapter_path)
        for l in range(len(self.transformer.h)):
            self.transformer.h[l].topic_adapter.load_state_dict(checkpoint[f"topic_adapter{l}"])

    def _convert_checkpoint_layer(self, checkpoint, l, prefix=""):
        state_dict = {}
        state_dict["ln.weight"] = checkpoint[f"{prefix}adapter{l}"]["0.ln.weight"]
        state_dict["ln.bias"] = checkpoint[f"{prefix}adapter{l}"]["0.ln.bias"]
        state_dict["we.weight"] = checkpoint[f"{prefix}adapter{l}"]["0.we.weight"]
        state_dict["wd.weight"] = checkpoint[f"{prefix}adapter{l}"]["0.wd.weight"]
        return state_dict
    
    def reset_kadapter_params(self, adapter_path, ckp=0):
        if self.config.num_kadapter == 1:
            checkpoint = torch.load(adapter_path)
            for l in range(len(self.transformer.h)):
                try:
                    self.transformer.h[l].kadapter.load_state_dict(checkpoint[f"kadapter{l}"])
                except:
                    state_dict = self._convert_checkpoint_layer(checkpoint, l, prefix="k")
                    self.transformer.h[l].kadapter.load_state_dict(state_dict)
        else:
            for i in range(self.config.num_kadapter):
                checkpoint = torch.load(adapter_path + '/kadapter-' + str(i) + '/checkpoint-' + str(ckp) + '/kadapter.pt')
                for l in range(len(self.transformer.h)):
                    try:
                        self.transformer.h[l].kadapter[i].load_state_dict(checkpoint[f"kadapter{l}"])
                    except:
                        state_dict = _convert_checkpoint_layer(checkpoint, l, prefix="k")
                        self.transformer.h[l].kadapter[i].load_state_dict(state_dict)
    
    
    def reset_dual_kadapter_params(self, adapter_path, ckp=0):
        if self.config.num_kadapter == 1:
            checkpoint = torch.load(adapter_path)
            for l in range(len(self.transformer.h)):
                self.transformer.h[l].pre_kadapter.load_state_dict(checkpoint[f"pre_kadapter{l}"])
                self.transformer.h[l].post_kadapter.load_state_dict(checkpoint[f"post_kadapter{l}"])

        else:
            for i in range(self.config.num_kadapter):
                checkpoint = torch.load(adapter_path + '/kadapter-' + str(i) + '/checkpoint-' + str(ckp) + '/kadapter.pt')
                for l in range(len(self.transformer.h)):
                    self.transformer.h[l].pre_kadapter[i].load_state_dict(checkpoint[f"pre_kadapter{l}"])
                    self.transformer.h[l].post_kadapter[i].load_state_dict(checkpoint[f"post_kadapter{l}"])
    
    def reset_multiadapter_params_from_params(self, params):
        if self.config.num_kadapter == 1:
            for l in range(len(self.transformer.h)):
                # try:
                self.transformer.h[l].kadapter.load_state_dict(params)
        else:
            for l in range(len(self.transformer.h)):
                self.transformer.h[l].kadapter.init_pretraiend_params(params[l])


    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        experts = kwargs.get("experts", None)
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "experts": experts,
        }


    def forward(
        self,
        input_ids=None,
        experts=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            experts=experts,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PretrainedModel.beam_search` or :meth:`~transformers.PretrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )



if __name__ == '__main__':
    from transformers import AutoConfig

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # set config
    config = AutoConfig.from_pretrained('gpt2')
    config.lm = True
    config.kadapter = True 
    config.num_kadapter = 2
    config.n_neck = 256
    # load model
    model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
    # print(model)
    model = model.to(device)