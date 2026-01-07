# Practice demo from LoRA From Scratch: Finetuning Large Models for Cheaper https://www.youtube.com/watch?v=nhRYyXGkjSU&t=8521s

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Literal, Union
from safetensors.torch import save_file

class LoRALayerBase:

    def __init__(self,
                 rank=8,
                 lora_alpha=8,
                 lora_dropout=0.0,
                 use_rslora=True):
        
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else lambda x: x
        self.use_rslora = use_rslora

        self.scaling = self.lora_alpha / self.rank**0.5 if use_rslora else self.lora_alpha / self.rank

    
    def _load_pretrained_weights(self, state_dict):

        self.weight.data = state_dict["weight"]
        if "bias" in state_dict.keys():
            self.bias.data = state_dict["bias"]

    

class LoRALinear(nn.Linear, LoRALayerBase):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 rank=8,
                 lora_alpha=8,
                 lora_dropout=0.0,
                 use_rslora=True,
                 **kwargs):
        
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)
        LoRALayerBase.__init__(self, 
                               rank=rank, 
                               lora_alpha=lora_alpha,
                               lora_dropout=lora_dropout,
                               use_rslora=use_rslora)
        
        self.weight.requires_grad = False # freeze the base weight

        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))  # 
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # initialize lora_A as kaiming_uniform
        # nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
    def _merge_weights(self):
        
        merge_weights = self.weight.data + self.scaling * (self.lora_A @ self.lora_B).T 

        state_dict = {"weight": merge_weights}
        if self.bias is not None:
            state_dict["bias"] = self.bias

        merged_linear = nn.Linear(self.in_features,
                                  self.out_features,
                                  bias=True if self.bias is not None else False)
        merged_linear.load_state_dict(state_dict)

        return merged_linear

    def forward(self, x):

        orig_layer_out = F.linear(x, self.weight, bias= self.bias)
        
        lora_mult = (self.lora_A @ self.lora_B) * self.scaling
        low_rank_out = self.lora_dropout(x) @ lora_mult
        # print(lora_mult.shape)
        # print(self.weight.shape)
        # print(low_rank_out.shape)
        # print(orig_layer_out.shape)
        output = orig_layer_out + low_rank_out

        return output

        # print(self.weight)
        # print(self.bias)

class LoRAEmbedding(nn.Embedding, LoRALayerBase):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 rank=8,
                 lora_alpha=8,
                 lora_dropout=0.0,
                 use_rslora=True,
                 **kwargs):
        ### initialize Inherited Classes ###
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayerBase.__init__(self,
                               rank=rank,
                               lora_alpha=lora_alpha,
                               lora_dropout=lora_dropout,
                               use_rslora=use_rslora)
        
        self.weight.requires_grad = False

        # print(self.weight.shape)

        self.lora_A = nn.Parameter(torch.zeros(num_embeddings, rank))   
        # print(self.lora_A.shape)
        self.lora_B = nn.Parameter(torch.zeros(rank, embedding_dim))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) 
    
    def _merge_weights(self):

        merged_weights = self.weight.data + self.scaling * (self.lora_A @ self.lora_B)

        state_dict = {"weight": merged_weights}

        merged_emb = nn.Embedding(self.num_embeddings, self.embedding_dim)
        merged_emb.load_state_dict(state_dict)

        return merged_emb
    
    def forward(self, x):

        orig_layer_out = F.embedding(input=x,
                                     weight=self.weight,
                                     padding_idx=self.padding_idx,
                                     max_norm=self.max_norm,
                                     norm_type=self.norm_type,
                                     scale_grad_by_freq=self.scale_grad_by_freq,
                                     sparse=self.sparse)
        
        low_rank_A_output = F.embedding(input=x,
                                        weight=self.lora_A,
                                        padding_idx=self.padding_idx,
                                        max_norm=self.max_norm,
                                        norm_type=self.norm_type,
                                        scale_grad_by_freq=self.scale_grad_by_freq,
                                        sparse=self.sparse)
        
        low_rank_output = (low_rank_A_output @ self.lora_B) * self.scaling

        output = orig_layer_out + low_rank_output

        return output
    

class LoRAConv2d(nn.Conv2d, LoRALayerBase):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 rank=8, 
                 lora_alpha=8,
                 lora_dropout=0.0,
                 use_rslora=True,
                 **kwargs):
        
        ### Initialize Inherited Classes ###
        nn.Conv2d.__init__(self,
                          in_channels= in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          bias=bias,
                          **kwargs)
        LoRALayerBase.__init__(self,
                               rank=rank,
                               lora_alpha=lora_alpha,
                               lora_dropout=lora_dropout,
                               use_rslora=use_rslora)
        
        # print(self.weight.shape)

        self.weight.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_channels, *self.kernel_size))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_channels))
        # self.lora_B = nn.Parameter(torch.zeros(self.out_channels, rank, 1, 1))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) 
    
    def _merge_weights(self):

        lora_A_flatten = self.lora_A.flatten(1)
        lora_mult = self.lora_B.T @ lora_A_flatten * self.scaling
        lora_mult = lora_mult.reshape(self.out_channels, self.in_channels, *self.kernel_size)

        merged_weights = self.weight.data + lora_mult

        state_dict = {"weight": merged_weights}
        if self.bias is not None:
            state_dict["bias"] = self.bias

        merged_conv = nn.Conv2d(self.in_channels,
                                self.out_channels,
                                kernel_size = self.kernel_size,
                                stride=self.stride,
                                padding=self.padding,
                                bias=True if self.bias is not None else False)

    def forward(self, x):

        orig_layer_out = F.conv2d(input=x,
                                  weight=self.weight,
                                  bias=self.bias,
                                  stride=self.stride,
                                  padding=self.padding)
        
        lora_rank_A_output = F.conv2d(input=x,
                                      weight=self.lora_A,
                                      bias=None,
                                      stride=self.stride,
                                      padding=self.padding)
        
        lora_rank_A_output = lora_rank_A_output.permute(0, 2, 3, 1)

        low_rank_output = self.lora_dropout(lora_rank_A_output) @ self.lora_B * self.scaling
        low_rank_output = low_rank_output.permute(0, 3, 1, 2)

        output = orig_layer_out + low_rank_output

        return output
        
        
        # print(orig_layer_out.shape)
        # print(self.lora_A.shape)
        # print(self.weight.shape)
        print(lora_rank_A_output.shape)
        

if __name__ == "__main__":

    # layer = LoRALinear(4, 4, rank=2)
    # rand = torch.randn(4, 4)
    # output = layer(rand)
    # print(output)
    # # print(layer)
    # merged = layer._merge_weights()
    # output2 = merged(rand)
    # print(output2)
    
    # layer = LoRAEmbedding(200, 64, rank=2)
    # layer = LoRAConv2d(in_channels=256, out_channels=384, kernel_size=4)
    # rand = torch.randn(4, 256, 64, 64)
    # layer(rand)

    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base")
    print(model)

