import torch
from torch import nn

from src.models.model_blocks import MLPBlock, MultiheadSelfAttention

class TransformerEncoderBlock(nn.Module):

    def __init__(self,
                 embedding_dim:int=768, 
                 num_heads:int=12, 
                 mlp_size:int=3072, 
                 mlp_dropout:float=0.1, 
                 attn_dropout:float=0): 
        super().__init__()

        self.msa_block = MultiheadSelfAttention(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)


        self.mlp_block =  MLPBlock(embedding_dim=embedding_dim,
                                   mlp_size=mlp_size,
                                   dropout=mlp_dropout)


    def forward(self, x):

        # Residual Connections
        x =  self.msa_block(x) + x
        x = self.mlp_block(x) + x

        return x