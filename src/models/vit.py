import torch 
from torch import nn

from data.preprocessing import PatchEmbedding
from transformer_encoder import TransformerEncoderBlock

class VisionTransformer(nn.Module):
    
    def __init__(self,
                 img_size = 224,
                 in_channels = 3,
                 patch_size = 16,
                 num_transformer_layers = 12,
                 embedding_dim = 768,
                 mlp_size = 3072,
                 num_heads = 12,
                 attn_dropout = 0,
                 mlp_dropout = 0.1,
                 embedding_dropout = 0.1,
                 num_classes = 37):
        super.__init__()

        assert img_size % patch_size == 0, f"Image size should be divisible by patch size"

        self.num_patches = (img_size * img_size) // patch_size**2
        self.class_embedding = nn.Parameter(data=torch.randn(1,1,embedding_dim),
                                            requires_grad=True)
        self.position_embedding = nn.Parameter(data=torch.randn(1,self.num_patches +1, embedding_dim),
                                               requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_size=mlp_size,
            mlp_dropout=mlp_dropout
        ) for _ in range(num_transformer_layers)])

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

        def forward(self,x):

            batch_size = x.shape[0]
            
            # infer size on its own
            class_token = self.class_embedding.expand(batch_size,-1,-1)

            x = self.patch_embedding(x)

            x = torch.cat((class_token,x),dim=1)

            x = self.position_embedding + x

            x = self.embedding_dropout(x)

            x = self.transformer_encoder(x)

            x = self.classifier(x[:,0])

            return x