import torch
import math
from torch import nn, Tensor
from typing import Type
import torch.nn.init as init
import torch.nn.functional as F
from F2P_Net.models.module_lib import Adapter, PromptGenerator
from F2P_Net.models.Auxiliary_encoder import Auxiliary_encoder_base_Block, Auxiliary_encoder_bottleneck_Block, Auxiliary_encoder


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    


class SAMImageEncodeWrapper(nn.Module):

    def __init__(self, ori_sam, fix: bool = True):
        super(SAMImageEncodeWrapper, self).__init__()
        self.sam_img_encoder = ori_sam.image_encoder
        if fix:
            for name, param in self.sam_img_encoder.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.sam_img_encoder(x)
        return x


class SAMPromptEncodeWrapper(nn.Module):

    def __init__(self, ori_sam, fix: bool = True):
        super(SAMPromptEncodeWrapper, self).__init__()
        self.sam_prompt_encoder = ori_sam.prompt_encoder
        if fix:
            for name, param in self.sam_prompt_encoder.named_parameters():
                param.requires_grad = False

    def forward(self, points=None, boxes=None, masks=None):
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points, boxes, masks)
        return sparse_embeddings, dense_embeddings

    def get_dense_pe(self):
        return self.sam_prompt_encoder.get_dense_pe()




class ViCNet(SAMImageEncodeWrapper):

    def __init__(
            self, ori_sam, hq_token: torch.Tensor, Auxiliary_encoder_num:int
    ):
        super(ViCNet, self).__init__(ori_sam=ori_sam, fix=True)

        self.prompt_generator = PromptGenerator(
            scale_factor=32, prompt_type='highpass',
            embed_dim=self.sam_img_encoder.patch_embed.proj.out_channels,
            tuning_stage=1234, depth=len(self.sam_img_encoder.blocks), input_type='fft', freq_nums=0.25,
            handcrafted_tune=True, embedding_tune=True, adaptor='adaptor',
            img_size=self.sam_img_encoder.img_size,
            patch_size=self.sam_img_encoder.patch_embed.proj.kernel_size[0]
        )

        self.hq_token = hq_token # The 1×256 embedding from the mask decoder

        '''
        A dimensionality reduction network comprising multiple Adapter modules was created using nn.Sequential (hq_token_down_proj).
        Each Adapter module's input dimension is consistent with the last dimension of hq_token, and the corresponding number of Adapter modules is created based on self.prompt_generator.shared_mlp.in_features.
        '''

        self.hq_token_down_proj = nn.Sequential(
            *[Adapter(in_features=hq_token.size(-1), mlp_ratio=0.125, add_last_layer=False)
              for _ in range(self.prompt_generator.shared_mlp.in_features)]
        )
        
        patch_height = self.sam_img_encoder.img_size / self.sam_img_encoder.patch_embed.proj.kernel_size[0]
        patch_width = self.sam_img_encoder.img_size / self.sam_img_encoder.patch_embed.proj.kernel_size[1]


        '''
        self.shared_up_proj is a linear layer that maps the input feature tensor from in_features to out_features.
        The input feature dimension is compressed (due to the 0.125 factor), while the output is adjusted to match the dimensions of the image patches. This may be used to restore the downsampled features to a higher resolution or for specific output structures.
        '''
        self.shared_up_proj = nn.Linear(
            in_features=int(hq_token.size(-1) * 0.125),
            out_features=int(patch_height * patch_width)
        )    
        

        if Auxiliary_encoder_num == 18:
            self.Auxiliary_encoder = Auxiliary_encoder(Auxiliary_encoder_base_Block, [2, 2, 2, 2]) # The structure of ResNet18
            self.channel_list = [64, 128, 256, 512]
        elif Auxiliary_encoder_num == 34:
            self.Auxiliary_encoder = Auxiliary_encoder(Auxiliary_encoder_base_Block, [3, 4, 6, 3]) # The structure of ResNet34
            self.channel_list = [64, 128, 256, 512]
        elif Auxiliary_encoder_num == 50:
            self.Auxiliary_encoder = Auxiliary_encoder(Auxiliary_encoder_bottleneck_Block, [3, 4, 6, 3]) # The structure of ResNet50
            self.channel_list = [256, 512, 1024, 2048]
        elif Auxiliary_encoder_num == 101:
            self.Auxiliary_encoder = Auxiliary_encoder(Auxiliary_encoder_bottleneck_Block, [3, 4, 23, 3]) # The structure of ResNet101
            self.channel_list = [256, 512, 1024, 2048]
        

        vit_feature_dims = 1024 # The feature dimension of ViT is 1024 when using ViT-L, and 768 when using ViT-B.

        # FPN for ResNet multi-scale features
        self.fpn = nn.ModuleList([
            nn.Conv2d(dim, vit_feature_dims, kernel_size=1)
            for dim in self.channel_list 
        ])


         

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x  # The default dimensions of the raw image input x are [B, C, H, W] = [1, 3, 1024, 1024].
        # ResNet feature extraction
        c1, c2, c3, c4 = self.Auxiliary_encoder(x)   
        
        x = self.sam_img_encoder.patch_embed(x) # 1*64*64*1024
       
        embedding_feature = self.prompt_generator.init_embeddings(x)#print(embedding_feature.shape)=[1*4096*32]
        
        handcrafted_feature = self.prompt_generator.init_handcrafted(inp)#print(handcrafted_feature.shape)=[1*32*64*64]

        hq_feature = torch.cat(
            [self.shared_up_proj(down_proj(self.hq_token)).unsqueeze(-1) for down_proj in self.hq_token_down_proj],# From mask Decoder comes 1*256-dimensional Embedding, after each Adapter, the feature dimension becomes 256*0.125=32,
            dim=-1  # After passing through the linear layer, the dimension is restored to the image patch dimension (h*w), and then Unsqueeze(-1) increases the dimension by one.
        )
        # hq_feature.shape=1*4096*32

        prompt = self.prompt_generator.get_prompt(handcrafted_feature, embedding_feature, hq_feature=hq_feature)
        if self.sam_img_encoder.pos_embed is not None:
            x = x + self.sam_img_encoder.pos_embed
        j = 0
        interm_embeddings = []
        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        for i, blk in enumerate(self.sam_img_encoder.blocks):
            x =  x  + prompt[i].reshape(B, H, W, -1)
            H_vit, W_vit = x.shape[1], x.shape[2]
            # Process each layer of features extracted by ResNet (c1, c2, c3, c4) by adjusting the number of channels via FPN.
            fpn_features = []
            for y, resnet_feature in enumerate([c1, c2, c3, c4]):
                    fpn_feature = self.fpn[y](resnet_feature)  # Use the corresponding FPN layer to process the features
                    fpn_features.append(fpn_feature)          # Save the processed features
                   # fpn torch.Size([1, 1024, 256, 256])
            resnet_features =  [F.interpolate(f, size=(H_vit, W_vit), mode='bilinear', align_corners=False) for f in fpn_features]  # Interpolate to ViT resolution

            if i in [5, 11, 17, 23]: # ViT-B: global_attn_indexes = [2, 5, 8, 11];ViT-L: global_attn_indexes = [5, 11, 17, 23]; ViT-H: global_attn_indexes = [7, 15, 23, 31]; Insert ResNet features before these positions
                resnet_features = resnet_features[j].permute(0, 2, 3, 1)  # [B, H, W, C];
                x= x + resnet_features   #  x torch.Size([1, 64, 64, 1024]),resnet_features[j] torch.Size([1, 64, 64, 1024])
                j = j + 1

            x = blk(x)  # The dimensions of x are [B, H, W, C], and blk is a Transformer block with dimensions [1, 64, 64, 1024].
            if blk.window_size == 0: # If it is a global attention block
                interm_embeddings.append(x)

        x = self.sam_img_encoder.neck(x.permute(0, 3, 1, 2)) # Dimension transformation [B, H, W, C] -> [B, C, H, W]
        return x, interm_embeddings
    


