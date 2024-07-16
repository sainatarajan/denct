import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class FeedForward(nn.Module):
    """
    A feed-forward neural network module.

    This class represents a feed-forward network structure using PyTorch's `nn.Module`.
    It contains a sequence of pre-defined layers, including two linear layers, a GELU activation
    layer, and two dropout layers.

    Args:
        dim (int): The dimensionality of the input and output.
        hidden_dim (int): The dimensionality of the hidden layer.
        dropout (float, optional): The dropout rate. Defaults to 0.0.

    Attributes:
        net (nn.Sequential): The actual network, constructed as a sequential model in PyTorch.

    Methods:
        forward(x): Defines the forward pass of the neural network.
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the network.
        """
        return self.net(x)


class Attention(nn.Module):
    """
    An Attention module used in Transformer models.

    This class represents a self-attention mechanism that is commonly found in transformer models.
    It includes a Layer Normalization, a linear transformation to Query, Key, Value (QKV),
    and an output linear transformation. There is also a scaling factor to ensure that gradients
    do not vanish or explode.

    Args:
        dim (int): The number of dimensions.
        heads (int, optional): The number of heads. Defaults to 8.
        dim_head (int, optional): The dimensionality of the head. Defaults to 64.
        dropout (float, optional): The dropout rate. Defaults to 0.

    Attributes:
        heads (int): The number of heads.
        scale (float): The scaling factor.
        norm (nn.LayerNorm): The layer normalization module.
        attend (nn.Softmax): The softmax module that calculates attention scores.
        dropout (nn.Dropout): The dropout module.
        to_qkv (nn.Linear): The linear transformation to QKV.
        to_out (nn.Sequential | nn.Identity): The output linear transformation.

    Methods:
        forward(x): Defines the forward pass of the module.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        Defines the forward pass of the attention mechanism.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of this attention mechanism.
        """
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Transformer model consisting of Attention and FeedForward layers.

    This class represents a basic implementation of a Transformer model. It contains
    multiple layers of Attention and FeedForward units.

    Args:
        dim (int): The dimensionality of the input and output.
        depth (int): The number of layers in the transformer.
        heads (int): The number of Attention heads.
        dim_head (int): The dimensionality of each Attention head.
        mlp_dim (int): The dimensionality of the inner layer in the FeedForward networks.
        dropout (float, optional): The dropout rate. Defaults to 0.0.

    Attributes:
        layers (nn.ModuleList): A list of Attention and FeedForward layers.

    Methods:
        forward(x): Defines the forward pass of the Transformer.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        """
        Defines the forward pass of the Transformer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the Transformer.
        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    """
    A PyTorch implementation of the Vision Transformer (ViT).

    This class provides the implementation of the Vision Transformer (ViT), a transformer-based
    model for visual tasks. It operates on patches of the input images, and has the ability to
    retain the sequence of patches, so it can handle videos as well as images.

    Args:
        image_size (tuple): A tuple specifying the height and width of the input images.
        image_patch_size (tuple): A tuple specifying the height and width of the patches to be extracted from the images.
        frames (int): The total number frames in a video.
        frame_patch_size (int): The size of the patches to be extracted from each frame.
        num_classes (int): The number of classes for the output layer of the model.
        dim (int): The number of dimensions for the model.
        depth (int): The number of layers in the transformer.
        heads (int): The number of heads for the multi-headed attention mechanism.
        mlp_dim (int): The inner dimensionality for the fully connected layer in the transformer.
        pool (str, optional): The type of pooling to apply. Accepted values are: 'cls' for cls token pooling and 'mean' for global average pooling. Defaults to 'cls'.
        channels (int, optional): The number of input channels. Defaults to 3.
        dim_head (int, optional): The dimensionality of each head in the multi-headed attention mechanism. Defaults to 64.
        dropout (float, optional): The overall dropout rate. Defaults to 0.0.
        emb_dropout (float, optional): The dropout rate for the embeddings. Defaults to 0.0.

    Attributes:
        to_patch_embedding (torch.nn.Sequential): The initial layers for creating patch embeddings.
        pos_embedding (torch.nn.Parameter): The positional embedding of the patches.
        cls_token (torch.nn.Parameter): The cls token for the transformer.
        dropout (torch.nn.Dropout): The dropout layer.
        transformer (Transformer): The transformer layers.
        pool (str): The type of pooling to use.
        to_latent (torch.nn.Identity): The identity layer to maintain the dimensionality of the transformer output.
        mlp_head (torch.nn.Sequential): The final layers, including a fully connected layer for the model output.

    Methods:
        forward(img): Defines the forward pass of the network.
    """
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads,
                 mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, ('Image dimensions must be '
                                                                                     'divisible by the patch size.')
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width,
                      pf=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video):
        """
        Defines the forward propagation of the Vision Transformer.

        Args:
            video (torch.Tensor): The input image/video tensor.

        Returns:
            torch.Tensor: The output tensor of the network.
        """
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
