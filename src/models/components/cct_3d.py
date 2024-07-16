import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# CCT Models

__all__ = ['cct_2', 'cct_4', 'cct_6', 'cct_7', 'cct_8', 'cct_14', 'cct_16']


def cct_2(*args, **kwargs):
    return _cct(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_4(*args, **kwargs):
    return _cct(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_6(*args, **kwargs):
    return _cct(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_7(*args, **kwargs):
    return _cct(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_8(*args, **kwargs):
    return _cct(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_14(*args, **kwargs):
    return _cct(num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


def cct_16(*args, **kwargs):
    return _cct(num_layers=16, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


def _cct(num_layers, num_heads, mlp_ratio, embedding_dim, kernel_size=3, stride=None, padding=None, *args, **kwargs):
    """
    Factory function to create an instance of a 'CCT' class with the supplied parameters.

    Parameters
    ----------
    num_layers : int
        The number of layers in the network.
    num_heads : int
        The number of heads in the multi-head attention mechanism.
    mlp_ratio : float
        The ratio used for the multi-layer perceptron.
    embedding_dim : int
        The dimension of the embeddings.
    kernel_size : int, optional
        The size of the kernel used in the convolutions, by default 3
    stride : int, optional
        The stride length for the convolutions. If no value is specified, it defaults to either 1 or
        half of kernel_size subtracted by 1, whichever is higher. Default is None
    padding : int, optional
        The amount of padding added to the input. If no value is specified, it defaults to either 1 or
        half of kernel_size. Default is None
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    CCT
        An instance of the CCT class with the supplied parameters.

    Notes
    -----
    This is a factory function to create an instance of CCT. The CCT class itself is not defined in this function.
    """
    stride = default(stride, max(1, (kernel_size // 2) - 1))
    padding = default(padding, max(1, (kernel_size // 2)))

    return CCT(num_layers=num_layers,
               num_heads=num_heads,
               mlp_ratio=mlp_ratio,
               embedding_dim=embedding_dim,
               kernel_size=kernel_size,
               stride=stride,
               padding=padding,
               *args, **kwargs)


# positional

def sinusoidal_embedding(n_channels, dim):
    """
    Generates sinusoidal embeddings with a specific dimension.

    This function generates a 2-D tensor with sinusoidally modulated values.
    Values in even indices (2i) are filled with sin(value), and the values in
    odd indices (2i+1) are filled with cos(value). This type of embedding is
    used in transformer architectures.

    Parameters
    ----------
    n_channels : int
        The number of channels for generating the sinusoidal embeddings.
    dim : int
        The dimension of the embedding.

    Returns
    -------
    torch.Tensor
        A tensor that is rearranged in a particular layout (1,...).

    Notes
    -----
    The rearrangement '... -> 1 ...' means that the tensor's dimensions are changed such that
    the first dimension becomes 1 (i.e., turning the tensor into a 2-D tensor of size (1, n_channels)).
    """
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])

    return rearrange(pe, '... -> 1 ...')


# modules

class Attention(nn.Module):
    """
        A PyTorch module implementing the attention mechanism in transformer models.

        The attention mechanism is a core component of transformer models. It
        carries out a three-part linear transformation (query, key, value) of the
        input data, applies a scaling factor, computes a softmax function over the
        result, and ultimately computes a dot product over the value and the
        softmax result followed by a projection.

        Parameters
        ----------
        dim : int
            The dimension of the input tensor.
        num_heads : int, optional
            The number of attention heads. (default is 8)
        attention_dropout : float, optional
            Dropout rate for the attention layer. (default is 0.1)
        projection_dropout : float, optional
            Dropout rate after projection. (default is 0.1)

        Attributes
        ----------
        heads : int
            The number of attention heads.
        scale : float
            The scaling factor for the queries.
        qkv : torch.nn.Linear
            A linear layer used to produce queries, keys, and values from input tensor.
        attn_drop : torch.nn.Dropout
            The dropout layer for attention scores.
        proj : torch.nn.Linear
            The final linear projection layer.
        proj_drop : torch.nn.Dropout
            The dropout layer after the final projection.
        """
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        """
        Initialize attention module.
        """
        super().__init__()
        self.heads = num_heads
        head_dim = dim // self.heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        """
        Run forward pass of the attention mechanism.

        The input tensor is first transformed to queries, keys, and values
        using linear transformation. Then using these, a scaled dot product
        attention is computed.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after attention and projection.

        """
        B, N, C = x.shape

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        q = q * self.scale

        attn = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        return self.proj_drop(self.proj(x))


class TransformerEncoderLayer(nn.Module):
    """
    A Transformer Encoder layer module based on the PyTorch's nn.Module.

    Consists of a self-attention mechanism followed by position-wise fully connected feed-forward network,
    and dropout layers.

    Parameters
    ----------
    d_model : int
        The number of expected features in the input.
    nhead : int
        The number of heads in the multi-head attention mechanism.
    dim_feedforward : int
        The dimension of the feedforward network model, default is 2048.
    dropout : float
        The dropout value, default is 0.1.
    attention_dropout : float
        The dropout value for the Attention layer, default is 0.1.
    drop_path_rate : float
        The rate of path drop, default is 0.1.

    Attributes
    ----------
    pre_norm : nn.LayerNorm
        Layer normalization.
    self_attn : Attention
        Multi-head attention mechanism.
    linear1 : nn.Linear
        Positionwise feed-forward network.
    dropout1 : nn.Dropout
        Dropout for linear1.
    norm1 : nn.LayerNorm
        Layer normalization.
    linear2 : nn.Linear
        Positionwise feed-forward network.
    dropout2 : nn.Dropout
        Dropout for linear2.
    drop_path : DropPath
        Droppath layer.
    activation : activation function
        Activation function used in positionwise feed-forward network, default is GELU.

    Methods
    -------
    forward(src)
        Pass the input through the encoder layer.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        """
        Construct all layers needed for TransformerEncoderLayer.
        """
        super().__init__()

        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate)

        self.activation = F.gelu

    def forward(self, src, *args, **kwargs):
        """
        Pass the input through the encoder layer transformations.

        Parameters
        ----------
        src : torch.Tensor
            The sequence to the encoder layer as a tensor.

        Returns
        -------
        torch.Tensor
            Output tensor of the TransformerEncoderLayer.
        """
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


class DropPath(nn.Module):
    """
    A PyTorch module implementing DropPath regularization.

    DropPath is a regularization method for convolutional neural networks.
    It works by stochastically dropping out entire paths (defined tensor channels)
    in the network's architecture during training, so each individual path
    in the network is forced to optionally learn to correctly classify inputs.

    Parameters
    ----------
    drop_prob : float, optional
        Probability of an element to be zeroed.

    Attributes
    ----------
    drop_prob : float
        Probability of an element to be zeroed.

    Methods
    -------
    forward(x)
        Pass the input through the DropPath layer.
    """
    def __init__(self, drop_prob=None):
        """
        Constructs all the necessary attributes for the DropPath object.

        Parameters
        ----------
        drop_prob : float, optional
            Probability of an element to be zeroed. Default is None.
        """
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Computes the dropout on the input tensor, which sets some elements
        to zero with probability equal to `drop_prob` during training.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Tensor with same shape of input, with some elements set to zero
            based on `drop_prob`.
        """
        batch, drop_prob, device, dtype = x.shape[0], self.drop_prob, x.device, x.dtype

        if drop_prob <= 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (batch, *((1,) * (x.ndim - 1)))

        keep_mask = torch.zeros(shape, device=device).float().uniform_(0, 1) < keep_prob
        output = x.div(keep_prob) * keep_mask.float()
        return output


class Tokenizer(nn.Module):
    """
    A PyTorch module for tokenizing input in transformer models.

    This class is responsible for the tokenization part of visual transformers.
    It contains sequential convolutional neural network layers designed to digest
    the input into tokens which will be consumed by the transformer encoder cells.

    Parameters
    ----------
    frame_kernel_size : int
        Kernel size for the frame dimension.
    kernel_size : int
        Kernel size for the height and width dimensions.
    stride : int
        Stride for the convolution operation for the height and width dimensions.
    padding : int
        Padding for the convolution operation for the height and width dimensions.
    frame_stride : int, optional
        Stride for the frame dimension, default is 1.
    frame_pooling_stride : int, optional
        Stride for frame pooling, default is 1.
    frame_pooling_kernel_size : int, optional
        Kernel size for frame pooling, default is 1.
    pooling_kernel_size : int, optional
        Kernel size for pooling operation, default is 3.
    pooling_stride : int, optional
        Stride size for pooling operation, default is 2.
    pooling_padding : int, optional
        Padding for pooling operation, default is 1.
    n_conv_layers : int, optional
        Number of convolutional layers, default is 1.
    n_input_channels : int, optional
        Number of input channels, default is 3.
    n_output_channels : int, optional
        Number of output channels, default is 64.
    in_planes : int, optional
        Number of planes, default is 64
    activation : str, optional
        activation function, default is None
    max_pool: bool, optional
        to use a max pool layer. default is True
    conv_bias : bool, optional
        to use a bias vector, default is False
    """
    def __init__(
            self,
            frame_kernel_size,
            kernel_size,
            stride,
            padding,
            frame_stride=1,
            frame_pooling_stride=1,
            frame_pooling_kernel_size=1,
            pooling_kernel_size=3,
            pooling_stride=2,
            pooling_padding=1,
            n_conv_layers=1,
            n_input_channels=3,
            n_output_channels=64,
            in_planes=64,
            activation=None,
            max_pool=True,
            conv_bias=False
    ):
        super().__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        n_filter_list_pairs = zip(n_filter_list[:-1], n_filter_list[1:])

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv3d(chan_in, chan_out,
                          kernel_size=(frame_kernel_size, kernel_size, kernel_size),
                          stride=(frame_stride, stride, stride),
                          padding=(frame_kernel_size // 2, padding, padding), bias=conv_bias),
                nn.Identity() if not exists(activation) else activation(),
                nn.MaxPool3d(kernel_size=(frame_pooling_kernel_size, pooling_kernel_size, pooling_kernel_size),
                             stride=(frame_pooling_stride, pooling_stride, pooling_stride),
                             padding=(frame_pooling_kernel_size // 2, pooling_padding,
                                      pooling_padding)) if max_pool else nn.Identity()
            )
                for chan_in, chan_out in n_filter_list_pairs
            ])

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, frames=8, height=224, width=224):
        """
        Determine the length of the sequence that the model will output
        based on the given input dimensions.

        Parameters
        ----------
        n_channels : int, optional
            The number of channels in the input, default is 3.
        frames : int, optional
            The number of frames in the input, default is 8.
        height : int, optional
            The height of the input, default is 224.
        width : int, optional
            The width of the input, default is 224.

        Returns
        -------
        int
            The length of the sequence that the model will output.
        """
        return self.forward(torch.zeros((1, n_channels, frames, height, width))).shape[1]

    def forward(self, x):
        """
        Perform the forward pass for the Tokenizer model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The rearranged tensor after applying the forward pass through
            the convolutional layers.
        """
        x = self.conv_layers(x)
        return rearrange(x, 'b c f h w -> b (f h w) c')

    @staticmethod
    def init_weight(m):
        """
        Initialize the weights of the 3D convolution layers in the Tokenizer model.

        If the input module is a Conv3D layer, its weights are initialized using
        Kaiming Normal initialization.

        Parameters
        ----------
        m : torch.nn.Module
            The module whose weights are to be initialized.
        """
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)


class TransformerClassifier(nn.Module):
    """
    Transformer classifier model, a PyTorch model that classifies input by using
    transformer encoder layers.

    This model initializes transformer blocks with multi-head self-attention
    layers, followed by a normalized fully connected (FC) layer for classification.
    The number of transformer blocks is defined by the `num_layers` parameter.

    Parameters
    ----------
    seq_pool : bool, optional
        If True, use global sequence pooling. Default is True.
    embedding_dim : int, optional
        Dimension of the embedding. Default is 768.
    num_layers : int, optional
        Number of transformer blocks. Default is 12.
    num_heads : int, optional
        Number of attention heads. Default is 12.
    mlp_ratio : float, optional
        Determines the hidden dimension size of the MLP 'feed-forward' layer
        within each transformer block. Default is 4.
    num_classes : int, optional
        Number of classes to classify. Default is 1000.
    dropout_rate : float, optional
        Rate of dropout used at various points in the model. Default is 0.1.
    attention_dropout : float, optional
        Dropout rate in attention layers. Default is 0.1.
    stochastic_depth_rate : float, optional
        Stochastic depth rate. Default is 0.1.
    positional_embedding : str, optional
        Type of positional embedding. Options are 'sine', 'learnable', 'none'. Default is sine
    sequence_length : int, optional
        Sequence length of the input sequence. Default is None
    """
    def __init__(
            self,
            seq_pool=True,
            embedding_dim=768,
            num_layers=12,
            num_heads=12,
            mlp_ratio=4.0,
            num_classes=1000,
            dropout_rate=0.1,
            attention_dropout=0.1,
            stochastic_depth_rate=0.1,
            positional_embedding='sine',
            sequence_length=None,
            *args, **kwargs
    ):
        super().__init__()
        assert positional_embedding in {'sine', 'learnable', 'none'}

        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert exists(sequence_length) or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        if positional_embedding == 'none':
            self.positional_emb = None
        elif positional_embedding == 'learnable':
            self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim))
            nn.init.trunc_normal_(self.positional_emb, std=0.2)
        else:
            self.register_buffer('positional_emb', sinusoidal_embedding(sequence_length, embedding_dim))

        self.dropout = nn.Dropout(p=dropout_rate)

        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=layer_dpr)
            for layer_dpr in dpr])

        self.norm = nn.LayerNorm(embedding_dim)

        self.fc = nn.Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        """
            Initialize the weights of the linear and LayerNorm layers in the TransformerClassifier model.

            If the input module is a Linear layer, its weights are initialized with trunc normal
            and its bias (if exists) is initialized with constant zero.
            If the input module is a LayerNorm, its bias and weights are initialized with constant zero
            and one respectively.


            Parameters
            ----------
            m : torch.nn.Module
                The module (a layer of the model) to be weight-initialized.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and exists(m.bias):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        The forward process includes padding, adding positional embeddings,
        processing through transformer blocks, and finally, applying the pooling operation
        before passing through the fully connected layer. The padding, positional embedding,
        and pooling implementation depends on the initialization parameters of the
        TransformerClassifier instance.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor from the fully connected layer. This output tensor is
            the raw output from the last model's layer, which can be taken as the
            output probabilities for each class after a softmax operation.
        """
        b = x.shape[0]

        if not exists(self.positional_emb) and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = repeat(self.class_emb, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_token, x), dim=1)

        if exists(self.positional_emb):
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.seq_pool:
            attn_weights = rearrange(self.attention_pool(x), 'b n 1 -> b n')
            x = einsum('b n, b n d -> b d', attn_weights.softmax(dim=1), x)
        else:
            x = x[:, 0]

        return self.fc(x)


# CCT Main model

class CCT(nn.Module):
    """
    Compact Convolutional Transformer (CCT) model, a PyTorch model used for image regression,
    which leverages a combination of tokenization through 3D convolution and regression with transformer.

    During initialization, it sets up a tokenizer and a transformer classifier.

    Parameters
    ----------
    img_size : int, optional
        The size (height and width) of the input images. Default is 224.
    num_frames : int, optional
        The number of frames for 3D convolution. Default is 8.
    embedding_dim : int, optional
        Dimensionality of the token embeddings in the transformer. Default is 768.
    n_input_channels : int, optional
        Number of input channels for the 3D ConvNet tokenizer. Default is 3.
    n_conv_layers : int, optional
        Number of convolutional layers in the tokenizer. Default is 1.
    frame_stride : int, optional
        The stride of the frames in the convolutional layer of the tokenizer. Default is 1.
    frame_kernel_size : int, optional
        The kernel size of the frames in the convolutional layer of the tokenizer. Default is 3.
    frame_pooling_kernel_size : int, optional
        The kernel size of the frames in the pooling layer of the tokenizer. Default is 1.
    frame_pooling_stride : int, optional
        The stride of the frames in the pooling layer of the tokenizer. Default is 1.
    kernel_size : int, optional
        The kernel size in the convolutional layer of the tokenizer. Default is 7.
    stride : int, optional
        The stride in the convolutional layer of the tokenizer. Default is 2.
    padding : int, optional
        The padding in the convolutional layer of the tokenizer. Default is 3.
    pooling_kernel_size : int, optional
        The kernel size in the pooling layer of the tokenizer. Default is 3.
    pooling_stride : int, optional
        The stride in the pooling layer of the tokenizer. Default is 2.
    pooling_padding : int, optional
        The padding in the pooling layer of the tokenizer. Default is 1.
    args, kwargs : arguments
        Variable length argument and keyword argument list to be passed to the TransformerClassifier model.
    """
    def __init__(
            self,
            img_size=224,
            num_frames=8,
            embedding_dim=768,
            n_input_channels=3,
            n_conv_layers=1,
            frame_stride=1,
            frame_kernel_size=3,
            frame_pooling_kernel_size=1,
            frame_pooling_stride=1,
            kernel_size=7,
            stride=2,
            padding=3,
            pooling_kernel_size=3,
            pooling_stride=2,
            pooling_padding=1,
            *args, **kwargs
    ):
        super().__init__()
        img_height, img_width = pair(img_size)

        self.tokenizer = Tokenizer(
            n_input_channels=n_input_channels,
            n_output_channels=embedding_dim,
            frame_stride=frame_stride,
            frame_kernel_size=frame_kernel_size,
            frame_pooling_stride=frame_pooling_stride,
            frame_pooling_kernel_size=frame_pooling_kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_padding=pooling_padding,
            max_pool=True,
            activation=nn.ReLU,
            n_conv_layers=n_conv_layers,
            conv_bias=False
        )

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(
                n_channels=n_input_channels,
                frames=num_frames,
                height=img_height,
                width=img_width
            ),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout_rate=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs
        )

    def forward(self, x):
        """
        Defines the computation performed at every call.

        In this method, the input gets passed through the tokenizer
        and then through the classifier.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The transformed tensor after being passed through the
            tokenizer and the classifier.
        """
        x = self.tokenizer(x)
        return self.classifier(x)
