import torch
import torch.nn as nn

def inflate_conv2d_to_conv3d(conv2d, time_dim=3):
    """
    Convert a 2D convolution layer to 3D by repeating weights along a depth dimension.
    
    Args:
        conv2d (nn.Conv2d): 2D convolution layer.
        time_dim (int): Depth dimension for 3D kernel (default: 3).
    
    Returns:
        nn.Conv3d: Inflated 3D convolution layer.
    """
    conv3d = nn.Conv3d(
        in_channels=conv2d.in_channels,
        out_channels=conv2d.out_channels,
        kernel_size=(time_dim, *conv2d.kernel_size),
        stride=(1, *conv2d.stride),
        padding=(time_dim // 2, *conv2d.padding),
        bias=conv2d.bias is not None
    )
    with torch.no_grad():
        weight2d = conv2d.weight.data
        weight3d = weight2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1) / time_dim
        conv3d.weight.copy_(weight3d)
        if conv2d.bias is not None:
            conv3d.bias.copy_(conv2d.bias.data)
    return conv3d

def to_2tuple(val):
    """
    Convert a scalar to a 2-tuple if needed.
    
    Args:
        val: Input value (scalar or tuple/list).
    
    Returns:
        tuple: 2-tuple of the input value.
    """
    return val if isinstance(val, (list, tuple)) else (val, val)

def inflate_densenet2d_to_3d(model_2d, time_dim=3):
    """
    Inflate a 2D DenseNet model to 3D by converting convolutional and pooling layers.
    
    Args:
        model_2d (nn.Module): Pretrained 2D DenseNet model.
        time_dim (int): Depth dimension for 3D kernels (default: 3).
    
    Returns:
        nn.Module: Inflated 3D DenseNet model.
    """
    def _inflate(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                setattr(module, name, inflate_conv2d_to_conv3d(child, time_dim))
            elif isinstance(child, nn.BatchNorm2d):
                bn3d = nn.BatchNorm3d(
                    child.num_features, child.eps, child.momentum,
                    child.affine, child.track_running_stats
                )
                bn3d.load_state_dict(child.state_dict())
                setattr(module, name, bn3d)
            elif isinstance(child, nn.MaxPool2d):
                k, s, p, d = map(to_2tuple, [child.kernel_size, child.stride, child.padding, child.dilation])
                pool3d = nn.MaxPool3d((1, *k), (1, *s), (0, *p), (1, *d), ceil_mode=child.ceil_mode)
                setattr(module, name, pool3d)
            elif isinstance(child, nn.AvgPool2d):
                k, s, p = map(to_2tuple, [child.kernel_size, child.stride, child.padding])
                pool3d = nn.AvgPool3d((1, *k), (1, *s), (0, *p), ceil_mode=child.ceil_mode)
                setattr(module, name, pool3d)
            else:
                _inflate(child)
    _inflate(model_2d)
    return model_2d