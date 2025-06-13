import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeatureExtractor:
    """
    Extract features from a 3D CNN for specific regions using global average pooling.
    """
    def __init__(self, model_3d):
        """
        Initialize feature extractor for 3D CNN, targeting specific convolution layers.
        
        Args:
            model_3d (nn.Module): 3D DenseNet model.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model_3d.features.to(self.device)  # Only use convolutional features
        self.model.eval()
        self.target_layers = {
            'denseblock4.denselayer16': 'last_conv',
            'denseblock2.denselayer12': 'third_last_conv',
            'denseblock1.denselayer6': 'fifth_last_conv'
        }
        self.feature_maps = {}
        self.hooks = []

        # Register forward hooks for target layers
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(self._save_features(name)))

    def _save_features(self, name):
        """
        Create a hook to save feature maps for a given layer.
        
        Args:
            name (str): Name of the layer.
        
        Returns:
            callable: Hook function to save feature maps.
        """
        def hook(module, input, output):
            self.feature_maps[name] = output.detach()
        return hook

    def extract_features(self, ct_volume, mask_volume):
        """
        Extract features from specified layers for each region using global average pooling.
        
        Args:
            ct_volume (np.ndarray): 3D CT scan volume.
            mask_volume (np.ndarray): 3D segmentation mask (0=background, 1=femur, 2=tibia).
        
        Returns:
            dict: Features for each region and layer.
        """
        # Prepare CT tensor
        ct_tensor = torch.FloatTensor(ct_volume).unsqueeze(0).unsqueeze(0)
        ct_tensor = ct_tensor.repeat(1, 3, 1, 1, 1)  # Repeat to match 3 input channels
        ct_tensor = (ct_tensor - ct_tensor.min()) / (ct_tensor.max() - ct_tensor.min() + 1e-8)
        ct_tensor = ct_tensor.to(self.device)
        mask_tensor = torch.FloatTensor(mask_volume).to(self.device)

        # Forward pass to collect features
        self.feature_maps.clear()
        with torch.no_grad():
            _ = self.model(ct_tensor)

        if not self.feature_maps:
            print("Error: No feature maps captured. Check hook registration.")
            return {}

        # Extract region-wise features
        features = {}
        for region_name, region_label in [('tibia', 2), ('femur', 1), ('background', 0)]:
            region_mask = (mask_tensor == region_label)
            if not region_mask.any():
                print(f"Warning: No region found for {region_name} (label {region_label})")
                continue

            region_features = {}
            for layer_key, feat_map in self.feature_maps.items():
                if layer_key not in self.target_layers:
                    continue
                _, _, D, H, W = feat_map.shape
                resized_mask = F.interpolate(region_mask.unsqueeze(0).unsqueeze(0).float(),
                                            size=(D, H, W), mode='nearest').squeeze()
                masked_feat = feat_map * resized_mask
                pooled = masked_feat.mean(dim=[2, 3, 4])
                region_features[self.target_layers[layer_key]] = pooled.cpu().numpy()
            features[region_name] = region_features

        return features

    def cleanup(self):
        """
        Remove forward hooks to free memory.
        """
        for hook in self.hooks:
            hook.remove()