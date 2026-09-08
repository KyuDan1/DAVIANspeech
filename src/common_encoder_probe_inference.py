"""File-local inference for completed research readouts, not packaged submission."""
from pathlib import Path

import numpy as np
import torch

from .common_encoder_probe import CommonTokenHead, FrozenEncoderTokens, complete_windows
from .full_coverage_wpt import masked_lme


class CommonProbePredictor:
    def __init__(self, checkpoint_path, root, device='cuda', allow_smoke=False):
        checkpoint_path, root = Path(checkpoint_path), Path(root)
        if not (checkpoint_path.parent.parent / 'completed.json').is_file():
            raise ValueError('only completed readout runs may be evaluated independently')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if checkpoint.get('model_type') in {'dense_component_v71', 'dense_component_adapter_v72'}:
            from .dense_component_inference_v71 import DenseComponentPredictor
            self.delegate = DenseComponentPredictor(checkpoint_path, root, device, allow_smoke)
            self.config, self.encoder = self.delegate.config, self.delegate.encoder
            return
        if checkpoint.get('model_type') == 'common_eat_adaptation_v1':
            from .common_eat_adaptation_inference import CommonEatAdaptationPredictor
            self.delegate = CommonEatAdaptationPredictor(checkpoint_path, root, device, allow_smoke)
            self.config, self.encoder = self.delegate.config, self.delegate.encoder
            return
        if checkpoint.get('purpose') != 'frozen-readout benchmark':
            raise ValueError('not a common encoder research checkpoint')
        self.config = checkpoint['config']
        self.encoder = FrozenEncoderTokens(checkpoint['encoder'], root / self.config['encoders'][checkpoint['encoder']], device)
        self.head = CommonTokenHead(checkpoint['input_dimension'], width=self.config['head_width'], pooling=checkpoint['pooling']).to(device)
        self.head.load_state_dict(checkpoint['state_dict'], strict=True)
        self.head.eval()

    @torch.no_grad()
    def __call__(self, audio):
        if hasattr(self, 'delegate'):
            return self.delegate(audio)
        windows, lengths, _ = complete_windows(audio)
        results = []
        size = self.config['encoder_chunk']
        for start in range(0, len(windows), size):
            tokens, mask = self.encoder(torch.from_numpy(windows[start:start + size]), torch.from_numpy(lengths[start:start + size]))
            results.append(self.head(tokens, mask))
        values = torch.cat(results)[None]
        mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        probabilities = masked_lme(values, mask, self.config['temperature']).sigmoid()[0].cpu().numpy().astype(np.float64)
        if not np.isfinite(probabilities).all():
            raise ValueError('nonfinite independent probabilities')
        return probabilities
