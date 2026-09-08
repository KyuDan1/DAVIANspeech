"""File-independent inference of a completed dense-component research run."""
from pathlib import Path

import numpy as np
import torch

from .common_encoder_probe import complete_windows
from .common_eat_adaptation_inference import CommonEatAdaptationPredictor
from .dense_component_v71 import DenseComponentHead, paired_dense_logits
from .dense_component_data_v71 import source_sha256


class DenseComponentPredictor:
    def __init__(self, path, root, device='cuda', allow_smoke=False):
        path, root = Path(path), Path(root)
        if not (path.parent.parent / 'completed.json').is_file():
            raise ValueError('only completed dense runs can be loaded')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        adapted = checkpoint.get('model_type') == 'dense_component_adapter_v72'
        if checkpoint.get('model_type') not in {'dense_component_v71', 'dense_component_adapter_v72'}:
            raise ValueError('not a dense-component checkpoint')
        if checkpoint.get('smoke', True) and not allow_smoke:
            raise ValueError('smoke is not a full trained candidate')
        self.config, self.variant = checkpoint['config'], checkpoint['variant']
        expected_variants = {'adapted_dense'} if adapted else {'file_only', 'dense_supervised'}
        if self.variant not in expected_variants:
            raise ValueError('unknown dense-head variant')
        parent_path = root / self.config['parent_checkpoint']
        if source_sha256(parent_path) != checkpoint['parent_checkpoint_sha256']:
            raise ValueError('frozen parent checkpoint changed')
        parent = CommonEatAdaptationPredictor(parent_path, root, device)
        self.encoder = parent.encoder.eval().requires_grad_(False)
        if adapted:
            self.encoder.adapters.load_state_dict(checkpoint['adapters'], strict=True)
        self.encoder.name = 'eat_dense_' + self.variant
        self.head = DenseComponentHead(width=self.config['head_width'], pooling=self.config['pooling']).to(device).eval()
        self.head.load_state_dict(checkpoint['state_dict'], strict=True)

    @torch.no_grad()
    def __call__(self, audio):
        windows, lengths, _ = complete_windows(audio)
        outputs, _, _ = paired_dense_logits(self.encoder, {self.variant: self.head}, torch.from_numpy(windows),
            torch.from_numpy(lengths), [len(windows)], self.config['encoder_chunk'], self.config['temperature'])
        result = outputs[self.variant].sigmoid()[0].cpu().numpy().astype(np.float64)
        if not np.isfinite(result).all():
            raise ValueError('nonfinite dense-component prediction')
        return result
