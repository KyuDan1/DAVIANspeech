"""File-local inference of a frozen-selected paired EAT experiment checkpoint."""
from pathlib import Path

import numpy as np
import torch

from .common_encoder_probe import CommonTokenHead, FrozenEncoderTokens, complete_windows
from .common_eat_adaptation import PairedEatRepresentation, paired_file_logits


class CommonEatAdaptationPredictor:
    def __init__(self, path, root, device='cuda', allow_smoke=False):
        path, root = Path(path), Path(root)
        if not (path.parent.parent / 'completed.json').is_file():
            raise ValueError('only completed paired EAT runs may be loaded')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint.get('model_type') != 'common_eat_adaptation_v1':
            raise ValueError('not a paired EAT checkpoint')
        if checkpoint.get('smoke', True) and not allow_smoke:
            raise ValueError('smoke checkpoint is not a full trained candidate')
        self.config, self.variant = checkpoint['config'], checkpoint['variant']
        if self.variant not in {'control', 'adapted'}:
            raise ValueError('unknown paired EAT variant')
        base = FrozenEncoderTokens('eat_large', root / self.config['encoder_path'], device=device)
        self.encoder = PairedEatRepresentation(base.model, self.config['adapter_blocks'],
                                               self.config['adapter_bottleneck']).to(device).eval()
        self.encoder.name = 'eat_large_' + self.variant
        if self.variant == 'adapted':
            self.encoder.adapters.load_state_dict(checkpoint['adapters'], strict=True)
        elif checkpoint['adapters'] is not None:
            raise ValueError('control checkpoint must not carry trained adapters')
        self.head = CommonTokenHead(checkpoint['input_dimension'], width=self.config['head_width'],
                                    pooling=self.config['pooling']).to(device).eval()
        self.head.load_state_dict(checkpoint['state_dict'], strict=True)

    @torch.no_grad()
    def __call__(self, audio):
        windows, lengths, _ = complete_windows(audio)
        logits = paired_file_logits(self.encoder, {self.variant: self.head}, torch.from_numpy(windows),
                                    torch.from_numpy(lengths), [len(windows)], self.config['encoder_chunk'],
                                    self.config['temperature'])[self.variant]
        result = logits.sigmoid()[0].cpu().numpy().astype(np.float64)
        if not np.isfinite(result).all():
            raise ValueError('nonfinite paired EAT predictions')
        return result
