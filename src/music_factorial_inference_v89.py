"""Completed v89 Music-only, file-independent native inference."""
import json
from pathlib import Path
import torch

from .common_eat_adaptation_inference import CommonEatAdaptationPredictor


class MusicFactorialPredictor:
    def __init__(self, run, root, variant, device='cuda'):
        run, root = Path(run), Path(root)
        if variant not in ['factorial_bce', 'factorial_ranked']:
            raise ValueError('unknown variant')
        report = json.loads((run / 'report.json').read_text())
        frozen = json.loads((run / 'frozen.json').read_text())
        if report['status'] != 'complete' or frozen['smoke']:
            raise ValueError('completed full training required')
        state = torch.load(run / f'{variant}.pt', map_location='cpu', weights_only=False)
        if (state['model_type'] != 'music_factorial_v89' or state['variant'] != variant
                or state['smoke'] or state['config'] != frozen['config']):
            raise ValueError('checkpoint metadata mismatch')
        parent_path = root / state['config']['parent_checkpoint']
        if state['parent_checkpoint_sha256'] != frozen['artifacts_sha256'][str(parent_path)]:
            raise ValueError('checkpoint lineage mismatch')
        self.model = CommonEatAdaptationPredictor(parent_path, root, device=device)
        if self.model.config['temperature'] != state['config']['temperature']:
            raise ValueError('aggregation mismatch')
        self.model.config = dict(self.model.config, encoder_chunk=1)
        self.model.encoder.adapters.load_state_dict(state['adapters'], strict=True)
        self.model.head.load_state_dict(state['state_dict'], strict=True)
        self.model.encoder.eval().requires_grad_(False)
        self.model.head.eval().requires_grad_(False)

    def __call__(self, audio):
        return float(self.model(audio)[2])
