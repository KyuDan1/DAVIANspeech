"""Completed v87 Music-only, file-independent native inference."""
import json
from pathlib import Path
import torch

from .common_eat_adaptation_inference import CommonEatAdaptationPredictor


class MusicCounterfactualPredictor:
    def __init__(self, run, root, variant, device='cuda'):
        run, root = Path(run), Path(root)
        if variant not in ['bce', 'invariant']:
            raise ValueError('unknown variant')
        report = json.loads((run / 'report.json').read_text())
        frozen = json.loads((run / 'frozen.json').read_text())
        if report['status'] != 'complete' or frozen['smoke']:
            raise ValueError('completed full training required')
        state = torch.load(run / f'{variant}.pt', map_location='cpu', weights_only=False)
        if (state['model_type'] != 'music_counterfactual_v87' or state['smoke']
                or state['variant'] != variant or state['config'] != frozen['config']):
            raise ValueError('checkpoint metadata mismatch')
        parent = root / state['config']['parent_checkpoint']
        if state['parent_checkpoint_sha256'] != frozen['artifacts_sha256'][str(parent)]:
            raise ValueError('checkpoint lineage mismatch')
        self.model = CommonEatAdaptationPredictor(parent, root, device=device)
        if self.model.config['temperature'] != state['config']['temperature']:
            raise ValueError('aggregation mismatch')
        # Native training uses single-window encoder calls; preserve that shape.
        self.model.config = dict(self.model.config, encoder_chunk=1)
        self.model.encoder.adapters.load_state_dict(state['adapters'], strict=True)
        self.model.head.load_state_dict(state['state_dict'], strict=True)
        self.model.encoder.eval().requires_grad_(False)
        self.model.head.eval().requires_grad_(False)

    def __call__(self, audio):
        # Never expose other four adapted outputs as replacement predictions.
        return float(self.model(audio)[2])
