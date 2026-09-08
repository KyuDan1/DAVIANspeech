"""Executable file-local v73 composition of frozen-selected research models."""
import hashlib
import json
from pathlib import Path

from .common_encoder_probe_inference import CommonProbePredictor
from .component_composition_v73 import compose_components


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class ComponentCompositionPredictor:
    def __init__(self, frozen_path, root, variant='equal_voice_pair_eat_music', device='cuda'):
        self.root = Path(root)
        self.frozen = json.loads(Path(frozen_path).read_text())
        config = self.frozen['config']
        if (config['schema'] != 'component_composition_v73'
                or variant not in {'xlsr_voice_eat_music', 'equal_voice_pair_eat_music'}
                or config['voice_pair_weights'] != [.5, .5]
                or config['file_rule'] != 'noisy_or_of_presence_weighted_components'):
            raise ValueError('unsupported frozen composition')
        self.variant, self.models, self.checkpoints = variant, {}, {}
        names = ['xlsr', 'eat'] + (['spear'] if variant == 'equal_voice_pair_eat_music' else [])
        for name in names:
            report_path = self.root / Path(config['predictions'][name]).parent / 'report.json'
            if checksum(report_path) != self.frozen['sources_sha256'][str(report_path)]:
                raise ValueError('frozen component attestation changed')
            report = json.loads(report_path.read_text())
            candidates = [(path, digest) for path, digest in report['artifacts_sha256'].items() if Path(path).name == 'head.pt']
            if len(candidates) != 1:
                raise ValueError('ambiguous selected component checkpoint')
            relative, expected = candidates[0]
            path = self.root / relative
            if checksum(path) != expected:
                raise ValueError('selected component checkpoint changed')
            self.checkpoints[name] = {'path': str(path), 'sha256': expected}
            self.models[name] = CommonProbePredictor(path, self.root, device=device)

    def __call__(self, audio):
        # Three models may coexist on the same GPU, but all consume only this
        # one waveform. No prior/neighbor file state is passed to composition.
        values = {name: model(audio)[None] for name, model in self.models.items()}
        return compose_components(values['xlsr'], values['eat'], values.get('spear'))[0]
