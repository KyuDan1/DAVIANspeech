"""Only identify current EERs from matched, current-pipeline probes."""
import argparse
from decimal import Decimal as D
import json


def analyze(music_probe_ads=None):
    ads=D('0.7538888889')
    voice=D('0.5')-(ads-D('0.689'))/D('0.2')
    remainder=1-ads-D('0.2')*voice
    result=dict(voice_eer=float(voice),file_eer=None,music_eer=None,
                weighted_file_music_error=float(remainder),
                file_music_constraint='0.5*FILE_EER + 0.3*MUSIC_EER = 0.211',
                voice_perfection_score_ceiling=float(D('0.7774307884')+D('0.18')*voice),
                target_ads_at_current_cps=float((D('0.85332')-D('0.1')*D('0.9893078836'))/D('0.9')),
                music_lineage='v18 is NOT a matched current Music baseline')
    if music_probe_ads is not None:
        music=D('0.5')-(ads-D(str(music_probe_ads)))/D('0.3')
        file=(remainder-D('0.3')*music)/D('0.5')
        if not 0<=music<=1 or not 0<=file<=1:
            raise ValueError('Inconsistent probe values or pipeline assumptions')
        result.update(music_eer=float(music),file_eer=float(file),
                      music_lineage='Requires actual current_music_probe_v84 result with other four outputs unchanged')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--music-probe-ads',help='Only actual current v84 probe ADS, never an older pipeline')
    args=parser.parse_args()
    print(json.dumps(analyze(args.music_probe_ads),indent=2))
