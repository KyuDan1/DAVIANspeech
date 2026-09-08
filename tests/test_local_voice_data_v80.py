import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.local_voice_data_v80 import LocalVoicePairs


class LocalVoiceDataTest(unittest.TestCase):
    def test_pair_geometry_and_codec_keys_are_matched(self):
        catalog=pd.DataFrame([
            dict(ID='r0',PATH='r0',GROUP_ID='r0',COMPONENT='VOICE',LABEL=0,GENERATOR='real'),
            dict(ID='r1',PATH='r1',GROUP_ID='r1',COMPONENT='VOICE',LABEL=0,GENERATOR='real'),
            dict(ID='f0',PATH='f0',GROUP_ID='f0',COMPONENT='VOICE',LABEL=1,GENERATOR='tts'),
            dict(ID='m0',PATH='m0',GROUP_ID='m0',COMPONENT='MUSIC',LABEL=0,GENERATOR='real')])
        cfg=dict(seed=8,pairs_per_epoch=4,durations=[4,60],insertions=[.5,2],snr=[0],channels=['clean'])
        waves={k:np.sin(np.arange(48000,dtype=np.float32)*frequency)*.1
               for k,frequency in [('r0',.03),('r1',.05),('f0',.07),('m0',.09)]}
        calls=[]
        def codec(audio,channel,ffmpeg,key):
            calls.append((channel,key,len(audio)))
            return audio
        with patch('src.local_voice_data_v80.cached_audio',side_effect=lambda path:(waves[path],path)), \
             patch('src.local_voice_data_v80.apply_channel',side_effect=codec):
            dataset=LocalVoicePairs(catalog,cfg,None)
            for i in range(4):
                pair,trace=dataset[i]
                self.assertEqual(calls[-2],calls[-1])
                self.assertEqual(pair[0][1:],(0,[]))
                self.assertEqual(pair[1][1],1)
                self.assertEqual(len(pair[0][0]),len(pair[1][0]))
                a,b=pair[1][2][0]
                self.assertAlmostEqual(b-a,trace['insertion'])
                self.assertNotEqual(trace['sources']['background']['group'],trace['sources']['real_insert']['group'])
                # Identical deterministic generation if the draw is repeated.
                repeated,_=dataset[i]
                np.testing.assert_array_equal(pair[0][0],repeated[0][0])
                np.testing.assert_array_equal(pair[1][0],repeated[1][0])


if __name__=='__main__':unittest.main()
