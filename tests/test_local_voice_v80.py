import unittest
import numpy as np
import torch
from src.common_encoder_probe import CommonTokenHead, complete_windows
from src.local_voice_v80 import (voice_projection, local_statistics, temporal_geometry,
    interval_labels, LocalVoiceHead, file_logit, balanced_interval_loss)


class LocalVoiceTest(unittest.TestCase):
    def test_projection_preserves_parent_global_voice(self):
        torch.manual_seed(8)
        parent=CommonTokenHead(12, width=8, pooling='mean')
        x=torch.randn(2,3,51,12)
        mask=torch.arange(51)[None] < torch.tensor([51,33])[:,None]
        values=voice_projection(parent,x,mask)
        mean=(values*mask[...,None]).sum(1)/mask.sum(1)[:,None]
        var=((values-mean[:,None]).square()*mask[...,None]).sum(1)/mask.sum(1)[:,None]
        y=(torch.cat([mean,(var+1e-5).sqrt()],-1)*parent.output_weight[1]).sum(-1)+parent.output_bias[1]
        torch.testing.assert_close(y,parent(x,mask)[:,1],atol=1e-7,rtol=1e-5)

    def test_padding_is_ignored(self):
        x=torch.randn(2,51,8)
        mask=torch.arange(51)[None] < torch.tensor([51,23])[:,None]
        a,m=local_statistics(x,mask,kernel=11)
        x[~mask]=float('nan')
        b,_=local_statistics(x,mask,kernel=11)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertTrue(torch.isfinite(a).all())
        self.assertEqual(m.shape,a.shape[:2])

    def test_ownership_and_interval(self):
        _,lengths,starts=complete_windows(np.zeros(60*16000,dtype=np.float32))
        t,valid=temporal_geometry(starts,lengths)
        chosen=t[valid]
        self.assertTrue(np.all(np.diff(chosen)>0))
        self.assertLess(np.max(np.diff(chosen)),.41)
        labels,active=interval_labels(t,valid,[[25,27]])
        self.assertGreater(labels[active].sum(),0)
        self.assertTrue(np.all((t[(labels==1)&active]>25.08)&(t[(labels==1)&active]<26.92)))

    def test_paired_initial_and_gradient(self):
        a=LocalVoiceHead(torch.zeros(16),torch.tensor(0.))
        b=LocalVoiceHead(a.weight,a.bias)
        features=torch.randn(2,8,16)
        y=a(features)
        torch.testing.assert_close(y,b(features),rtol=0,atol=0)
        valid=torch.ones(2,8,dtype=torch.bool)
        labels=torch.zeros_like(y); labels[0,3:5]=1
        loss=file_logit(y,valid)+balanced_interval_loss(y,labels,valid)
        loss.backward()
        self.assertTrue(torch.isfinite(a.weight.grad).all())


if __name__=='__main__':
    unittest.main()
