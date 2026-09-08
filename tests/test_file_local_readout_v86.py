import unittest
import torch
from src.common_encoder_probe import CommonTokenHead
from src.file_local_readout_v86 import file_statistics,lme


class FileStatisticsTest(unittest.TestCase):
    def test_global_matches_parent_file(self):
        torch.manual_seed(14)
        parent=CommonTokenHead(12,width=8,pooling='mean').eval()
        tokens=torch.randn(2,3,71,12)
        mask=torch.arange(71)[None]<torch.tensor([71,43])[:,None]
        glob,local,valid=file_statistics(parent,tokens,mask)
        result=(glob*parent.output_weight[0]).sum(-1)+parent.output_bias[0]
        torch.testing.assert_close(result,parent(tokens,mask)[:,0],atol=1e-6,rtol=1e-5)
        self.assertEqual(tuple(local.shape),(2,8,16))
        self.assertEqual(valid.sum(-1).tolist(),[8,5])

    def test_padding_values_ignored(self):
        parent=CommonTokenHead(12,width=8,pooling='mean').eval()
        tokens=torch.randn(1,3,71,12)
        mask=torch.arange(71)[None]<43
        before=file_statistics(parent,tokens,mask)
        tokens[:,:,43:]=100000
        after=file_statistics(parent,tokens,mask)
        for a,b in zip(before,after):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_constant_logits_preserved(self):
        for n in [1,5,300]:
            torch.testing.assert_close(lme(torch.full((n,),1.7)),torch.tensor(1.7))


if __name__=='__main__':unittest.main()
