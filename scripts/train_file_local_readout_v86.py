"""Two File-only heads: same TRAIN draws and frozen encoder, different pooling."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]


def main():
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from src.common_encoder_probe import CommonTokenHead,FrozenEncoderTokens,complete_windows
    from src.local_voice_v80 import LocalVoiceHead
    from src.local_voice_data_v80 import identity_collate
    from src.file_local_readout_v86 import file_statistics,lme
    from src.dense_component_data_v71 import DenseTrainingBags
    from src.full_coverage_wpt import resolve_ffmpeg
    from src.pipeline import load_audio
    from src.evaluate_diagnostic import official_eer
    from train_dense_component_v71 import audited_catalog
    from train_common_encoder_probe import audit_protected_sources
    from train_paired_wpt_file_v60 import balanced_weights,sha256
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-report',type=Path)
    args=parser.parse_args()
    config_path=ROOT/'configs/file_local_readout_v86.yaml'
    config=yaml.safe_load(config_path.read_text())
    if args.output.exists():raise FileExistsError(args.output)
    inventory_path=ROOT/config['inventory']
    inventory=json.loads(inventory_path.read_text())
    assert inventory['status']=='complete_metadata_cache'
    for path,digest in inventory['artifacts_sha256'].items():
        assert sha256(path)==digest
    frames={}
    for name,record in inventory['tables'].items():
        assert sha256(record['path'])==record['sha256']
        frames[name]=pd.read_csv(record['path'],low_memory=False)
    train,dev=frames['train'],frames['development']
    audit=audit_protected_sources(train,ROOT/config['partition_config'])
    catalog=audited_catalog(config,train)
    ffmpeg=resolve_ffmpeg()
    paths=[Path(__file__),config_path,inventory_path,ROOT/config['partition_config'],ROOT/config['source_catalog'],
           ROOT/config['encoder']/'model.safetensors',ROOT/config['parent_head'],ffmpeg,
           *[ROOT/'src'/n for n in ['file_local_readout_v86.py','local_voice_v80.py','common_encoder_probe.py',
                                  'dense_component_data_v71.py','long_component_stress.py','telephone_channel.py','xlsr_antideepfake.py']]]
    hashes={str(p):sha256(p) for p in paths}
    if not args.smoke:
        if args.smoke_report is None:raise ValueError('completed matching smoke required')
        previous=json.loads(args.smoke_report.read_text())
        frozen=json.loads((args.smoke_report.parent/'frozen.json').read_text())
        assert previous['status']=='complete_smoke' and frozen['config']==config and frozen['artifacts_sha256']==hashes
    args.output.mkdir(parents=True)
    frozen=dict(config=config,artifacts_sha256=hashes,source_audit=audit,smoke=args.smoke,
                training_role='TRAIN only',protected_evaluation_consumed=False)
    (args.output/'frozen.json').write_text(json.dumps(frozen,indent=2))
    print(json.dumps(dict(stage='audited',train=len(train),dev=len(dev),catalog=len(catalog))),flush=True)
    torch.set_num_threads(2);torch.manual_seed(config['seed'])
    encoder=FrozenEncoderTokens('xlsr',ROOT/config['encoder'])
    parent=CommonTokenHead(1920,width=96,pooling='mean').cuda().eval()
    parent.load_state_dict(torch.load(ROOT/config['parent_head'],map_location='cpu',weights_only=False)['state_dict'])
    parent.requires_grad_(False)
    heads={name:LocalVoiceHead(parent.output_weight[0],parent.output_bias[0]).cuda() for name in ['global','local']}
    opts={name:torch.optim.AdamW(head.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay']) for name,head in heads.items()}
    @torch.no_grad()
    def features(windows,lengths):
        global_parts=[];local_parts=[];baseline=[]
        for window,length in zip(windows,lengths):
            tokens,mask=encoder(torch.from_numpy(window[None]),torch.tensor([length]))
            glob,local,valid=file_statistics(parent,tokens,mask)
            original=parent(tokens,mask)[:,0]
            reconstructed=(glob*parent.output_weight[0]).sum(-1)+parent.output_bias[0]
            torch.testing.assert_close(original,reconstructed,atol=2e-6,rtol=1e-5)
            global_parts.append(glob);local_parts.append(local[valid]);baseline.append(original)
        return {'global':torch.cat(global_parts),'local':torch.cat(local_parts)},lme(torch.cat(baseline),2.)
    started=time.monotonic();history=[]
    draws=4 if args.smoke else config['samples_per_epoch']
    for epoch in range(1,(1 if args.smoke else config['epochs'])+1):
        data=DenseTrainingBags(train,catalog,config,balanced_weights(train),ffmpeg,draws,epoch)
        kwargs=dict(batch_size=None,collate_fn=identity_collate,num_workers=0 if args.smoke else config['workers'])
        if kwargs['num_workers']:kwargs['multiprocessing_context']='spawn'
        losses={name:[] for name in heads}
        with (args.output/f'epoch_{epoch}_traces.jsonl').open('w') as stream:
            for index,(windows,lengths,target,_,_,trace) in enumerate(DataLoader(data,**kwargs)):
                if not np.isfinite(target[0]) or target[0] not in [0,1]:raise ValueError('invalid File label')
                values,_=features(windows,lengths)
                for name,head in heads.items():
                    logits=lme(head(values[name]),config['temperature'])
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,logits.new_tensor(float(target[0])))
                    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                    opts[name].zero_grad();loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True)
                    opts[name].step();losses[name].append(float(loss.detach()))
                assert all(p.grad is None for p in parent.parameters()) and all(p.grad is None for p in encoder.model.parameters())
                stream.write(json.dumps(trace)+'\n');stream.flush()
                if index%100==0:print(json.dumps(dict(epoch=epoch,draw=index,seconds=time.monotonic()-started)),flush=True)
        history.append(dict(epoch=epoch,loss={k:float(np.mean(v)) for k,v in losses.items()}))
    for name,head in heads.items():
        assert not torch.equal(head.weight,parent.output_weight[0])
        torch.save(dict(state_dict=head.state_dict(),config=config,variant=name,smoke=args.smoke,scope='File only'),args.output/f'{name}.pt')
    results=[]
    if not args.smoke:
        scores={name:[] for name in ['parent','global','local']}
        for index,row in dev.iterrows():
            windows,lengths,_=complete_windows(load_audio(Path(row.PATH)))
            values,parent_logit=features(windows,lengths)
            scores['parent'].append(float(parent_logit.sigmoid()))
            with torch.no_grad():
                for name,head in heads.items():scores[name].append(float(lme(head(values[name]),config['temperature']).sigmoid()))
            if index%300==0:print(json.dumps(dict(stage='development',files=index)),flush=True)
        for name,values in scores.items():
            frame=dev.copy();frame['FILE_FAKE_PROB']=values
            frame.to_csv(args.output/f'{name}_development.csv',index=False)
            results.append(dict(model=name,FILE_EER=official_eer(frame.FILE_FAKE,values)))
    assert all(sha256(p)==digest for p,digest in hashes.items())
    report=dict(status='complete_smoke' if args.smoke else 'complete',history=history,results=results,
                seconds=time.monotonic()-started,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
                trainable_parameters_per_head=193,protected_evaluation_consumed=False,automatic_submission_allowed=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
