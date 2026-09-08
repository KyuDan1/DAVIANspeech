#!/usr/bin/env python3
"""Paired local XLS-R Voice head, file-only versus file+dense TRAIN supervision."""
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
    from sklearn.metrics import roc_curve
    from src.lossless_weights_v77 import sha256
    from src.common_encoder_probe import CommonTokenHead,FrozenEncoderTokens,complete_windows
    from src.local_voice_v80 import (voice_projection,local_statistics,temporal_geometry,
        interval_labels,LocalVoiceHead,file_logit,balanced_interval_loss)
    from src.local_voice_data_v80 import LocalVoicePairs,identity_collate
    from src.pipeline import load_audio
    from src.full_coverage_wpt import resolve_ffmpeg
    from train_dense_component_v71 import audited_catalog
    from train_common_encoder_probe import audit_protected_sources
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/local_voice_v80.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-report',type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config=yaml.safe_load(args.config.read_text())
    smoke_record=None
    if not args.smoke:
        if args.smoke_report is None:
            raise ValueError('full training requires a completed matching --smoke-report')
        smoke_record=json.loads(args.smoke_report.read_text())
        if smoke_record['status']!='complete_smoke_only':
            raise ValueError('smoke did not complete')
        old=json.loads((args.smoke_report.parent/'frozen.json').read_text())
        if old['config']!=config or sha256(args.smoke_report.parent/'frozen.json')!=smoke_record['frozen_sha256']:
            raise ValueError('smoke configuration/provenance changed')
        for path,digest in old['code_and_weights_sha256'].items():
            if sha256(path)!=digest:
                raise ValueError('smoke artifact changed')
    ffmpeg=resolve_ffmpeg()
    inv_path=ROOT/config['inventory']
    inv=json.loads(inv_path.read_text())
    if inv['status']!='complete_metadata_cache' or not inv['source_manifests_equal_completed_parent']:
        raise ValueError('source-audited completed inventory required')
    for path,digest in inv['artifacts_sha256'].items():
        if sha256(path)!=digest:
            raise ValueError('inventory provenance changed')
    frames={}
    for name,record in inv['tables'].items():
        if sha256(record['path'])!=record['sha256']:
            raise ValueError('inventory table changed')
        frames[name]=pd.read_csv(record['path'],dtype={'ID':str,'DATASET':str},low_memory=False)
    train=frames['train']
    dev=frames['development'].loc[frames['development'].VOICE_PRESENT.eq(1)].reset_index(drop=True)
    audit=audit_protected_sources(train,ROOT/config['partition_config'])
    catalog=audited_catalog(config,train)
    sources=[Path(__file__),args.config,inv_path,ROOT/config['partition_config'],
        ROOT/config['source_catalog'],ROOT/config['parent_head'],ROOT/config['encoder']/'model.safetensors',
        ROOT/'src/local_voice_v80.py',ROOT/'src/local_voice_data_v80.py',
        ROOT/'src/common_encoder_probe.py',ROOT/'src/dense_component_data_v71.py',
        ROOT/'src/long_component_stress.py',ROOT/'src/telephone_channel.py',
        ROOT/'src/pipeline.py',ROOT/'src/xlsr_antideepfake.py',
        ROOT/'scripts/train_dense_component_v71.py',ROOT/'scripts/train_common_encoder_probe.py',
        ROOT/'src/full_coverage_wpt.py',ffmpeg]
    hashes={str(p.resolve()):sha256(p) for p in sources}
    frozen=dict(config=config,code_and_weights_sha256=hashes,source_audit=audit,
        smoke=args.smoke,train_catalog_rows=len(catalog),development_voice_rows=len(dev),
        protected_evaluation_consumed=False,automatic_submission_allowed=False)
    if args.smoke_report is not None:
        frozen['smoke_report_sha256']=sha256(args.smoke_report)
    args.output.mkdir(parents=True)
    (args.output/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps(dict(stage='audited',catalog=len(catalog),dev=len(dev),smoke=args.smoke)),flush=True)
    torch.set_num_threads(2)
    torch.manual_seed(config['seed'])
    encoder=FrozenEncoderTokens('xlsr',ROOT/config['encoder'])
    # Verify the geometry constants against the actual native convolution.
    kernels=encoder.model.ssl.config.conv_kernel
    strides=encoder.model.ssl.config.conv_stride
    receptive,jump=1,1
    for k,s in zip(kernels,strides):
        receptive+=(k-1)*jump
        jump*=s
    if (receptive,jump)!=(400,320):
        raise ValueError('native convolution does not match temporal geometry')
    parent=CommonTokenHead(1920,width=96,pooling='mean').cuda()
    parent.load_state_dict(torch.load(ROOT/config['parent_head'],map_location='cpu',weights_only=False)['state_dict'])
    parent.eval().requires_grad_(False)
    heads={name:LocalVoiceHead(parent.output_weight[1],parent.output_bias[1]).cuda()
           for name in ['file_only','dense']}
    opts={name:torch.optim.AdamW(head.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
          for name,head in heads.items()}
    maximum_parent_difference=0.

    @torch.no_grad()
    def features(audio):
        nonlocal maximum_parent_difference
        windows,lengths,starts=complete_windows(audio)
        parts,masks,baseline=[],[],[]
        for window,length in zip(windows,lengths):
            tokens,mask=encoder(torch.from_numpy(window[None]),torch.tensor([length]))
            values=voice_projection(parent,tokens,mask)
            mean=(values*mask[...,None]).sum(1)/mask.sum(1)[:,None]
            var=((values-mean[:,None]).square()*mask[...,None]).sum(1)/mask.sum(1)[:,None]
            equivalent=(torch.cat([mean,(var+1e-5).sqrt()],-1)*parent.output_weight[1]).sum(-1)+parent.output_bias[1]
            native=parent(tokens,mask)[:,1]
            difference=float((equivalent-native).abs().max())
            maximum_parent_difference=max(maximum_parent_difference,difference)
            torch.testing.assert_close(equivalent,native,rtol=1e-5,atol=2e-6)
            part,valid=local_statistics(values,mask,config['kernel'],config['stride'])
            parts.append(part);masks.append(valid);baseline.append(native)
        centers,ownership=temporal_geometry(starts,lengths,tokens=tokens.shape[2],stride=config['stride'])
        valid=torch.cat(masks)&torch.from_numpy(ownership).cuda()
        baseline=torch.cat(baseline)
        global_score=(torch.logsumexp(2*baseline,0)-np.log(len(baseline)))/2
        return torch.cat(parts).detach(),valid,centers,global_score

    started=time.monotonic()
    history=[]
    pairs=2 if args.smoke else config['pairs_per_epoch']
    for epoch in range(1,(1 if args.smoke else config['epochs'])+1):
        dataset=LocalVoicePairs(catalog,config,ffmpeg,epoch,pairs)
        kwargs=dict(batch_size=None,num_workers=0 if args.smoke else config['workers'],collate_fn=identity_collate)
        if kwargs['num_workers']:
            kwargs['multiprocessing_context']='spawn'
        loader=DataLoader(dataset,**kwargs)
        losses={name:[] for name in heads}
        with (args.output/f'epoch_{epoch}_traces.jsonl').open('x') as stream:
            for step,(pair,trace) in enumerate(loader):
                pair_losses={name:[] for name in heads}
                for audio,label,spans in pair:
                    values,valid,centers,_=features(audio)
                    target,active=interval_labels(centers,valid.cpu().numpy(),spans,config['boundary_margin'])
                    target=torch.from_numpy(target).cuda();active=torch.from_numpy(active).cuda()
                    initial={name:head(values) for name,head in heads.items()}
                    if epoch==1 and step==0:
                        torch.testing.assert_close(initial['file_only'],initial['dense'],rtol=0,atol=0)
                    for name,logits in initial.items():
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(
                            file_logit(logits,valid,config['temperature']),logits.new_tensor(float(label)))
                        if name=='dense':
                            loss=loss+config['dense_weight']*balanced_interval_loss(logits,target,active)
                        pair_losses[name].append(loss)
                for name,parts in pair_losses.items():
                    loss=torch.stack(parts).mean()
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite training loss')
                    opts[name].zero_grad();loss.backward()
                    torch.nn.utils.clip_grad_norm_(heads[name].parameters(),5.,error_if_nonfinite=True)
                    opts[name].step();losses[name].append(float(loss.detach()))
                if any(p.grad is not None for p in encoder.model.parameters()) or any(p.grad is not None for p in parent.parameters()):
                    raise ValueError('frozen encoder/projection accumulated gradients')
                stream.write(json.dumps(trace)+'\n');stream.flush()
                if step%50==0:
                    print(json.dumps(dict(epoch=epoch,pair=step,loss={k:float(np.mean(v)) for k,v in losses.items()},
                        seconds=time.monotonic()-started)),flush=True)
        history.append(dict(epoch=epoch,loss={k:float(np.mean(v)) for k,v in losses.items()}))
        (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    for name,head in heads.items():
        if torch.equal(head.weight,parent.output_weight[1]) and torch.equal(head.bias,parent.output_bias[1]):
            raise ValueError('training did not update the local head')
        torch.save(dict(state_dict=head.state_dict(),config=config,smoke=args.smoke,variant=name,
            parent_sha256=sha256(ROOT/config['parent_head']),scope='Voice only'),args.output/f'{name}.pt')
    # A smoke consumes TRAIN only. Full run evaluates fixed final weights once.
    results=[]
    if not args.smoke:
        scores={name:[] for name in ['global_parent',*heads]}
        with torch.no_grad():
            for index,row in dev.iterrows():
                values,valid,_,global_score=features(load_audio(Path(row.PATH)))
                scores['global_parent'].append(float(global_score.sigmoid()))
                for name,head in heads.items():
                    scores[name].append(float(file_logit(head(values),valid,config['temperature']).sigmoid()))
                if index%200==0:
                    print(json.dumps(dict(stage='development',files=index,seconds=time.monotonic()-started)),flush=True)
        for name,predictions in scores.items():
            frame=dev.copy();frame['VOICE_FAKE_PROB']=predictions
            frame.to_csv(args.output/f'{name}_development.csv',index=False)
            fp,tp,_=roc_curve(frame.VOICE_FAKE,frame.VOICE_FAKE_PROB,pos_label=1,drop_intermediate=False)
            i=np.argmin(abs(fp-(1-tp)))
            results.append(dict(variant=name,rows=len(frame),VOICE_EER=float((fp[i]+1-tp[i])/2)))
    if any(sha256(path)!=digest for path,digest in hashes.items()):
        raise ValueError('frozen experiment artifact changed')
    report=dict(status='complete_smoke_only' if args.smoke else 'complete',results=results,
        seconds=time.monotonic()-started,maximum_parent_logit_difference=maximum_parent_difference,
        peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,trainable_parameters_per_head=193,
        both_heads_updated=True,initial_heads_bit_exact=True,frozen_encoder_gradients_absent=True,
        frozen_sha256=sha256(args.output/'frozen.json'),protected_evaluation_consumed=False,
        automatic_submission_allowed=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
