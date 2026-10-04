#!/usr/bin/env python3
"""One bounded reconstruction-loss ablation; no new networks or optimizer implementation."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from concepts import make_model,windows,train_anchor,dataset_metrics,replay
from learning import write_result
from training import verify_sources


def run(upstream,concept_dir,output_dir,steps=900):
    if type(steps) is not int or not 1<=steps<=1500:raise ValueError('budget 1..1500 required')
    torch.set_num_threads(1);sources=verify_sources(upstream)
    from loguru import logger
    logger.disable('gear_sonic')
    with np.load(concept_dir/'motions.npz',allow_pickle=False) as z:bank={k:z[k].copy() for k in z.files}
    train=windows(bank,range(24));heldout=windows(bank,range(24,32));model=make_model(upstream)
    fit=train_anchor(model,train,steps=steps,aux_coef=0.)
    report={'execution':'COMPLETE','scope':'offline reconstruction-loss ablation, not a SONIC PPO benchmark',
        'source_sha256':sources,'shared_bank_sha256':hashlib.sha256((concept_dir/'motions.npz').read_bytes()).hexdigest(),
        'training':fit,'heldout':dataset_metrics(model,heldout),
        'replay':replay(model,bank,list(range(24,32))),
        'comparison':'same initial joint model, samples, batch size, optimizer, steps and full future input as anchor; only aux_coef .5 to 0',
        'heldout_episode_ids':list(range(24,32))}
    write_result(output_dir/'no_aux.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',type=Path,default=os.environ.get('SONIC_UPSTREAM'))
    p.add_argument('--concept-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--steps',type=int,default=900)
    a=p.parse_args();r=run(a.upstream,a.concept_dir,a.output_dir,a.steps);print(json.dumps(r,allow_nan=False))
