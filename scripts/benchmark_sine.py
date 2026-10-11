"""Repeat isolated native image-benchmark cases without changing its timing protocol."""
import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'data/config_oneblob_sine.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batches',type=int,nargs='+',default=[65536,1048576])
    parser.add_argument('--modes',nargs='+',choices=['SavedZ','SignRecon','Cutlass'],default=['SavedZ','SignRecon','Cutlass'])
    parser.add_argument('--rounds',type=int,default=3)
    args=parser.parse_args()
    if args.output.exists() or args.rounds<3 or any(b<16384 or b>2097152 or b&(b-1) for b in args.batches):
        parser.error('Use a new output directory,three or more rounds,and original power-of-two batches')
    out=args.output.resolve();out.mkdir(parents=True)
    executable=args.executable.resolve();image=args.image.resolve();base=json.loads(args.config.read_text())
    samples=[];commands=[];configs={}
    for mode in args.modes:
        config=json.loads(json.dumps(base));config['network']['sine_backward_mode']='SignRecon' if mode=='SignRecon' else 'SavedZ'
        path=out/(mode+'.json');path.write_text(json.dumps(config,indent=2)+'\n');configs[mode]=path
    for round_index in range(args.rounds):
        modes=args.modes[round_index%len(args.modes):]+args.modes[:round_index%len(args.modes)]
        for batch in (args.batches if round_index%2==0 else list(reversed(args.batches))):
            for mode in modes:
                folder=out/f'round{round_index+1}-{mode}-b{batch}';folder.mkdir()
                backend='cutlass' if mode=='Cutlass' else 'fully_fused'
                command=[str(executable),str(image),str(configs[mode]),backend,str(batch)]
                print('Running: '+' '.join(command),flush=True)
                with (folder/'console.log').open('w') as log:
                    result=subprocess.run(command,cwd=folder,stdout=log,stderr=subprocess.STDOUT)
                commands.append(dict(command=command,returncode=result.returncode))
                (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
                if result.returncode or 'Uncaught exception:' in (folder/'console.log').read_text():
                    raise RuntimeError('Failed benchmark:'+str(folder))
                row=json.loads((folder/'bench_result_ours.json').read_text())[backend][0]
                if row['batch_size']!=batch or not all(math.isfinite(row[k]) and row[k]>0 for k in ('training_throughput','inference_throughput')):
                    raise RuntimeError('Invalid measured throughput')
                samples.append(dict(mode=mode,round=round_index+1,batch=batch,raw=row,
                    training_ms=1000*batch/row['training_throughput'],inference_ms=1000*batch/row['inference_throughput']))
                (out/'samples.json').write_text(json.dumps(samples,indent=2)+'\n')
    summary=[]
    for batch in args.batches:
        for mode in args.modes:
            rows=[r for r in samples if r['batch']==batch and r['mode']==mode]
            summary.append(dict(mode=mode,batch=batch,**{k:dict(median=statistics.median(r[k] for r in rows),min=min(r[k] for r in rows),max=max(r[k] for r in rows)) for k in ('training_ms','inference_ms')}))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (executable,image,args.config,Path(__file__).resolve())},indent=2)+'\n')
    (out/'status.json').write_text(json.dumps(dict(status='pass',cases=len(samples),protocol='Original warmup,first-half exclusion,sampling/loss readback and cooldowns retained'),indent=2)+'\n')


if __name__=='__main__':
    main()
