"""Million-batch image encoding quality/time experiment, no profiler."""
import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=10000)
    parser.add_argument('--seeds',type=int,nargs='+',default=[1337,2026,3407])
    parser.add_argument('--labels',nargs='+',choices=['saved-z','fragment-sign','cutlass'],default=['saved-z','fragment-sign','cutlass'])
    parser.add_argument('--build',type=Path)
    parser.add_argument('--skip-build',action='store_true')
    parser.add_argument('--cuda-architectures')
    parser.add_argument('--cuda-compiler',type=Path)
    parser.add_argument('--input-image',type=Path,help='Reuse the exact shared 1024 RGB8 PNG instead of resizing again')
    args=parser.parse_args()
    if args.output.exists() or args.steps<1:
        parser.error('Use new output directory and positive steps')
    root=Path(__file__).resolve().parents[1]
    output=args.output.resolve(); output.mkdir(parents=True)
    build=args.build.resolve() if args.build else root/'build-sine-quality'
    original_image=root/'data/images/albert.jpg'
    image=output/'input-1024.png'
    if args.input_image:
        import shutil
        with Image.open(args.input_image) as source:
            if source.size!=(1024,1024) or source.mode!='RGB' or source.format!='PNG':
                parser.error('--input-image must be a 1024x1024 RGB PNG')
        with Image.open(original_image) as source:
            original_size=source.size
        shutil.copyfile(args.input_image,image)
    else:
        with Image.open(original_image) as source:
            original_size=source.size
            source.convert('RGB').resize((1024,1024),Image.Resampling.LANCZOS).save(image)
    write_json(output/'preprocessing.json',dict(original=str(original_image),original_size=original_size,
        resized_size=[1024,1024],resample='Pillow LANCZOS; direct square resize',mode='RGB8',
        original_sha256=hashlib.sha256(original_image.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),reused_fixture=bool(args.input_image)))
    base=json.loads((root/'data/config_oneblob_sine.json').read_text())
    labels=tuple(args.labels)
    if len(set(labels))!=len(labels) or 'saved-z' not in labels:
        parser.error('Use unique labels including saved-z')
    commands=[]
    def run(command,log,cwd=root):
        print('Running: '+' '.join(map(str,command)),flush=True)
        with log.open('w') as stream:
            result=subprocess.run(list(map(str,command)),cwd=cwd,stdout=stream,stderr=subprocess.STDOUT)
        commands.append(dict(command=list(map(str,command)),returncode=result.returncode,log=str(log)))
        write_json(output/'commands.json',commands)
        if result.returncode:
            raise RuntimeError(f'Inspect {log}')
    if not args.skip_build:
        configure=['cmake','-S',root,'-B',build,'-DCMAKE_BUILD_TYPE=Release',
                   '-DTCNN_BUILD_BENCHMARK=ON','-DTCNN_BUILD_EXAMPLES=OFF','-DTCNN_BUILD_TESTS=OFF']
        if args.cuda_architectures: configure.append('-DTCNN_CUDA_ARCHITECTURES='+args.cuda_architectures)
        if args.cuda_compiler: configure.append('-DCMAKE_CUDA_COMPILER='+str(args.cuda_compiler.resolve()))
        run(configure,output/'configure.log')
        run(['cmake','--build',build,'--target','bench_image_quality','-j','4'],output/'build.log')
    elif not (build/'bench_image_quality').is_file():
        parser.error('--skip-build requires an existing bench_image_quality executable')
    run(['nvidia-smi'],output/'gpu.txt')
    nvcc=str(args.cuda_compiler.resolve()) if args.cuda_compiler else shutil.which('nvcc')
    if nvcc: run([nvcc,'--version'],output/'nvcc.txt')
    executable=build/'bench_image_quality'
    def config(label,seed,steps,interval,warmup):
        data=json.loads(json.dumps(base))
        data['network'].update(otype='MLP' if label=='cutlass' else 'FullyFusedMLP',
                               sine_backward_mode='SignRecon' if label=='fragment-sign' else 'SavedZ')
        data['experiment']=dict(batch=1048576,steps=steps,eval_every=interval,warmup=warmup,seed=seed)
        return data
    def case(label,seed,steps,interval,warmup,folder):
        folder.mkdir(parents=True)
        write_json(folder/'config.json',config(label,seed,steps,interval,warmup))
        run([executable,'train',image,folder/'config.json',folder],folder/'train.log',folder)
        decoded=folder/'decoded'; decoded.mkdir()
        run([executable,'decode',image,folder/'model.msgpack',decoded],folder/'decode.log',folder)
        before=np.fromfile(folder/'prediction.f32',dtype=np.float32)
        after=np.fromfile(decoded/'prediction.f32',dtype=np.float32)
        if before.shape!=after.shape or not np.array_equal(before,after):
            raise RuntimeError(f'Serialized-model roundtrip mismatch: {folder}')
        reference=np.asarray(Image.open(folder/'reference.png'),dtype=np.float64)
        prediction=np.asarray(Image.open(decoded/'reconstruction.png'),dtype=np.float64)
        mse=float(np.mean(((prediction-reference)/255)**2))
        metrics=json.loads((decoded/'decode.json').read_text())['metrics']
        if not math.isclose(mse,metrics['decoded8_mse'],rel_tol=1e-10,abs_tol=1e-14):
            raise RuntimeError('Independent PNG metric check failed')
        error=np.clip(np.abs(prediction-reference)*4,0,255).astype(np.uint8)
        Image.fromarray(error).save(folder/'error-x4.png')
        report=json.loads((folder/'report.json').read_text())
        report.update(label=label,seed=seed,roundtrip_bitwise=True,png_metric_verified=True)
        report['prediction_sha256']=hashlib.sha256((folder/'prediction.f32').read_bytes()).hexdigest()
        report['decoded_prediction_sha256']=hashlib.sha256((decoded/'prediction.f32').read_bytes()).hexdigest()
        # Retain model, PNGs and float hashes, not duplicate full-resolution float dumps.
        (folder/'prediction.f32').unlink()
        (decoded/'prediction.f32').unlink()
        targets={}
        for target in (20,25,30,35):
            hits=[r for r in report['curve'] if r['decoded8_psnr_db']>=target]
            targets[str(target)]=dict(reached=bool(hits),training_seconds=hits[0]['training_seconds'] if hits else None,
                                     step=hits[0]['step'] if hits else None)
        report['targets_db']=targets
        best_decoded=folder/'decoded-best'; best_decoded.mkdir()
        run([executable,'decode',image,folder/'best-model.msgpack',best_decoded],folder/'decode-best.log',folder)
        best_metrics=json.loads((best_decoded/'decode.json').read_text())['metrics']
        if not math.isclose(best_metrics['decoded8_psnr_db'],report['best']['decoded8_psnr_db'],abs_tol=1e-10):
            raise RuntimeError('Best checkpoint reload quality mismatch')
        (best_decoded/'prediction.f32').unlink()
        report['best_reloaded']=best_metrics
        write_json(folder/'verified-report.json',report)
        return report
    for label in labels:
        case(label,1337,2,1,2,output/'smoke'/label)
    smoke_weights=[hashlib.sha256((output/'smoke'/l/'initial-weights.f32').read_bytes()).hexdigest() for l in labels]
    if len(set(smoke_weights))!=1:
        raise RuntimeError('Initial weight mismatch across backends')
    smoke_predictions=[np.fromfile(output/'smoke'/l/'initial-prediction.f32',dtype=np.float32) for l in labels]
    write_json(output/'smoke-check.json',dict(status='pass',initial_weight_sha256=smoke_weights[0],
        initial_prediction_max_abs_vs_saved_z={l:float(np.max(np.abs(v-smoke_predictions[0]))) for l,v in zip(labels,smoke_predictions)},
        serialized_roundtrip='bitwise float predictions; independent PNG MSE checks passed'))
    reports=[]
    for index,seed in enumerate(args.seeds):
        order=labels[index%len(labels):]+labels[:index%len(labels)]
        for label in order:
            folder=output/f'seed{seed}'/label
            report=case(label,seed,args.steps,100,125,folder)
            reports.append(report)
            write_json(output/'reports.json',reports)
            print(json.dumps(dict(label=label,seed=seed,training_seconds=report['training_seconds'],
                                  final_psnr8=report['final']['decoded8_psnr_db'])),flush=True)
        weight_hashes=[hashlib.sha256((output/f'seed{seed}'/l/'initial-weights.f32').read_bytes()).hexdigest() for l in labels]
        if len(set(weight_hashes))!=1:
            raise RuntimeError(f'Paired weight mismatch seed{seed}')
    summary=[]
    for label in labels:
        cases=[r for r in reports if r['label']==label]
        summary.append(dict(label=label,seeds=args.seeds,training_seconds_median=statistics.median(r['training_seconds'] for r in cases),
                            final_psnr8=[r['final']['decoded8_psnr_db'] for r in cases],
                            best_psnr8=[r['best']['decoded8_psnr_db'] for r in cases],
                            final_display_float_psnr=[r['final']['display_float_psnr_db'] for r in cases],
                            model_bytes=[r['model_bytes'] for r in cases],targets=[r['targets_db'] for r in cases]))
    write_json(output/'summary.json',summary)
    # Paired-seed curves, without averaging away divergent convergence behavior.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for x_key,name in (('step','psnr-step'),('training_seconds','psnr-training-time'),('wall_seconds','psnr-wall-time')):
        fig,axes=plt.subplots(1,len(args.seeds),figsize=(5*len(args.seeds),4),squeeze=False)
        for ax,seed in zip(axes[0],args.seeds):
            for r in reports:
                if r['seed']==seed:
                    ax.plot([p[x_key] for p in r['curve']],[p['decoded8_psnr_db'] for p in r['curve']],label=r['label'],
                            color={'saved-z':'#2563eb','fragment-sign':'#dc2626','cutlass':'#16a34a'}[r['label']])
            ax.set_title(f'seed {seed}'); ax.set_xlabel(x_key); ax.set_ylabel('Decoded RGB8 PSNR (dB)'); ax.grid(alpha=.2); ax.legend()
        fig.tight_layout(); fig.savefig(output/f'{name}.png',dpi=150); plt.close(fig)
    paths=[executable,image,original_image,root/'scripts/run_image_quality.py',root/'benchmarks/image/quality.cu',root/'src/fully_fused_mlp.cu']
    write_json(output/'hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    write_json(output/'status.json',dict(status='pass',steps=args.steps,seeds=args.seeds,batch=1048576,
        scope='One-image native image fitting/FP16 model encoding; not a general codec benchmark',
        timing='Training includes pixel sampling through Adam, all steps counted; evaluation/export/warmup separate; cold setup estimate includes benchmark fixture preparation',
        targets='First observed checkpoint, not interpolated; unreached targets censored',quality='gamma2.2 display RGB8 PSNR; unclipped linear MSE also recorded'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
