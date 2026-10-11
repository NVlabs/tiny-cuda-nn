"""Build and run native Sine tests without private data or server dependencies."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cuda-architectures')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]; build=args.build.resolve(); out=args.output.resolve()
    if out.exists(): parser.error('Use a new result directory')
    out.mkdir(parents=True)
    def run(command,name,env=None):
        print('Running: '+' '.join(map(str,command)),flush=True)
        with (out/name).open('w') as stream:
            subprocess.run(list(map(str,command)),cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    configure=['cmake','-S',root,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DTCNN_BUILD_TESTS=ON',
        '-DTCNN_BUILD_EXAMPLES=OFF','-DTCNN_BUILD_BENCHMARK=OFF']
    if args.cuda_architectures: configure.append('-DTCNN_CUDA_ARCHITECTURES='+args.cuda_architectures)
    run(configure,'configure.log')
    run(['cmake','--build',build,'--config','Release','--target','test_sine','probe_sine_fragment_layout','-j','2'],'build.log')
    env=dict(os.environ,TCNN_SINE_NUMERICS_REPORT=str(out/'unit-metrics.json'))
    run(['ctest','--test-dir',build,'-C','Release','-R','^(test_sine|probe_sine_fragment_layout)$','--output-on-failure'],'ctest.log',env)
    listing=json.loads(subprocess.check_output(['ctest','--test-dir',str(build),'-C','Release','--show-only=json-v1'],text=True))
    files=[root/'src/fully_fused_mlp.cu',root/'src/network.cu',root/'include/tiny-cuda-nn/networks/fully_fused_mlp.h',
           root/'include/tiny-cuda-nn/common_device.h',root/'tests/test_sine.cu',root/'tests/probe_sine_fragment_layout.cu',Path(__file__).resolve()]
    files.extend(Path(t['command'][0]) for t in listing['tests'] if t['name'] in ('test_sine','probe_sine_fragment_layout'))
    (out/'hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
    (out/'status.json').write_text(json.dumps(dict(status='pass',scope='Controlled FP16 native semantics;SignRecon remains approximate'),indent=2)+'\n')


if __name__=='__main__':
    main()
