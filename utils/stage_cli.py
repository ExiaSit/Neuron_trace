import argparse, json
from pathlib import Path
def run_spec(run):
    parser=argparse.ArgumentParser(description='单样本阶段；完整批处理请使用 run_pipeline.py')
    parser.add_argument('--spec',required=True,help='完整流程生成的 manifests/ID.json')
    args=parser.parse_args()
    run(json.loads(Path(args.spec).read_text()))
