"""
local_asset_factory · benchmarks · run_hunyuan_baseline
Official baseline execution for A/B testing the Hunyuan3D pipeline.
"""
import argparse
import json
import logging
from pathlib import Path
import time
import torch

from local_asset_factory.preflight.preflight_runner import PreflightRunner, PreflightConfig
from local_asset_factory.multiview.canonical_views import build_canonical_view_set
from local_asset_factory.geometry.hunyuan2mv_client import Hunyuan2MVClient
from services.hunyuan3d_2mv.backend import Hunyuan2MVBackend
from local_asset_factory.orchestration.vram_scheduler import VRAMScheduler
from local_asset_factory.scoring.mesh_scoring import score_candidate

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

class DummyStore:
    def __init__(self, output_dir):
        self.job_root = Path(output_dir)
        self.job_root.mkdir(parents=True, exist_ok=True)
    def path(self, p):
        target = self.job_root / p
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
    def register(self, cand):
        return cand
    def preserve_failed_candidate(self, cid, reason):
        pass

class DummyRequest:
    def __init__(self, views, job_id, inference_steps):
        self.views = views
        self.job_id = job_id
        self.timeout_seconds = 600
        self.seeds = [11]
        self.inference_steps = inference_steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", required=True)
    parser.add_argument("--back", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--mode", choices=["standard", "turbo"], default="standard")
    parser.add_argument("--output-dir", default="outputs/baseline")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    views = {
        "front": args.front,
        "back": args.back,
        "left": args.left,
        "right": args.right
    }
    
    log.info("Starting baseline run for mode: %s", args.mode)

    # 1. Preflight
    runner = PreflightRunner(PreflightConfig())
    results = runner.validate_view_set(views, job_id="baseline")
    if not runner.all_passed(results):
        log.error("Preflight failed:")
        print(runner.summary(results))
        return

    # 2. Canonical Views (Joint Registration)
    vs = build_canonical_view_set(views, out_dir / "canonical", "baseline")

    # 3. Shape Generation
    backend = Hunyuan2MVBackend()
    scheduler = VRAMScheduler()
    client = Hunyuan2MVClient(backend, scheduler)
    store = DummyStore(out_dir)
    
    steps = 30 if args.mode == "standard" else 10
    
    checkpoint_views = vs.checkpoint_views()
    view_paths = {k: str(out_dir / "canonical" / v.relative_path) for k, v in checkpoint_views.items()}
    req = DummyRequest(view_paths, "baseline", steps)

    log.info("Running candidate generation...")
    cand = client.generate_candidate(req, store, seed=11, variant=args.mode, steps=steps)

    if cand.passed_gates:
        log.info("Candidate generated successfully.")
        
        # 4. Scoring
        score_res = score_candidate(str(out_dir / cand.relative_path), view_paths)
        
        # 5. Output metrics
        metrics_file = out_dir / "metrics.json"
        metrics_file.write_text(json.dumps(score_res, indent=2))
        log.info("Metrics written to %s", metrics_file)
        
        if score_res["passed"]:
            log.info("Baseline passed all strict mesh gates!")
        else:
            log.warning("Baseline failed strict mesh gates: %s", score_res["failed_gates"])
    else:
        log.error("Candidate generation failed: %s", cand.warnings)

if __name__ == "__main__":
    main()
