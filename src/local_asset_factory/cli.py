"""
local_asset_factory · CLI
Command-line entry point for running preflight checks, candidate generation, and pipeline execution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .observability.artifact_store import ArtifactStore
from .preflight.preflight_runner import PreflightRunner
from .multiview.canonical_views import build_canonical_view_set


def main():
    parser = argparse.ArgumentParser(
        prog="asset-factory",
        description="Local 3D character factory — Hunyuan multiview character pipeline",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Sub-commands")

    # asset-factory preflight --views front.png left.png back.png --job-id TEST
    pf_parser = subparsers.add_parser("preflight", help="Run input preflight checks on view set")
    pf_parser.add_argument("--front", required=True, help="Path to front view image")
    pf_parser.add_argument("--left", required=True, help="Path to left view image")
    pf_parser.add_argument("--back", required=True, help="Path to back view image")
    pf_parser.add_argument("--right", help="Optional right view image")
    pf_parser.add_argument("--job-id", default="cli-job-001", help="Job ID")

    # asset-factory normalize-views --front f.png --left l.png --back b.png --output-dir artifacts/views
    norm_parser = subparsers.add_parser("normalize-views", help="Crop and normalize views to 1024x1024")
    norm_parser.add_argument("--front", required=True)
    norm_parser.add_argument("--left", required=True)
    norm_parser.add_argument("--back", required=True)
    norm_parser.add_argument("--right")
    norm_parser.add_argument("--output-dir", required=True)
    norm_parser.add_argument("--job-id", default="cli-job-001")

    args = parser.parse_args()

    if args.subcommand == "preflight":
        views = {"front": args.front, "left": args.left, "back": args.back}
        if args.right:
            views["right"] = args.right

        runner = PreflightRunner()
        results = runner.validate_view_set(views, job_id=args.job_id)
        print(runner.summary(results))

        if not runner.all_passed(results):
            sys.exit(1)
        sys.exit(0)

    elif args.subcommand == "normalize-views":
        views = {"front": args.front, "left": args.left, "back": args.back}
        if args.right:
            views["right"] = args.right

        vs = build_canonical_view_set(views, args.output_dir, job_id=args.job_id)
        print(f"CanonicalViewSet created at {args.output_dir} with {len(vs.all_views())} views.")
        sys.exit(0)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
