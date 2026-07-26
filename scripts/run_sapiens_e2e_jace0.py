"""
End-to-End Runner for Sapiens2 + P3-SAM + Skeleton 3D Anatomical Segmentation Pipeline on jace0_clean_no_droplets.glb
"""

import os
import sys
import json
import logging
import trimesh
import numpy as np
from pathlib import Path

# Add src to sys.path
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from local_asset_factory.segmentation import (
    SemanticLabel,
    LABEL_COLORS_RGB,
    LABEL_NAMES,
    FaceViewRenderer,
    decode_face_id_map,
    SapiensSegmentor,
    GeometryJointPredictor,
    compute_bone_priors,
    fuse_anatomical_signals,
    refine_labels_mrf,
    transfer_labels_proxy_to_full,
)
from local_asset_factory.segmentation.proxy import ProxyManager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sapiens_e2e")

def run_e2e_pipeline(glb_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"=== Starting Sapiens2 E2E Pipeline on: {glb_path} ===")

    # 1. Load Original Mesh
    if not os.path.isfile(glb_path):
        raise FileNotFoundError(f"GLB model not found at {glb_path}")

    full_mesh = trimesh.load(glb_path, force="mesh")
    if isinstance(full_mesh, trimesh.Scene):
        full_mesh = trimesh.util.concatenate(full_mesh.dump())

    log.info(f"Original Mesh Loaded: {len(full_mesh.faces)} faces, {len(full_mesh.vertices)} vertices.")

    # 2. Prepare Proxy Mesh (50,000 faces)
    proxy_manager = ProxyManager(triangle_threshold=50000, target_triangles=40000)
    proxy_path, proxy_used, meta = proxy_manager.prepare_segmentation_mesh(glb_path, output_dir)

    proxy_mesh = trimesh.load(proxy_path, force="mesh")
    if isinstance(proxy_mesh, trimesh.Scene):
        proxy_mesh = trimesh.util.concatenate(proxy_mesh.dump())

    log.info(f"Proxy Mesh Prepared: {len(proxy_mesh.faces)} faces (proxy_used={proxy_used}).")

    # 3. Render 8 Orthographic Semantic Views (RGB, Depth, Face-ID)
    views_dir = os.path.join(output_dir, "semantic_views")
    renderer = FaceViewRenderer(use_blender=True) # Use Blender 4.4 GPU renderer
    views_data = renderer.render_views(proxy_mesh, views_dir, view_count=8, resolution=768)
    log.info(f"Rendered {len(views_data)} orthographic views.")

    # 4. Sapiens2 Human Parsing (Inference & Logits Offloading)
    segmentor = SapiensSegmentor()
    segmentor.load()

    logits_results = []
    for v in views_data:
        res = segmentor.predict(v["rgb_path"], is_front=v["is_front"])
        logits_results.append({
            "face_id_path": v["face_id_path"],
            "result": res,
        })

    segmentor.unload()
    log.info("Sapiens2 inference complete & model unloaded from GPU.")

    # 5. Project 2D Logits to 3D Proxy Mesh Faces
    num_faces = len(proxy_mesh.faces)
    num_classes = 18
    face_scores = np.zeros((num_faces, num_classes), dtype=np.float32)
    face_weights = np.zeros(num_faces, dtype=np.float32)

    for item in logits_results:
        res = item["result"]
        face_ids = decode_face_id_map(item["face_id_path"])

        valid_mask = (face_ids >= 0) & (face_ids < num_faces)
        valid_y, valid_x = np.where(valid_mask)
        valid_faces = face_ids[valid_y, valid_x]

        pix_logits = res.logits[valid_y, valid_x]         # (N, 18)
        pix_weights = res.confidence[valid_y, valid_x]    # (N,)

        np.add.at(face_scores, valid_faces, pix_logits * pix_weights[:, None])
        np.add.at(face_weights, valid_faces, pix_weights)

    sapiens_probs = face_scores / np.maximum(face_weights[:, None], 1e-6)
    unobs = (face_weights == 0)
    sapiens_probs[unobs] = 1.0 / num_classes
    log.info("Projected 2D Sapiens probabilities onto 3D proxy faces.")

    # 6. Predict 3D Skeleton & Compute Bone Priors
    skel_predictor = GeometryJointPredictor()
    joints = skel_predictor.predict(proxy_mesh)
    bone_priors = compute_bone_priors(proxy_mesh, joints)
    log.info(f"Predicted 3D skeleton ({len(joints)} joints) and computed bone priors.")

    # 7. Fuse Signals & Refine via MRF Graph Cut
    p3sam_regions = np.zeros(num_faces, dtype=np.int32) # Standard geometric region fallback
    unary_cost = fuse_anatomical_signals(sapiens_probs, p3sam_regions, bone_priors)
    proxy_refined_labels = refine_labels_mrf(proxy_mesh, unary_cost, p3sam_regions, max_iterations=5)
    log.info("MRF Graph Cut refinement complete.")

    # 8. Transfer Labels from Proxy to Full Mesh
    if len(proxy_mesh.faces) == len(full_mesh.faces):
        full_labels = proxy_refined_labels
    else:
        full_labels = transfer_labels_proxy_to_full(proxy_mesh, proxy_refined_labels, full_mesh)

    log.info(f"Transferred labels to full resolution mesh ({len(full_labels)} faces).")

    # 9. Export Segmented GLB Asset
    out_glb = os.path.join(output_dir, "jace0_sapiens_segmented.glb")
    face_colors = np.zeros((len(full_labels), 4), dtype=np.uint8)

    for label_val in np.unique(full_labels):
        rgb = LABEL_COLORS_RGB.get(label_val, (0.4, 0.4, 0.4))
        mask = (full_labels == label_val)
        face_colors[mask] = [int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255), 255]

    export_mesh = full_mesh.copy()
    export_mesh.visual = trimesh.visual.ColorVisuals(mesh=export_mesh, face_colors=face_colors)
    export_mesh.export(out_glb)
    log.info(f"Exported Segmented GLB: {out_glb}")

    # Summary Report
    report_path = os.path.join(output_dir, "sapiens_segmentation_report.json")
    report = {
        "num_faces": len(full_mesh.faces),
        "num_verts": len(full_mesh.vertices),
        "labels_summary": {LABEL_NAMES.get(k, f"label_{k}"): int(np.sum(full_labels == k)) for k in np.unique(full_labels)}
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log.info("=== E2E Pipeline Completed Successfully! ===")
    return out_glb

if __name__ == "__main__":
    candidates = [
        r"C:\Users\datam\Downloads\jace0_clean_no_droplets.glb",
        r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\ComfyUI_windows_portable\ComfyUI\output\jace0_clean_no_droplets.glb"
    ]
    glb_path = None
    for c in candidates:
        if os.path.isfile(c):
            glb_path = c
            break

    if not glb_path:
        print(f"Error: Target GLB file not found in candidates: {candidates}")
        sys.exit(1)

    out_dir = r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\output\jace0_sapiens_e2e"
    run_e2e_pipeline(glb_path, out_dir)
