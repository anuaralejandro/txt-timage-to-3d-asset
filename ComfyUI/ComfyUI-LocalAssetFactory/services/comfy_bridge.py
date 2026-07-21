"""
ComfyUI-LocalAssetFactory · ComfyUI Bridge
A reusable wrapper around the local ComfyUI HTTP API.

Responsibilities:
- Load workflow JSON templates.
- Inject prompts, seeds, and parameters into workflow nodes.
- Queue prompts via the /prompt endpoint.
- Poll for completion via /history.
- Retrieve generated images/files via /view.
- Uniform interface used by concept-image, texture, and TRELLIS nodes.
"""

from __future__ import annotations

import copy
import json
import os
import time
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

from .. import config
from ..utilities.logging_utils import get_logger
from ..utilities.retry import retry

log = get_logger(__name__)


class ComfyBridgeError(RuntimeError):
    """Fatal error communicating with local ComfyUI."""


class ComfyBridge:
    """Wrapper for the local ComfyUI REST API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.base_url = (base_url or config.COMFYUI_BASE_URL).rstrip("/")
        self.timeout = timeout or config.DEFAULT_TIMEOUT_SECONDS
        self.client_id = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Quick check that ComfyUI is reachable."""
        try:
            r = requests.get(f"{self.base_url}/system_stats", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Workflow loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_workflow(path: str) -> Dict[str, Any]:
        """Load a ComfyUI workflow JSON from disk."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Workflow not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            wf = json.load(f)
        log.info("Loaded workflow: %s (%d nodes)", path, len(wf))
        return wf

    # ------------------------------------------------------------------
    # Workflow manipulation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def set_node_input(
        workflow: Dict[str, Any],
        node_id: str,
        field: str,
        value: Any,
    ) -> Dict[str, Any]:
        """Set a single input field on a node (by string ID)."""
        wf = copy.deepcopy(workflow)
        if node_id not in wf:
            raise KeyError(f"Node '{node_id}' not found in workflow")
        inputs = wf[node_id].get("inputs", {})
        inputs[field] = value
        wf[node_id]["inputs"] = inputs
        return wf

    @staticmethod
    def find_node_by_class(
        workflow: Dict[str, Any], class_type: str
    ) -> Optional[str]:
        """Return the first node ID matching *class_type*, or None."""
        for nid, node in workflow.items():
            if node.get("class_type") == class_type:
                return nid
        return None

    @staticmethod
    def find_nodes_by_class(
        workflow: Dict[str, Any], class_type: str
    ) -> List[str]:
        """Return all node IDs matching *class_type*."""
        return [
            nid
            for nid, node in workflow.items()
            if node.get("class_type") == class_type
        ]

    def inject_prompt_params(
        self,
        workflow: Dict[str, Any],
        *,
        positive_prompt: str = "",
        negative_prompt: str = "",
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
        checkpoint_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Best-effort injection of common parameters into known node types."""
        wf = copy.deepcopy(workflow)

        # CLIP Text Encode — positive
        if positive_prompt:
            for nid in self.find_nodes_by_class(wf, "CLIPTextEncode"):
                title = wf[nid].get("_meta", {}).get("title", "").lower()
                inputs = wf[nid].get("inputs", {})
                if "positive" in title or "prompt" in title or "text" in inputs:
                    inputs["text"] = positive_prompt
                    wf[nid]["inputs"] = inputs
                    log.info("Injected positive prompt into node %s", nid)
                    break  # inject into first matching only

        # CLIP Text Encode — negative
        if negative_prompt:
            for nid in self.find_nodes_by_class(wf, "CLIPTextEncode"):
                title = wf[nid].get("_meta", {}).get("title", "").lower()
                inputs = wf[nid].get("inputs", {})
                if "negative" in title:
                    inputs["text"] = negative_prompt
                    wf[nid]["inputs"] = inputs
                    log.info("Injected negative prompt into node %s", nid)
                    break

        # KSampler
        sampler_ids = self.find_nodes_by_class(wf, "KSampler") + \
                      self.find_nodes_by_class(wf, "KSamplerAdvanced")
        for nid in sampler_ids:
            inputs = wf[nid].get("inputs", {})
            if seed is not None:
                inputs["seed"] = seed
            if steps is not None:
                inputs["steps"] = steps
            if cfg is not None:
                inputs["cfg"] = cfg
            wf[nid]["inputs"] = inputs

        # Empty Latent Image
        if width or height:
            for nid in self.find_nodes_by_class(wf, "EmptyLatentImage"):
                inputs = wf[nid].get("inputs", {})
                if width:
                    inputs["width"] = width
                if height:
                    inputs["height"] = height
                wf[nid]["inputs"] = inputs

        # Checkpoint Loader
        if checkpoint_name:
            for cls in ("CheckpointLoaderSimple", "CheckpointLoader"):
                for nid in self.find_nodes_by_class(wf, cls):
                    inputs = wf[nid].get("inputs", {})
                    inputs["ckpt_name"] = checkpoint_name
                    wf[nid]["inputs"] = inputs

        return wf

    # ------------------------------------------------------------------
    # Queue & poll
    # ------------------------------------------------------------------

    @retry(max_retries=1, delay=2.0, exceptions=(requests.RequestException,))
    def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """Submit a prompt to ComfyUI and return the prompt ID."""
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }
        r = requests.post(
            f"{self.base_url}/prompt",
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        prompt_id = data.get("prompt_id", "")
        if not prompt_id:
            raise ComfyBridgeError(f"No prompt_id in response: {data}")
        log.info("Queued prompt: %s", prompt_id)
        return prompt_id

    def wait_for_completion(
        self,
        prompt_id: str,
        poll_interval: float = 2.0,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Poll /history until the prompt finishes or timeout.

        Returns the history entry for the prompt.
        """
        deadline = time.time() + (timeout or self.timeout)
        while time.time() < deadline:
            try:
                r = requests.get(
                    f"{self.base_url}/history/{prompt_id}",
                    timeout=10,
                )
                if r.status_code == 200:
                    history = r.json()
                    if prompt_id in history:
                        entry = history[prompt_id]
                        status = entry.get("status", {})
                        if status.get("completed", False) or status.get("status_str") == "success":
                            log.info("Prompt %s completed.", prompt_id)
                            return entry
                        if status.get("status_str") == "error":
                            raise ComfyBridgeError(
                                f"ComfyUI prompt {prompt_id} failed: "
                                f"{status.get('messages', 'unknown error')}"
                            )
            except requests.RequestException:
                pass  # transient network issue
            time.sleep(poll_interval)

        raise ComfyBridgeError(
            f"Timeout waiting for prompt {prompt_id} after {timeout or self.timeout}s"
        )

    # ------------------------------------------------------------------
    # Retrieve outputs
    # ------------------------------------------------------------------

    def get_output_images(
        self,
        history_entry: Dict[str, Any],
    ) -> List[Tuple[str, Image.Image]]:
        """Extract generated images from a completed history entry.

        Returns list of (filename, PIL.Image) tuples.
        """
        results: List[Tuple[str, Image.Image]] = []
        outputs = history_entry.get("outputs", {})
        for node_id, node_out in outputs.items():
            images_info = node_out.get("images", [])
            for img_info in images_info:
                filename = img_info.get("filename", "")
                subfolder = img_info.get("subfolder", "")
                img_type = img_info.get("type", "output")
                if not filename:
                    continue
                try:
                    img = self._download_image(filename, subfolder, img_type)
                    results.append((filename, img))
                except Exception as exc:
                    log.warning("Could not retrieve image %s: %s", filename, exc)
        return results

    def _download_image(
        self,
        filename: str,
        subfolder: str = "",
        img_type: str = "output",
    ) -> Image.Image:
        """Download a single image from ComfyUI /view endpoint."""
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": img_type,
        }
        r = requests.get(f"{self.base_url}/view", params=params, timeout=30)
        r.raise_for_status()
        return Image.open(BytesIO(r.content))

    # ------------------------------------------------------------------
    # High-level execute
    # ------------------------------------------------------------------

    def execute_workflow(
        self,
        workflow: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Queue a workflow, wait for completion, return the history entry."""
        prompt_id = self.queue_prompt(workflow)
        return self.wait_for_completion(prompt_id, timeout=timeout)

    def execute_and_get_images(
        self,
        workflow: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> List[Tuple[str, Image.Image]]:
        """Queue, wait, and return the output images."""
        entry = self.execute_workflow(workflow, timeout=timeout)
        return self.get_output_images(entry)

    def get_output_files(
        self,
        history_entry: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Extract generated non-image files (GLB, OBJ, etc.) from history.

        Returns list of dicts with filename, subfolder, type.
        """
        results: List[Dict[str, str]] = []
        outputs = history_entry.get("outputs", {})
        for node_id, node_out in outputs.items():
            # Check for gltf / mesh outputs
            for key in ("gltf", "mesh", "files", "models", "3d"):
                files_info = node_out.get(key, [])
                if isinstance(files_info, list):
                    for f_info in files_info:
                        if isinstance(f_info, dict) and f_info.get("filename"):
                            results.append(f_info)
        return results
