import os
import sys
import argparse
import shutil
from pathlib import Path

def print_success(msg): print(f"\033[92m[SUCCESS]\033[0m {msg}")
def print_info(msg): print(f"\033[94m[INFO]\033[0m {msg}")
def print_warn(msg): print(f"\033[93m[WARN]\033[0m {msg}")
def print_error(msg): print(f"\033[91m[ERROR]\033[0m {msg}")

def link_or_copy_dir(src: Path, dst: Path, dry_run: bool = False):
    """Links (junctions on Windows) or copies directories idempotently."""
    if not src.exists():
        print_error(f"Source directory does not exist: {src}")
        return False

    if dst.exists():
        is_linked = dst.is_symlink()
        if os.name == 'nt' and not is_linked:
            is_linked = os.path.isjunction(str(dst)) if hasattr(os.path, 'isjunction') else False
        if is_linked:
            print_info(f"Directory already linked: {dst}")
            return True
        else:
            print_info(f"Directory exists (copying/updating files): {dst}")
            if not dry_run:
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print_success(f"Updated files in {dst}")
            return True

    if dry_run:
        print_info(f"[DRY-RUN] Would link/copy {src} -> {dst}")
        return True

    try:
        # Create junction on Windows, symlink on Unix
        if os.name == 'nt':
            import _winapi
            _winapi.CreateJunction(str(src.absolute()), str(dst.absolute()))
        else:
            os.symlink(src.absolute(), dst.absolute(), target_is_directory=True)
        print_success(f"Linked: {src} -> {dst}")
    except OSError as e:
        print_warn(f"Failed to create link ({e}). Attempting to copy instead...")
        try:
            shutil.copytree(src, dst)
            print_success(f"Copied: {src} -> {dst}")
        except Exception as copy_e:
            print_error(f"Failed to copy: {copy_e}")
            return False
    return True

def link_or_copy_file(src: Path, dst: Path, dry_run: bool = False):
    if not src.exists():
        print_error(f"Source file does not exist: {src}")
        return False
        
    if dst.exists():
        print_info(f"File already exists, overwriting if different: {dst}")
        if not dry_run:
            try:
                # Remove if it exists to replace safely
                if dst.is_symlink():
                    dst.unlink()
                else:
                    os.remove(dst)
            except OSError as e:
                print_error(f"Could not remove existing file: {e}")
                return False

    if dry_run:
        print_info(f"[DRY-RUN] Would link/copy {src} -> {dst}")
        return True
        
    try:
        if os.name == 'nt':
            # Create hardlink on Windows
            import _winapi
            _winapi.CreateHardLink(str(dst.absolute()), str(src.absolute()))
        else:
            os.symlink(src.absolute(), dst.absolute())
        print_success(f"Linked: {src} -> {dst}")
    except OSError as e:
        print_warn(f"Failed to create link ({e}). Attempting to copy instead...")
        try:
            shutil.copy2(src, dst)
            print_success(f"Copied: {src} -> {dst}")
        except Exception as copy_e:
            print_error(f"Failed to copy: {copy_e}")
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Installs ComfyUI-LocalAssetFactory custom nodes to a ComfyUI instance.")
    parser.add_argument("--comfyui-dir", type=str, required=True, help="Path to the target ComfyUI root directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without modifying files.")
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent.absolute()
    target_comfy = Path(args.comfyui_dir).absolute()
    
    if not (target_comfy / "main.py").exists() and not (target_comfy / "comfy").exists():
        print_error(f"Could not find a valid ComfyUI installation at {target_comfy}")
        sys.exit(1)
        
    custom_nodes_dir = target_comfy / "custom_nodes"
    if not custom_nodes_dir.exists():
        if args.dry_run:
            print_info(f"[DRY-RUN] Would create {custom_nodes_dir}")
        else:
            custom_nodes_dir.mkdir(parents=True, exist_ok=True)
            print_success(f"Created {custom_nodes_dir}")
            
    # Install Custom Nodes
    src_nodes = repo_root / "comfyui" / "ComfyUI-LocalAssetFactory"
    dst_nodes = custom_nodes_dir / "ComfyUI-LocalAssetFactory"
    
    link_or_copy_dir(src_nodes, dst_nodes, args.dry_run)
    
    # Install Workflows
    src_workflows = repo_root / "workflows"
    dst_workflows = target_comfy / "user" / "default" / "workflows" / "LocalAssetFactory"
    
    if not dst_workflows.parent.exists():
        if args.dry_run:
            print_info(f"[DRY-RUN] Would create {dst_workflows.parent}")
        else:
            dst_workflows.parent.mkdir(parents=True, exist_ok=True)
            
    link_or_copy_dir(src_workflows, dst_workflows, args.dry_run)
    
    print_success("Installation complete.")

if __name__ == "__main__":
    main()
