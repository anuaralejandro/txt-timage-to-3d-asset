import os

def get_size(start_path):
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    try:
                        total_size += os.path.getsize(fp)
                    except OSError:
                        pass
    except OSError:
        pass
    return total_size

def analyze_directory(base_path, max_depth=2, current_depth=0):
    if current_depth > max_depth:
        return
    
    items = []
    try:
        for name in os.listdir(base_path):
            path = os.path.join(base_path, name)
            if os.path.isdir(path):
                size = get_size(path)
                items.append((name, size, True, path))
            else:
                size = os.path.getsize(path)
                items.append((name, size, False, path))
    except OSError:
        return

    items.sort(key=lambda x: x[1], reverse=True)
    
    for name, size, is_dir, path in items:
        if size < 100 * 1024 * 1024: # Ignore smaller than 100MB
            continue
            
        indent = "  " * current_depth
        if is_dir:
            print(f"{indent}📁 {name}/ - {size / (1024**3):.2f} GB")
            if current_depth < max_depth:
                analyze_directory(path, max_depth, current_depth + 1)
        else:
            print(f"{indent}📄 {name} - {size / (1024**2):.2f} MB")

base_path = r"C:\Users\datam\Videos\ComftyUI-text-2-3d-asset-gen\ComfyUI_windows_portable"
print(f"Analyzing {base_path} (Showing items > 100MB)\n")
total = get_size(base_path)
print(f"Total size: {total / (1024**3):.2f} GB\n")
analyze_directory(base_path, max_depth=3)
