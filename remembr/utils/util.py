
from pathlib import Path
from inspect import signature
import yaml
from PIL import Image
import numpy as np

def get_frames(file):
    import cv2

    vidcap = cv2.VideoCapture(file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = frame_count

    if fps == None or frame_count == None:
        # if one of fps or frame_count is None, still recompute
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0 or frame_count == 0:
        print("Video file not found. return empty images.")
        return [
            Image.new("RGB", (720, 720)),
        ] * num_frames
    
    duration = frame_count / fps
    frame_interval = frame_count // num_frames
    if frame_interval == 0 and frame_count <= 1:
        print("frame_interval is equal to 0. return empty image.")
        return [
            Image.new("RGB", (720, 720)),
        ] * num_frames

    images = []
    count = 0
    success = True
    frame_indices = np.linspace(0, frame_count-1 , num_frames, dtype=int)

    while success:
        if frame_count >= num_frames:
            success, frame = vidcap.read()
            if count in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                images.append(im_pil)
                if len(images) >= num_frames:
                    return images
            count += 1
        else:
            # Left padding frames if the video is not long enough
            success, frame = vidcap.read()
            if success:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                images.append(im_pil)
                count += 1
            elif count >= 1:
                width, height = images[-1].size
                images = [Image.new("RGB", (width, height))] * (num_frames - len(images)) + images
                print("padding frames:", (num_frames - len(images)))
                return images
            else: 
                break
    raise ValueError("Did not find enough frames in the video. return empty image.")
        

def file_to_string(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read().strip()

import json, re

def json_fixer(bad_json: str) -> str:
    # step 1: clean up duplicate braces
    
    add_json_mark = True if "```json" in bad_json else False
    clean = re.sub(r"^```json|```$", "", bad_json.strip(), flags=re.MULTILINE).strip()
    if clean[:2] == "{{" and clean[-2:] == "}}":
        print("Fixing duplicate braces...")
        # Replace leading consecutive '{'
        clean = re.sub(r'^\{+', '{', clean)
        # Replace trailing consecutive '}'
        clean = re.sub(r'\}+$', '}', clean)
    clean = re.sub(r',(\s*})', r'\1', clean)  # remove trailing commas before closing braces

    # --- Step 2: fix common missing brace pattern ---
    if clean.strip().endswith("]"):
        open_braces = clean.count("{")
        close_braces = clean.count("}")
        if open_braces > close_braces:
            clean = clean.rstrip("]") + "}" + "]"

    # --- Step 3: try to parse and pretty-print ---
    try:
        obj = json.loads(clean, )
        fixed = json.dumps(obj, indent=2, ensure_ascii=False)
        return "```json\n" + fixed + "\n```" if add_json_mark else fixed
    except json.JSONDecodeError as e:
        return bad_json  # return original if unable to fix
    

def int_or_json(s):
    # try integer
    try:
        return int(s)
    except ValueError:
        pass

    # try JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Argument must be int or valid JSON, got: {s}")


def predefined_filename(caption_path, qa_path):
    return (f'{Path(caption_path).name[:-5]}__{Path(qa_path).name[:-5]}.json').replace("/", "_") 


def print_cfg(name, cfg):
    print(f"\n========== {name} ==========")
    print(json.dumps(cfg, indent=2, sort_keys=True))


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def deep_merge(base, child):
    """
    child overrides base recursively
    """
    out = dict(base)
    for k, v in child.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml_with_base(cfg_path):
    cfg_path = Path(cfg_path)
    cfg = load_yaml(cfg_path)

    if "_base_" in cfg:
        base_path = cfg_path.parent / cfg["_base_"]
        base_cfg = load_yaml_with_base(base_path)

        cfg = deep_merge(base_cfg, cfg)
        cfg.pop("_base_", None)

    return cfg


def validate_params(cls, params):
    sig = signature(cls.__init__)
    valid = set(sig.parameters.keys()) - {"self"}
    unknown = set(params) - valid
    if unknown:
        raise ValueError(f"Unknown params: {unknown}")


def instantiate_from_yaml(cfg_path, cls=None):
    if cfg_path is None:
        if cls is not None:
            return None, {}
        else:
            return {}
    
    cfg = load_yaml_with_base(cfg_path)
    params = cfg["params"]
    if cls is not None:
        validate_params(cls, params)
        return cls(**params), params
    else:
        return params