# various stock functions
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image, ImageOps
import argparse
import base64
import io
import numpy as np
import os
import random
import re
import sys
import time
import torch

# miscellaneous functions and a class to do with logging or silencing noisy components
def log(message):
    print(datetime.now().strftime("%Y%m%d %H:%M:%S:"), message)

class Silence:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr

# data for use with the tagging node - you will need to customise this
tag_banlist = ['english_text',]

# functions for adding noise to embeddings
def scramble_embedding(clip_output, noise):
    if noise == 0:
        return clip_output
    log(f'fuzzing clip object @ {id(clip_output)}, strength: {noise}.')
    embedding = clip_output[0][0]
    embed_min, embed_max = torch.min(embedding), torch.max(embedding)
    log(f' embedding ranged from {embed_min:.2f} to {embed_max:.2f}')
    pooled_out = clip_output[0][1]['pooled_output']
    pooled_min, pooled_max = torch.min(pooled_out), torch.max(pooled_out)
    log(f' pooled_out ranged from {pooled_min:.2f} to {pooled_max:.2f}')
    embedding += (noise * torch.randn_like(embedding))
    pooled_out += (noise * torch.randn_like(pooled_out))
    return [[embedding, {'pooled_output': pooled_out}]]

# functions for handling conversions between the comfy IMAGE tensor type, images on disk, PIL images etc
def image_tensor_to_pil_rgba(tensor, alpha_value = 1.0):
    tensor = tensor.squeeze(0)
    alpha_channel = torch.full(tensor.shape[:-1] + (1,), alpha_value)
    image_with_alpha = torch.cat([tensor, alpha_channel], dim=-1)
    image_np = image_with_alpha.numpy()
    image_np = (image_np * 255).astype(np.uint8)
    pil_version = Image.fromarray(image_np, mode='RGBA')
    return pil_version

def image_tensor_to_png_url(image_tensor):
    image_tensor = image_tensor.squeeze(0)
    im_np = image_tensor.numpy()
    im_np = (im_np * 255).astype(np.uint8)
    im_pi = Image.fromarray(im_np)
    buffer = io.BytesIO()
    im_pi.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{encoded}'

def load_image_as_image_tensor(fn):
    i = Image.open(fn)
    i = ImageOps.exif_transpose(i)
    im = i.convert('RGB')
    im = np.array(im).astype(np.float32) / 255.0
    # add an extra dimension, taking this to [1, h, w, 3]
    im = torch.from_numpy(im)[None,]
    return (im, i.info)

# scale images so they are within 10% or so of common sdxl pixel counts (1024x1024, 832 x 1216, 1216 x 832, etc)
def scale_image_to_common_size(im):
    import comfy.utils
    # [1, h, w, 3] -> [1, 3, h, w]
    samples = im.movedim(-1, 1)
    width = samples.shape[3]
    height = samples.shape[2]
    pixels = width * height

    # standard sdxl image resolutions -
    pixel_standards = [1024 * 1024, 832 * 1216]

    # test whether in tolerance
    if(min([x * 0.9 for x in pixel_standards]) <= pixels <= max([x * 1.1 for x in pixel_standards])):
        return (im, width, height)
    
    # otherwise, we rescale
    nearest_standard = min(pixel_standards, key=lambda p: abs(p-pixels)) 
    scale_factor = (nearest_standard / pixels) ** 0.5

    new_width = round(width * scale_factor)
    new_height = round(height * scale_factor)

    scaled = comfy.utils.common_upscale(samples, new_width, new_height, 'lanczos', 'disabled')
    # [1, 3, h, w] -> [1, h, w, 3]
    scaled = scaled.movedim(1, -1)
    return (scaled, new_width, new_height)

# functions for writing out image variants to ORA layeredimages files
# requires the installation of the layeredimage package, which can interact
# badly with some users' comfyui installations
def save_images_to_ora(base, layers, fn, layernames = None):
    import layeredimage.io
    import layeredimage.layeredimage
    import layeredimage.layergroup
    ora_layers = [layeredimage.layergroup.Layer('base', image_tensor_to_pil_rgba(base))]
    for idx, detailer_i in enumerate(layers):
        if layernames:
            layername = layernames[idx]
        else:
            layername = f'detailer-{idx}'
        ora_layers.append(layeredimage.layergroup.Layer(layername, diff_image_tensors_rgba(base, detailer_i)))                        
    ora = layeredimage.layeredimage.LayeredImage(layersAndGroups=ora_layers)
    layeredimage.io.saveLayerImage(fn, ora)

# determine the differences between two image tensors, used with the above function
def diff_image_tensors_rgba(tensor_ground, tensor_altered, threshold=0):
    tensor_ground = tensor_ground.squeeze(0)
    tensor_altered = tensor_altered.squeeze(0)
    diff = torch.abs(tensor_altered - tensor_ground)
    mask = torch.sum(diff, dim=-1) > threshold
    mask_rgb = mask.unsqueeze(-1).repeat(1, 1, 3)
    tensor_difference = torch.where(mask_rgb, tensor_altered, torch.tensor([0.5, 0.0, 0.0]))  # Black for unchanged
    alpha_channel = (mask).float()
    tensor_difference_a = torch.cat([tensor_difference, alpha_channel.unsqueeze(-1)], dim=-1)
    np_difference = tensor_difference_a.numpy()
    np_difference = (np_difference * 255).astype(np.uint8)
    pil_difference = Image.fromarray(np_difference, mode='RGBA')
    return pil_difference

# functions to sleep, either while a file exists or until a certain window of time
def sleep_while_holdfile(holdfile = None) -> None:
    import os.path
    if holdfile == None:
        return
    if not os.path.exists(holdfile):
        return
    log(f'sleeping for holdfile {holdfile}')
    while os.path.exists(holdfile):
        time.sleep(60)
    log(f'waking from holdfile {holdfile}')

def sleep_while_outside(start_time_str, end_time_str) -> None:
    start_time = datetime.strptime(start_time_str, "%H:%M").time()
    end_time = datetime.strptime(end_time_str, "%H:%M").time()
    
    now = datetime.now()
    current_time = now.time()

    # e.g. 06:00 -> 23:00
    if start_time <= end_time:
        if current_time < start_time:
            target_datetime = datetime.combine(now, start_time)
        elif current_time > end_time:
            target_datetime = datetime.combine(now + timedelta(days=1), start_time)
        else:
            target_datetime = None
    # e.g. 23:00 -> 06:00
    else:
        if current_time < start_time and current_time > end_time:
            target_datetime = datetime.combine(now, start_time)
        else:
            target_datetime = None

    if(not target_datetime):
        return

    time_to_sleep = target_datetime - now
    hours, remainder = divmod(time_to_sleep.total_seconds(), 3600)
    minutes, _ = divmod(remainder, 60)   
    print(f"** sleeping {int(hours):02}:{int(minutes):02} until {target_datetime.strftime('%H:%M')}.")
    time.sleep(time_to_sleep.total_seconds())

# allows folders to be given as relative paths if base_folder is provided
def list_files_in_folders(folders, base_folder = None, extensions = ['.png', '.jpg']):
    result = []
    base_path = Path(base_folder) if base_folder else None
    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.is_absolute() and base_path:
            folder_path = base_path / folder_path
        
        if folder_path.is_dir():
            for file_path in folder_path.iterdir():
                if file_path.is_file() and file_path.suffix in extensions:
                    result.append(file_path)
        else:
            print(f"Warning: {folder_path} is not a valid directory.")   
    return result

# parse and extract keywords in the form <lora:name:model_strength:clip_strength>, including removing them from the original input prompt
def extract_lora_keywords(input_str):
    # Define regex to capture text in angle brackets and optional numbers separated by colons
    pattern = re.compile(r'<lora:([^:>]+)(?::([\d.]+))?(?::([\d.]+))?>')
    
    result_str = input_str
    matches = []
    
    # Find all matches
    for match in pattern.finditer(input_str):
        name = f'{match.group(1)}.safetensors'
        num1 = float(match.group(2)) if match.group(2) else 1.0
        num2 = float(match.group(3)) if match.group(3) else num1
        matches.append((name, num1, num2))
        result_str = result_str.replace(match.group(0), "")
    
    pieces = [(re.sub(r'\s+', ' ', x)).strip() for x in result_str.split(',')]
    result_str = ', '.join(x for x in pieces if x)

    return result_str, matches

# randomise an int if -1, or just return it otherwise
def rseed(inseed=-1):
    if inseed == -1:
        return random.randint(1, 1125899906842624)
    return inseed

# functions for validating argparse arguments
def args_valid_file_path(path_str):
    path = Path(path_str)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"The provided path, '{path}', does not exist.")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"The provided path, '{path}', is not a file.")
    return path

def args_validate_time(value):
    """Ensure time is in HH:MM format."""
    if not re.match(r'^\d{2}:\d{2}$', value):
        raise argparse.ArgumentTypeError(f"Invalid time format: {value}. Expected format is 'HH:MM'.")
    return value

def args_read_prompts_from_file(filename):
    """Read prompts from a file, one per line."""
    if not os.path.isfile(filename):
        raise argparse.ArgumentTypeError(f"File {filename} not found.")
    with open(filename, 'r') as file:
        return [line.strip() for line in file if not line.strip().startswith('#')]

def args_parse_int_tuple(value):
    """Parse a string representing a tuple in the form 'int,int' into a 2-tuple of integers."""
    try:
        x, y = map(int, value.split(','))
        return (x, y)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid tuple format: '{value}', expected format: 'int,int'")

def args_parse_float_tuple(value):
    """Parse a string representing a tuple in the form 'int,int' into a 2-tuple of integers."""
    try:
        x, y = map(float, value.split(','))
        return (x, y)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid tuple format: '{value}', expected format: 'float,float'")

def args_parse_bounding_box(value):
    """Parse a string representing a tuple in the form 'left:top+width+height' into a 4-tuple of integers."""
    if value == 'auto':
        return value
    boxes = []
    for part in value.split():
        try:
            left_top, size = part.split('+', 1)
            left, top = map(int, left_top.split(':'))
            width, height = map(int, size.split('+'))
            boxes.append((left, top, width, height))
        except:
            stock.log(f'xx error parsing bbox {part}')
            pass
    return boxes

