# lists of available checkpoints
# you will need to customise this file to match your local setup

# the format for a checkpoint 'object' is a tuple -
#   - shortname
#   - path to checkpoint, relative to the comfy checkpoint folder
#   - path to a lora, if any - applied at 0.7/0.7 strength
#   - positive prompt prefix
#   - negative prompt prefix
#   - path to VAE (required)

everything = [
    ('pony', 'pony/ponyDiffusionV6XL_v6StartWithThisOne.safetensors', None, 'score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up', '', 'sdxl_vae.safetensors'),
    ('sxzluma_real', 'pony/sxzLuma_a3PDXLVAE.safetensors', None, 'score_9, score_8_up, score_7_up, score_6_up, realistic', 'score_5, score_4, ps1 screenshot, 3d, niji, wrinkled skin, double head, mutant, bad anatomy, fewer digits, bad hands, extra limbs, twisted, cropped, out of frame', 'sdxl_vae.safetensors'),
]

# example of creating 'checkpoint' tuples from the same checkpoint but with different prompts
sxz_styles = ['photo, realistic, cinematic', 'scenery, environment']
for s in sxz_styles:
    tag = s.split(', ')[0].replace(' ', '_')
    tag = tag.replace('\\(', '').replace('\\)', '')
    everything.append((f'sxzluma_real_{tag}', 'pony/sxzLuma_a3PDXLVAE.safetensors', None, f'score_9, score_8_up, score_7_up, score_6_up, realistic, {s}', 'score_5, score_4, ps1 screenshot, 3d, niji, wrinkled skin, double head, mutant, bad anatomy, fewer digits, bad hands, extra limbs, twisted, cropped, out of frame', 'sdxl_vae.safetensors'))
    everything.append((f'sxzluma_anime_{tag}', 'pony/sxzLuma_a3PDXLVAE.safetensors', None, f'score_9, score_8_up, score_7_up, score_6_up, anime, {s}', 'score_5, score_4, ps1 screenshot, 3d, niji, wrinkled skin, double head, mutant, bad anatomy, fewer digits, bad hands, extra limbs, twisted, cropped, out of frame', 'sdxl_vae.safetensors'))

# using a style lora
everything.extend([
    ('pony-sty-rainbow', 'pony/ponyDiffusionV6XL_v6StartWithThisOne.safetensors', 'pony/style/Rainbow Style SDXL_LoRA_Pony Diffusion V6 XL.safetensors', 'score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up', '', 'sdxl_vae.safetensors'),
])

# dictionary form, indexed by shortname
everything_d = {x[0]: x for x in everything}

# paths for llm components, if you use them
llama_model_path = '/mnt/z/comfy/ComfyUI/custom_nodes/ComfyUI-LLaVA-Captioner/models/llama/llava-v1.5-7b-Q4_K.gguf'
llama_clip_path = '/mnt/z/comfy/ComfyUI/custom_nodes/ComfyUI-LLaVA-Captioner/models/llama/llava-v1.5-7b-mmproj-Q4_0.gguf'

