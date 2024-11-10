import argparse
import gc
import numpy as np
import os
import random
import re
import sys
import torch

import comfy.utils

from datetime import datetime

# boilerplate generated by the comfy python extension
# recursively scan upwards for an item
def find_path(name: str, path: str = None) -> str:
    if path is None:
        path = os.getcwd()
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        return path_name
    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None
    return find_path(name, parent_directory)

def add_extra_model_paths() -> None:
    from main import load_extra_path_config
    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)

def import_custom_and_start() -> None:
    import asyncio
    import execution
    import server
    # FIXME - this was using the wrong name
    from nodes import init_custom_nodes as init_extra_nodes
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    init_extra_nodes()

comfy_path = find_path('ComfyUI')
sys.path.append(comfy_path)

this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_path)
import common
import checkpoints
import llm
from common import Silence
from common import WeightedList

def main(checkpoint_list, input_prompts, noise = 0.0, sizes = [(1024, 1024), (832, 1216), (1216, 832)], rescale=1.0, rescale_denoise=0.4, skip_original_face = False, additional_detailer_seeds = [], skip_detailing = False, detailer_selector = -1, use_dtg = False, dtg_rating = 'safe', dtg_target = '<|long|>', dtg_temperature = 0.7, banlist = [], rerun_lora = False, fd_checkpoint = None, frontload_tags = ['rating_safe'], frontload_neg = [], mandatory_tags = [], seeds = [-1], diffusion_start = None, diffusion_stop = None, save_ora = False, output_folder='.', holdfile_path=None):
    with Silence():
        add_extra_model_paths()
        import_custom_and_start()
    from nodes import NODE_CLASS_MAPPINGS

    with torch.inference_mode():
        with Silence():
            clipEncoderClass = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            cplsClass = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
            fdpClass = NODE_CLASS_MAPPINGS["FaceDetailerPipe"]()
            freeuClass = NODE_CLASS_MAPPINGS["FreeU_V2"]()
            imageSaverClass = NODE_CLASS_MAPPINGS["Image Saver"]()
            kSamplerClass = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]()
            latentClass = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
            loraLoaderClass = NODE_CLASS_MAPPINGS["LoraLoader"]()
            ultraProviderClass = NODE_CLASS_MAPPINGS["UltralyticsDetectorProvider"]()
            vaeLoaderClass = NODE_CLASS_MAPPINGS["VAELoader"]()

            ultra_provider = ultraProviderClass.doit(model_name="bbox/face_yolov8m.pt")[0]

        # optional fd_checkpoint
        if fd_checkpoint:
            fdc_sname, fdc_name, fdc_lora, fdc_pos, fdc_neg, fd_vae_name = fd_checkpoint
            fdc_model, fdc_clip, _ = cplsClass.load_checkpoint(ckpt_name=fdc_name)
            fdc_positive_cond = common.scramble_embedding(clipEncoderClass.encode(text = fdc_pos, clip=fdc_clip)[0], noise)
            fdc_negative_cond = common.scramble_embedding(clipEncoderClass.encode(text = fdc_neg, clip=fdc_clip)[0], noise)
            fdc_vae = vaeLoaderClass.load_vae(vae_name=fd_vae_name)[0]

        # the intermingling of loras and prompts has been a disaster for the human race
        prompt_buckets = {'base': {'loras': [], 'prompts': []}}
        for prompt in input_prompts:
            positive_prompt, loras = common.extract_lora_keywords(prompt)
            if loras:
                bucket = '|'.join(f'{x}:{y}:{z}' for x, y, z in loras)
                if bucket not in prompt_buckets:
                    prompt_buckets[bucket] = {'loras': loras, 'prompts': [prompt]}
                else:
                    prompt_buckets[bucket]['prompts'].append(prompt)
            if not loras or rerun_lora:
                prompt_buckets['base']['prompts'].append(positive_prompt)
        
        if not prompt_buckets['base']['prompts']:
            del prompt_buckets['base']
        
        for pb, bucket_prompts in prompt_buckets.items():
            common.log(f'** processing prompts from prompt bucket {pb}')
            for checkpoint_tuple in checkpoint_list:
                if not checkpoint_tuple:
                    checkpoint_tuple = random.choice(list(checkpoints.everything_d.values()))
                common.log(f'** selected {checkpoint_tuple}')
                checkpoint_shortname, checkpoint, lora, positive_stem, negative_stem, vae_name = checkpoint_tuple

                positive_stem = WeightedList(positive_stem)
                negative_stem = WeightedList(negative_stem)

                for cpe in frontload_tags:
                    positive_stem.parse(cpe)
                for cpe in frontload_neg:
                    negative_stem.parse(cpe)

                base_model, clip_object, _ = cplsClass.load_checkpoint(ckpt_name=checkpoint)
                vae = vaeLoaderClass.load_vae(vae_name=vae_name)[0]

                # lora manglement
                lora_list = ([(lora, 0.7, 0.7)] if lora else []) + bucket_prompts['loras'] 
                for lora, ms, cs in lora_list:
                    common.log(f'** patching model with lora:{lora}:{ms}:{cs}')
                    lora_result = loraLoaderClass.load_lora(lora_name = lora, strength_model = ms, strength_clip = cs, model = base_model, clip = clip_object)
                    base_model = lora_result[0]
                    clip_object = lora_result[1]

                # freeu version of the base model
                free_model = freeuClass.patch(b1=1.3, b2=1.4, s1=0.9, s2=0.2, model=base_model)[0]

                # set clip skip
                clip_object.clip_layer(-2)

                # create "empty" conditionings for face detailing
                positive_d_cond = common.scramble_embedding(clipEncoderClass.encode(text = positive_stem.to_string(suppress_lora=True), clip=clip_object)[0], noise)
                negative_d_cond = common.scramble_embedding(clipEncoderClass.encode(text = negative_stem.to_string(suppress_lora=True), clip=clip_object)[0], noise)
            
                for (prompt_idx, prompt) in enumerate(bucket_prompts['prompts']):
                    common.log(f'++ prompt {prompt_idx} - {prompt}')

                    positive_prompt_without_stem = WeightedList()
                    for cpe in frontload_tags:
                        positive_prompt_without_stem.parse(cpe)
                    positive_prompt_without_stem.parse(prompt)
                                      
                    for dimensions in sizes:
                        local_positive_prompt_without_stem = WeightedList(positive_prompt_without_stem)
                        if use_dtg:
                            local_positive_prompt_without_stem.sort()
                            results = llm.runDTGPromptWrapper(
                                local_positive_prompt_without_stem.get_keys(suppress_lora=True), 
                                dtg_rating, 
                                dimensions[0], 
                                dimensions[1], 
                                dtg_target, 
                                banlist,
                                dtg_temperature
                            )
                            common.log(f'** dtg returned the following tags, which will be added to the originals: {results[2]}')
                            local_positive_prompt_without_stem.parse(', '.join(results[2]))

                        for mandatory in mandatory_tags:
                            if '|' in mandatory:
                                trigger, piece = mandatory.split('|', maxsplit=1)
                                if trigger in local_positive_prompt_without_stem:
                                    common.log(f'** mandatory term trigger \'{trigger}\' seen, injecting \'{piece}\'')
                                    local_positive_prompt_without_stem.parse(piece)
                            else:
                                local_positive_prompt_without_stem.parse(mandatory)

                        # final assembly
                        local_positive_prompt_without_stem.sort()
                        final_positive_prompt = WeightedList(positive_stem)
                        final_positive_prompt.extend(local_positive_prompt_without_stem)
                        final_positive_prompt.consolidate_keys(lambda x: max(x))
                        common.log(f'** positive prompt: {final_positive_prompt.to_string()}')

                        positive_cond = common.scramble_embedding(clipEncoderClass.encode(text=final_positive_prompt.to_string(suppress_lora=True), clip=clip_object)[0], noise)

                        input_latent = latentClass.generate(width=dimensions[0], height=dimensions[1], batch_size=1)[0]
                        for seed in seeds:
                            if(diffusion_start and diffusion_stop): common.sleep_while_outside(diffusion_start, diffusion_stop)
                            common.sleep_while_holdfile(holdfile_path)
                            if seed == -1: seed = common.rseed(seed)
                            ksampler_samples = kSamplerClass.sample(
                                model=free_model,
                                add_noise='enable',
                                noise_seed=seed,
                                steps=25,
                                cfg=6.5,
                                sampler_name="euler_ancestral",
                                scheduler="normal",
                                positive=positive_cond,
                                negative=negative_d_cond,
                                latent_image=input_latent,
                                start_at_step=0,
                                end_at_step=10000,
                                return_with_leftover_noise='disabled',
                                denoise=1.0
                            )[0]['samples']
                            sampled_images = [vae.decode(ksampler_samples)]

                            if rescale != 1.0:
                                nw = round(ksampler_samples.shape[3] * rescale)
                                nh = round(ksampler_samples.shape[2] * rescale)
                                upscaled_latent = {'samples': comfy.utils.common_upscale(ksampler_samples, nw, nh, 'bicubic', 'disabled')}
                                upscaled_samples = kSamplerClass.sample(
                                    model=free_model,
                                    add_noise='enable',
                                    noise_seed=seed,
                                    steps=12,
                                    cfg=6.5,
                                    sampler_name="euler_ancestral",
                                    scheduler="normal",
                                    positive=positive_cond,
                                    negative=negative_d_cond,
                                    latent_image=upscaled_latent,
                                    start_at_step=0,
                                    end_at_step=10000,
                                    return_with_leftover_noise='disabled',
                                    denoise=rescale_denoise
                                )[0]['samples']
                                sampled_images.append(vae.decode(upscaled_samples))

                            detailer_images = []

                            if not skip_detailing:
                                # the pipe packing code didn't import cleanly, but detailer pipes are just 14-tuples:
                                #  model, clip, vae, positive, negative, wildcard, bbox, 
                                #  segm_detector_opts / None, sam_model_opt / None, detailer_hook / None, 
                                #  refiner_model / None, refiner_clip / None, refiner_pos / None, refiner_neg / None
                                
                                detailer_pipes = [
                                    (base_model, clip_object, vae, positive_cond, negative_d_cond, '', ultra_provider, None, None, None, None, None, None, None),
                                    (base_model, clip_object, vae, positive_d_cond, negative_d_cond, '', ultra_provider, None, None, None, None, None, None, None),
                                    (free_model, clip_object, vae, positive_cond, negative_d_cond, '', ultra_provider, None, None, None, None, None, None, None),
                                    (free_model, clip_object, vae, positive_d_cond, negative_d_cond, '', ultra_provider, None, None, None, None, None, None, None),
                                ]
                                # an additional sd15 face detail if requested
                                if fd_checkpoint:
                                    detailer_pipes.append(
                                        (fdc_model, fdc_clip, fdc_vae, fdc_positive_cond, fdc_negative_cond, '', ultra_provider, None, None, None, None, None, None, None),
                                    )
                                if detailer_selector > -1:
                                    detailer_pipes = [detailer_pipes[detailer_selector]]
                                for out_image_idx, detailer_pipe in enumerate(detailer_pipes):
                                    for detailer_seed in [seed] + additional_detailer_seeds:
                                        detailer_seed = common.rseed(detailer_seed)
                                        common.log(f'** running detailer pipe {out_image_idx+1} of {len(detailer_pipes)} with seed {detailer_seed}')
                                        face_detailer = fdpClass.doit(
                                            guide_size=384,
                                            guide_size_for=True,
                                            max_size=1024,
                                            seed=detailer_seed,
                                            steps=20,
                                            cfg=7.5,
                                            sampler_name="euler",
                                            scheduler="normal",
                                            denoise=0.35,
                                            feather=5,
                                            noise_mask=True,
                                            force_inpaint=False,
                                            bbox_threshold=0.5,
                                            bbox_dilation=10,
                                            bbox_crop_factor=3,
                                            sam_detection_hint="center-1",
                                            sam_dilation=0,
                                            sam_threshold=0.93,
                                            sam_bbox_expansion=0,
                                            sam_mask_hint_threshold=0.70,
                                            sam_mask_hint_use_negative="False",
                                            drop_size=10,
                                            refiner_ratio=0.2,
                                            cycle=2,
                                            inpaint_model=False,
                                            noise_mask_feather=0,
                                            image=sampled_images[-1],
                                            detailer_pipe=detailer_pipe,
                                        )
                                        detailer_images.append(face_detailer[0])

                            if not detailer_images:
                                output_images = sampled_images
                            elif skip_original_face:
                                output_images = detailer_images
                            else:
                                output_images = sampled_images + detailer_images

                            # deduplicate
                            unique = []
                            for t in output_images:
                                if not any(torch.equal(t, ut) for ut in unique):
                                    unique.append(t)
                            # edge case where skip original is on but no detections
                            if not unique:
                                unique = sampled_images

                            common.log(f'** writing {len(unique)} images')
                            for (out_image_idx, out_img) in enumerate(unique):
                                # half of this metadata is wrong but a1111 doesn't really understand multistage generation
                                out_height = out_img.shape[1]
                                out_width = out_img.shape[2]
                                saver = imageSaverClass.save_files(
                                    filename=f't2i_{checkpoint_shortname}_{prompt_idx}_%time_%seed_{out_image_idx}',
                                    path=output_folder,
                                    extension="png",
                                    steps=25,
                                    cfg=7.5,
                                    modelname=checkpoint,
                                    sampler_name="euler_ancestral",
                                    scheduler="normal",
                                    positive=final_positive_prompt.to_string(),
                                    negative=negative_stem.to_string(),
                                    seed_value=seed,
                                    width=out_width,
                                    height=out_height,
                                    lossless_webp=True,
                                    quality_jpeg_or_webp=100,
                                    optimize_png=False,
                                    counter=0,
                                    denoise=1,
                                    time_format="%Y-%m-%d-%H%M%S",
                                    save_workflow_as_json=False,
                                    embed_workflow_in_png=False,
                                    strip_a1111_params="nothing",
                                    images=out_img,
                                )

                            if save_ora:
                                common.save_images_to_ora(sampled_images[-1], detailer_images, f'{output_folder}/ora_t2i_{checkpoint_shortname}_{prompt_idx}_{datetime.now().strftime("%Y-%m-%d-%H%M%S")}_{seed}_{out_image_idx}.ora')
                                
                del base_model, clip_object, free_model
                gc.collect()
                torch.cuda.empty_cache()

def parse_args():
    acceptable_checkpoints = list(checkpoints.everything_d.keys()) + ['*']

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', choices=acceptable_checkpoints, default=['*'], help="List of checkpoints. Default is ['*'].")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prompts', nargs='+', help="List of prompts")
    group.add_argument('--prompt_file', type=common.args_read_prompts_from_file, help="File with prompts (one per line)")

    parser.add_argument('--sizes', nargs='+', type=common.args_parse_int_tuple, default=[(1024, 1024), (832, 1216), (1216, 832)], help="List of sizes as 'width,height'. Default is [(1024,1024), (832,1216), (1216,832)]")

    parser.add_argument('--skip_original_face', action='store_true', default=False, help="Don't include original face in the outputs. Default is False.")
    parser.add_argument('--skip_detailing', action='store_true', default=False, help="Skip detailing. Default is False.")
    parser.add_argument('--single_detailer', type=int, default=-1, help="Select only a single detailing pipe. Requires knowledge about indexes in code.")
    parser.add_argument('--rerun_lora', action='store_true', default=False, help="Rerun LoRA infected prompts without their LoRA. Default is False.")

    parser.add_argument('--use_dtg', action='store_true', default=False, help = "Enable the DTG tag extension LLM.")
    parser.add_argument('--dtg_rating', default='safe', help="Set desired prompt safety for DTG.")
    parser.add_argument('--dtg_target', choices=['<|very_short|>', '<|short|>', '<|long|>', '<|very_long|>'], default='<|long|>', help="Set desired prompt length for DTG.")
    parser.add_argument('--dtg_temperature', type=float, default=0.5, help="Set inference temperature for DTG.")
    parser.add_argument('--banlist', type=str, nargs='*', default=common.tag_banlist, help='Tags that will be excluded from wd14 tagging.')

    parser.add_argument('--save_ora', action='store_true', default=False, help="Save ORA file after detailing. Default is False.")

    parser.add_argument('--rescale', type=float, default=1.0, help="HRFix. Defaults to 1.0 (disabled).")
    parser.add_argument('--rescale_denoise', type=float, default=0.4, help="HRFix denoise. Defaults to 0.4 (disabled).")
    parser.add_argument('--noise', type=float, default=0, help="Noise strength, applied to all embeddings. Default 0.0.")

    parser.add_argument('--fd_checkpoint', choices=acceptable_checkpoints, default=None, help="fd_checkpoint. Default is None.")

    parser.add_argument('--frontload_tags', nargs='+', default=['rating_safe'], help="Frontload tags. Default is ['rating_safe'].")
    parser.add_argument('--frontload_neg', nargs='+', default=[], help="Frontload negative tags. Default is [].")
    parser.add_argument('--mandatory_tags', nargs='+', default=[], help="Mandatory tags. Default is [].")

    parser.add_argument('--seeds', nargs='+', type=int, default=[-1], help="List of seeds. Default is [-1]. -1 is reevaluated each time, use -2 for a 'fixed random' seed.")
    parser.add_argument('--additional_detailer_seeds', nargs='*', type=int, default=[], help="List of additional detailer seeds. Default is [].")

    parser.add_argument('--diff_start', type=common.args_validate_time, default=None, help="Diffusion start time in 'HH:MM' format.")
    parser.add_argument('--diff_stop', type=common.args_validate_time, default=None, help="Diffusion end time in 'HH:MM' format.")

    parser.add_argument('--output_folder', type=str, default='.', help="Output folder path. Default is '.', your comfy output directory.")
    parser.add_argument('--hold_file', type=str, default='hold.txt', help="Holdfile path. Default is 'hold.txt'.")

    parser.add_argument('--endless', action='store_true', default=False, help="Run forever.")
    parser.add_argument('--endless_sleep', type=float, default=0, help="Run forever delay.")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    common.log(args)

    checkpoint_list = [checkpoints.everything_d[x] if x != '*' else None for x in args.checkpoints ]
    
    if args.fd_checkpoint and args.fd_checkpoint != '*':
        fd_checkpoint = checkpoints.everything_d[args.fd_checkpoint]
    elif args.fd_checkpoint == '*':
        fd_checkpoint = random.choice(checkpoints.everything)
    else:
        fd_checkpoint = None

    # -2 is a "fixed random seed" which is generated once and then stays the same
    # -1 is the classic random seed which regenerates every time it is called on
    seeds = [common.rseed(-1) if x == -2 else x for x in args.seeds]

    if args.prompts:
        input_prompts = args.prompts
    else:
        input_prompts = args.prompt_file

    # prevent comfy from complaining about my args
    sys.argv = [sys.argv[0]]

    while True:
        main(
            checkpoint_list, 
            input_prompts, 
            sizes=args.sizes,
            seeds=seeds,
            noise=args.noise,
            additional_detailer_seeds=args.additional_detailer_seeds,
            skip_original_face=args.skip_original_face, 
            skip_detailing=args.skip_detailing,
            detailer_selector=args.single_detailer,
            rerun_lora=args.rerun_lora,
            use_dtg=args.use_dtg,
            dtg_rating=args.dtg_rating,
            dtg_target=args.dtg_target,
            dtg_temperature=args.dtg_temperature,
            banlist=args.banlist,
            rescale=args.rescale,
            rescale_denoise=args.rescale_denoise,
            fd_checkpoint=fd_checkpoint, 
            frontload_tags = args.frontload_tags,
            frontload_neg = args.frontload_neg,
            mandatory_tags = args.mandatory_tags,
            diffusion_start = args.diff_start,
            diffusion_stop = args.diff_stop,
            save_ora = args.save_ora,
            holdfile_path = args.hold_file,
            output_folder = args.output_folder
        )
        if not args.endless:
            break
        else:
            import time
            time.sleep(args.endless_sleep)
