import argparse
import multiprocessing
import os
import random
import re
import sys
import torch

import comfy.utils

from datetime import datetime
from pathlib import Path

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

def main(checkpoint_list, input_files, force_prompt = None, from_step = [0], denoise_programs = [(0.4, 1.0)], noise = 0.0, skip_detailing = False, detailer_selector = -1, use_llm = False, llm_prompt = None, llm_sysmessage = None, use_dtg = False, dtg_rating = 'safe', dtg_target = '<|long|>', dtg_temperature = 0.7, degrade_prompt=0.8, skip_original_face = False, fd_checkpoint = None, use_tagger = True, banlist = [], frontload_tags = ['rating_safe'], frontload_neg = [], mandatory_tags = [], seeds = [-1], additional_detailer_seeds = [], skip = 0, sleep_start = None, sleep_stop = None, output_folder='.', holdfile_path=None, save_ora = False):
    with Silence():
        add_extra_model_paths()
        import_custom_and_start()
    from nodes import NODE_CLASS_MAPPINGS
    current_index = 0
    maximum_index = len(checkpoint_list) * len(input_files) * len(from_step) * len(denoise_programs) * len(seeds) 
    maximum_index_pad = len(str(maximum_index))
    common.log(f'~ > maximum index is: {maximum_index}')

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
            VAEEncoderClass = NODE_CLASS_MAPPINGS["VAEEncode"]()
            vaeLoaderClass = NODE_CLASS_MAPPINGS["VAELoader"]()
            wd14TaggerClass = NODE_CLASS_MAPPINGS["WD14Tagger|pysssss"]()

            ultra_provider = ultraProviderClass.doit(model_name="bbox/face_yolov8m.pt")[0]

        # optional fd_checkpoint
        if fd_checkpoint:
            _, fdc_name, fdc_lora, fdc_pos, fdc_neg, fdc_vae = fd_checkpoint
            fdc_model, fdc_clip, _ = cplsClass.load_checkpoint(ckpt_name=fdc_name)
            fdc_positive_cond = common.scramble_embedding(clipEncoderClass.encode(text = fdc_pos, clip=fdc_clip)[0], noise)
            fdc_negative_cond = common.scramble_embedding(clipEncoderClass.encode(text = fdc_neg, clip=fdc_clip)[0], noise)
            fdc_vae = vaeLoaderClass.load_vae(vae_name=fdc_vae)[0]

        for checkpoint_tuple in checkpoint_list:
            if not checkpoint_tuple:
                checkpoint_tuple = random.choice(list(checkpoints.everything_d.values()))
            checkpoint_shortname, checkpoint, lora, positive_stem, negative_stem, vae_name = checkpoint_tuple

            positive_stem = WeightedList(positive_stem)
            negative_stem = WeightedList(negative_stem)

            child_combinations = len(input_files) * len(from_step) * len(denoise_programs) * len(seeds)
            if(current_index + child_combinations <= skip):
                common.log(f'** skipping checkpoint {checkpoint_shortname}')
                current_index += child_combinations
                continue

            for cpe in frontload_tags:
                positive_stem.parse(cpe)
            for cpe in frontload_neg:
                negative_stem.parse(cpe)

            common.log(f'** loading checkpoint {checkpoint_shortname}')
            # checkpoint
            base_model, clip_object, _ = cplsClass.load_checkpoint(ckpt_name=checkpoint)
            vae = vaeLoaderClass.load_vae(vae_name=vae_name)[0]

            # lora
            if(lora):
                common.log(f'** patching checkpoint with lora {lora}')
                lora_result = loraLoaderClass.load_lora(lora_name = lora, strength_model = 0.7, strength_clip = 0.7, model = base_model, clip = clip_object)
                base_model = lora_result[0]
                clip_object = lora_result[1]
            # freeu version
            free_model = freeuClass.patch(b1=1.3, b2=1.4, s1=0.9, s2=0.2, model=base_model)[0]
            # set clip skip
            clip_object.clip_layer(-2)

            # 'basic' conditionings for this model
            positive_d_cond = common.scramble_embedding(clipEncoderClass.encode(text = positive_stem.to_string(), clip=clip_object)[0], noise)
            negative_d_cond = common.scramble_embedding(clipEncoderClass.encode(text = negative_stem.to_string(), clip=clip_object)[0], noise)
          
            for image_fn in input_files:
                child_combinations = len(from_step) * len(denoise_programs) * len(seeds)
                if(current_index + child_combinations <= skip):
                    common.log(f'** skipping image {image_fn}')
                    current_index += child_combinations
                    continue

                input_image, metadata = common.load_image_as_image_tensor(image_fn)
                input_image, image_width, image_height = common.scale_image_to_common_size(input_image)

                positive_prompt_without_stem = WeightedList()
                for cpe in frontload_tags:
                    positive_prompt_without_stem.parse(cpe)
                negative_prompt_without_stem = WeightedList()
                for cpe in frontload_neg:
                    negative_prompt_without_stem.parse(cpe)

                if use_tagger and not force_prompt:
                    with Silence():
                        tagger_tags = wd14TaggerClass.tag(
                            model="wd-v1-4-moat-tagger-v2",
                            threshold=0.35,
                            character_threshold=0.85,
                            replace_underscore=False,
                            trailing_comma=False,
                            exclude_tags=','.join(banlist),
                            image=input_image,
                        )['result'][0][0]                   
                    if tagger_tags:
                        if degrade_prompt < 1:
                            tag_parts = [x.strip() for x in tagger_tags.split(',')]
                            original_length = len(tag_parts)
                            adjusted_length = int(round(original_length * degrade_prompt))
                            if adjusted_length < original_length:
                                common.log(f'** degrading {original_length} tags down to {adjusted_length}')
                                common.log(f'** - tagger tags were {tagger_tags}')
                                indexes_to_remove = set(random.sample(range(original_length), original_length - adjusted_length))
                                tag_parts = [item for idx, item in enumerate(tag_parts) if idx not in indexes_to_remove]
                                tagger_tags = ', '.join(tag_parts)
                                common.log(f'** - tagger tags now {tagger_tags}')
                        positive_prompt_without_stem.parse(tagger_tags)
                elif force_prompt:
                    positive_prompt_without_stem.parse(force_prompt)

                if use_dtg:
                    positive_prompt_without_stem.sort()
                    results = llm.runDTGPromptWrapper(positive_prompt_without_stem.get_keys(), dtg_rating, image_width, image_height, dtg_target, banlist, dtg_temperature)
                    common.log(f'** dtg returned the following tags, which will replace the originals: {results[1]}')
                    positive_prompt_without_stem = WeightedList(', '.join(results[1]))

                supplementary_text = ''

                if use_llm:
                    if positive_prompt_without_stem.get_keys():
                        prompt = llm_prompt + f' To assist you, the image is known to contain the following tags: {positive_prompt_without_stem.to_string_weightless()}.'
                    else:
                        prompt = llm_prompt
                    supplementary_text = llm.runLlamaPromptWrapper(prompt, sys_message = llm_sysmessage, image = input_image).strip()

                for mandatory in mandatory_tags:
                    if '|' in mandatory:
                        trigger, piece = mandatory.split('|', maxsplit=1)
                        if trigger in positive_prompt_without_stem:
                            common.log(f'** mandatory term trigger \'{trigger}\' seen, injecting \'{piece}\'')
                            positive_prompt_without_stem.parse(piece)
                    else:
                        positive_prompt_without_stem.parse(mandatory)
                
                # time to tie it all together
                positive_prompt_without_stem.sort()
                final_positive_prompt = WeightedList(positive_stem)
                final_positive_prompt.extend(positive_prompt_without_stem)
                final_positive_prompt.consolidate_keys(lambda x: max(x))
                
                final_negative_prompt = WeightedList(negative_stem)
                final_negative_prompt.extend(negative_prompt_without_stem)
                final_negative_prompt.consolidate_keys(lambda x: max(x))

                positive_string = f'{final_positive_prompt.to_string()}'
                if supplementary_text:
                    positive_string += f'. {supplementary_text}'
                negative_string = f'{final_negative_prompt.to_string()}'

                positive_cond = clipEncoderClass.encode(text = positive_string, clip=clip_object)[0]
                negative_cond = clipEncoderClass.encode(text = negative_string, clip=clip_object)[0]

                # do our initial denoising over the original image from from_step
                common.log(f'{current_index} < {image_fn}')
                common.log(f'{current_index} + {positive_string}')
                common.log(f'{current_index} - {negative_string}')

                for from_step_value in from_step:
                    child_combinations = len(denoise_programs) * len(seeds)
                    if(current_index + child_combinations <= skip):
                        common.log(f'** skipping from_step value {from_step_value}')
                        current_index += child_combinations
                        continue

                    if from_step_value == 0:
                        input_latent = latentClass.generate(width=image_width, height=image_height, batch_size=1)[0]
                    else:
                        input_latent = VAEEncoderClass.encode(pixels=input_image, vae=vae)[0]

                    for seed in seeds:
                        child_combinations = len(denoise_programs)
                        if(current_index + child_combinations <= skip):
                            common.log(f'** skipping seed {seed}')
                            current_index += child_combinations
                            continue

                        seed=common.rseed(seed)
                        common.log(f'{current_index} ! seed: {seed}, from step: {from_step_value}, denoise programs: {denoise_programs}')
                        if(sleep_start and sleep_stop): common.sleep_while_outside(sleep_start, sleep_stop)
                        common.sleep_while_holdfile(holdfile_path)
                        ksampler_samples = kSamplerClass.sample(
                            model=free_model,
                            add_noise='enable',
                            noise_seed=seed,
                            steps=25,
                            cfg=6.5,
                            sampler_name="euler_ancestral",
                            scheduler="normal",
                            positive=positive_cond,
                            negative=negative_cond,
                            latent_image=input_latent,
                            start_at_step=from_step_value,
                            end_at_step=10000,
                            return_with_leftover_noise='disabled',
                            denoise=1.0
                        )[0]['samples']
                        sampled_images = [vae.decode(ksampler_samples)]
                        detailer_images = []
                        all_images = [sampled_images[-1]]

                        for (upscale_denoise, upscale_scale) in denoise_programs:
                            current_index += 1
                            if(current_index <= skip):
                                common.log(f'** denoise program: upscale to {upscale_scale} and denoise at {upscale_denoise}')
                                continue
                            common.log(f'{current_index} ~ denoise program: x{upscale_scale} ~{upscale_denoise}')
                            file_stub = f'{image_fn.stem[:15]}Z_{current_index:0{maximum_index_pad}}_o_{maximum_index}_{checkpoint_shortname}_sd{from_step_value}_ud{upscale_denoise}_us{upscale_scale}'
                            common.log(f'{current_index} > {file_stub}')
                            if upscale_scale != 1.0:
                                nw = round(ksampler_samples.shape[3] * upscale_scale)
                                nh = round(ksampler_samples.shape[2] * upscale_scale)
                                upscaled_latent = {'samples': comfy.utils.common_upscale(ksampler_samples, nw, nh, 'bicubic', 'disabled')}
                                if(sleep_start and sleep_stop): common.sleep_while_outside(sleep_start, sleep_stop)
                                common.sleep_while_holdfile(holdfile_path)
                                upscaled_samples = kSamplerClass.sample(
                                    model=free_model,
                                    add_noise='enable',
                                    noise_seed=seed,
                                    steps=12,
                                    cfg=6.5,
                                    sampler_name="euler_ancestral",
                                    scheduler="normal",
                                    positive=positive_cond,
                                    negative=negative_cond,
                                    latent_image=upscaled_latent,
                                    start_at_step=0,
                                    end_at_step=10000,
                                    return_with_leftover_noise='disabled',
                                    denoise=upscale_denoise
                                )[0]['samples']
                                sampled_images.append(vae.decode(upscaled_samples))
                                all_images.append(sampled_images[-1])
                            
                            if not skip_detailing:
                                # the pipe packing code didn't import cleanly, but detailer pipes are just 14-tuples:
                                #  model, clip, vae, positive, negative, wildcard, bbox, 
                                #  segm_detector_opts / None, sam_model_opt / None, detailer_hook / None, 
                                #  refiner_model / None, refiner_clip / None, refiner_pos / None, refiner_neg / None
                                detailer_pipes = [
                                    (base_model, clip_object, vae, positive_cond, negative_cond, '', ultra_provider, None, None, None, None, None, None, None),
                                    (base_model, clip_object, vae, positive_d_cond, negative_d_cond, '', ultra_provider, None, None, None, None, None, None, None),
                                    (free_model, clip_object, vae, positive_cond, negative_cond, '', ultra_provider, None, None, None, None, None, None, None),
                                    (free_model, clip_object, vae, positive_d_cond, negative_d_cond, '', ultra_provider, None, None, None, None, None, None, None),
                                ]
                                if fd_checkpoint:
                                    detailer_pipes.extend([
                                        (fdc_model, fdc_clip, fdc_vae, fdc_positive_cond, fdc_negative_cond, '', ultra_provider, None, None, None, None, None, None, None),
                                    ])
                                if detailer_selector > -1:
                                    detailer_pipes = [detailer_pipes[detailer_selector]]
                                for dp_idx, detailer_pipe in enumerate(detailer_pipes):
                                    for d_seed in [seed] + additional_detailer_seeds:
                                        d_seed = common.rseed(d_seed)
                                        if(sleep_start and sleep_stop): common.sleep_while_outside(sleep_start, sleep_stop)
                                        common.sleep_while_holdfile(holdfile_path)
                                        common.log(f'** running detailer pipe {dp_idx+1} of {len(detailer_pipes)} with seed {d_seed}')
                                        face_detailer = fdpClass.doit(
                                            guide_size=384,
                                            guide_size_for=True,
                                            max_size=1024,
                                            seed=d_seed,
                                            steps=20,
                                            cfg=7.5,
                                            sampler_name="euler",
                                            scheduler="normal",
                                            denoise=0.4,
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
                                        all_images.append(face_detailer[0])
                                        detailer_images.append(face_detailer[0])
                            
                        # unique
                        if skip_original_face and detailer_images:
                            candidates = detailer_images
                        else:
                            candidates = all_images
                        unique_candidates = []
                        for t in candidates:
                            if not any(torch.equal(t, ut) for ut in unique_candidates):
                                unique_candidates.append(t)

                        common.log(f'** writing {len(unique_candidates)} images')
                        for (image_idx, output_image) in enumerate(unique_candidates):
                            # half of this metadata is wrong but a1111 doesn't really understand multistage generation
                            out_height = output_image.shape[1]
                            out_width = output_image.shape[2]
                            saver = imageSaverClass.save_files(
                                filename=f'{file_stub}_%time_%seed_{image_idx}',
                                path=output_folder,
                                extension="png",
                                steps=25,
                                cfg=7.5,
                                modelname=checkpoint,
                                sampler_name="euler_ancestral",
                                scheduler="normal",
                                positive=positive_string,
                                negative=negative_string,
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
                                images=output_image,
                            )
                        
                        if save_ora:
                            common.save_images_to_ora(sampled_images[-1], detailer_images, f'{output_folder}/ora_{file_stub}_{datetime.now().strftime("%Y-%m-%d-%H%M%S")}_{seed}.ora')

def parse_args() -> argparse.ArgumentParser:
    acceptable_checkpoints = list(checkpoints.everything_d.keys()) + ['*']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', choices=acceptable_checkpoints, default=['*'], help="List of checkpoints. Default is ['*'].")
    parser.add_argument('--input_filenames', type=common.args_valid_file_path, nargs='+', help="List of images to process.")

    parser.add_argument('--from_step', type=int, nargs='+', default=[14], help="Start denoising the original image from this step. At zero, creates a latent from scratch. Default is 14.")

    parser.add_argument('--denoise_programs', type=common.args_parse_float_tuple, nargs='+', default=[(0.0, 1.0)], help="Upscale denoising profiles - the upscaler will denoise at the strength of the first item, after scaling up by the second (e.g. 0.4,1.1). The default is (0.0, 1.0), which does not upscale.")

    parser.add_argument('--noise', type=float, default=0, help="Noise strength, applied to all embeddings. Default 0.0.")

    parser.add_argument('--skip_detailing', action='store_true', default=False, help="Skip detailing. Default is False.")
    parser.add_argument('--single_detailer', type=int, default=-1, help="Select only a single detailing pipe. Requires knowledge about indexes in code.")

    parser.add_argument('--force_prompt', type=str, default=None, help="Forced prompt, displacing the LLM and tagger.")
    parser.add_argument('--disable_tagger', action='store_true', default=False, help="Disable the wd14 tagger.")
    parser.add_argument('--degrade_prompt', type=float, default=0.8, help="If less than 1 only retain this percent of detected tags. Default is 0.8, i.e. discard 2 in every 10 tags. Also shuffles prompt.")
    parser.add_argument('--banlist', type=str, nargs='*', default=common.tag_banlist, help='Tags that will be excluded from wd14 tagging.')

    parser.add_argument('--use_dtg', action='store_true', default=False, help = "Enable the DTG tag extension LLM.")
    parser.add_argument('--dtg_rating', default='safe', help="Set desired prompt safety for DTG.")
    parser.add_argument('--dtg_target', choices=['<|very_short|>', '<|short|>', '<|long|>', '<|very_long|>'], default='<|long|>', help="Set desired prompt length for DTG.")
    parser.add_argument('--dtg_temperature', type=float, default=0.5, help="Set inference temperature for DTG.")

    parser.add_argument('--use_llm', action='store_true', default=False, help="Seek opinion of local LLM on image contents.")
    parser.add_argument('--llm_prompt', type=str, default='Please describe this image.', help="User prompt for local LLM.")
    parser.add_argument('--llm_sysmessage', type=str, default=None, help="A system prompt for the local LLM. If None, a sensible default is provided.")

    parser.add_argument('--skip_original_face', action='store_true', default=False, help="Don't include original face in the outputs. Default is False.")

    parser.add_argument('--fd_checkpoint', choices=acceptable_checkpoints, default=None, help='fd_checkpoint - an additional checkpoint only used for face detailing. Defaults to None.')

    
    parser.add_argument('--frontload_tags', nargs='+', default=['rating_safe'], help="Frontload tags. Default is ['rating_safe'].")
    parser.add_argument('--frontload_neg', nargs='+', default=[], help="Frontload negative tags. Default is [].")
    parser.add_argument('--mandatory_tags', nargs='+', default=[], help="Mandatory tags. Default is [].")

    parser.add_argument('--seeds', nargs='+', type=int, default=[-1], help="List of seeds. Default is [-1]. -1 is reevaluated each time, use -2 for a 'fixed random' seed.")
    parser.add_argument('--additional_detailer_seeds', nargs='*', type=int, default=[], help="List of additional detailer seeds. Default is [].")

    parser.add_argument('--skip', type=int, default=0, help='Attempt to skip the first n iterations - used for resuming an interrupted job.')

    parser.add_argument('--diff_start', type=common.args_validate_time, default=None, help="Diffusion start time in 'HH:MM' format.")
    parser.add_argument('--diff_stop', type=common.args_validate_time, default=None, help="Diffusion end time in 'HH:MM' format.")

    parser.add_argument('--output_folder', type=str, default='.', help="Output folder path. Default is '.', or the comfy output folder.")
    parser.add_argument('--hold_file', type=str, default='hold.txt', help="Holdfile path. Default is 'hold.txt' in the CWD.")

    parser.add_argument('--save_ora', action='store_true', default=False, help="Save ORA file after detailing. Default is False.")

    return parser.parse_args()

if __name__ == "__main__":
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

    if args.force_prompt:
        args.use_llm = False
        args.disable_tagger = True

    # prevent comfy from complaining about my args
    sys.argv = [sys.argv[0]]

    main(
        checkpoint_list=checkpoint_list,
        input_files=args.input_filenames, 
        force_prompt=args.force_prompt,
        from_step=args.from_step,
        denoise_programs=args.denoise_programs,
        noise=args.noise,
        skip_detailing=args.skip_detailing,
        detailer_selector=args.single_detailer,
        use_llm=args.use_llm,
        llm_prompt=args.llm_prompt,
        llm_sysmessage=args.llm_sysmessage,
        use_dtg=args.use_dtg,
        dtg_rating=args.dtg_rating,
        dtg_target=args.dtg_target,
        dtg_temperature=args.dtg_temperature,
        degrade_prompt=args.degrade_prompt,
        skip_original_face=args.skip_original_face,
        fd_checkpoint=fd_checkpoint,
        use_tagger=not args.disable_tagger,
        banlist=args.banlist,
        frontload_tags=args.frontload_tags,
        frontload_neg=args.frontload_neg,
        mandatory_tags=args.mandatory_tags,
        seeds=args.seeds,
        additional_detailer_seeds=args.additional_detailer_seeds,
        skip=args.skip,
        sleep_start=args.diff_start,
        sleep_stop=args.diff_stop,
        output_folder=args.output_folder,
        holdfile_path=args.hold_file,
        save_ora=args.save_ora
    )