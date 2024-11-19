import argparse
import os
import random
import sys
import torch
from datetime import datetime

this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_path)
from boilerplate import *
comfy_path = find_path('ComfyUI')
sys.path.append(comfy_path)

import comfy.utils
import common
import checkpoints

from nodes import NODE_CLASS_MAPPINGS

from common import Silence
from common import WeightedList

from node_wrappers import *

def main(args):
    with Silence():
        add_extra_model_paths()
        import_custom_and_start()

    current_index = 0
    maximum_index = len(args.checkpoint_list) * len(args.input_filenames) * len(args.from_step) * len(args.denoise_programs) * len(args.seeds) 
    maximum_index_pad = len(str(maximum_index))
    common.log(f'~ > maximum index is: {maximum_index}')

    with torch.inference_mode():
        with Silence():
            clipEncoderClass = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            freeuClass = NODE_CLASS_MAPPINGS["FreeU_V2"]()
            imageSaverClass = NODE_CLASS_MAPPINGS["Image Saver"]()
            latentClass = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
            VAEEncoderClass = NODE_CLASS_MAPPINGS["VAEEncode"]()

        # optional fd_checkpoint
        if args.fd_checkpoint_tuple:
            fdc_sname, fdc_name, fdc_model, fdc_clip, fdc_vae, fdc_pos, fdc_neg, fdc_positive_cond, fdc_negative_cond = loadCheckpointProfile(args.fd_checkpoint_tuple, noise = args.noise)

        for checkpoint_tuple in args.checkpoint_list:
            child_combinations = len(args.input_filenames) * len(args.from_step) * len(args.denoise_programs) * len(args.seeds)
            if(current_index + child_combinations <= args.skip):
                common.log(f'** skipping checkpoint tuple {checkpoint_tuple}')
                current_index += child_combinations
                continue

            if not checkpoint_tuple:
                checkpoint_tuple = random.choice(list(checkpoints.everything_d.values()))
            checkpoint_shortname, checkpoint, model_object, clip_object, vae, positive_stem, negative_stem, positive_d_cond, negative_d_cond = loadCheckpointProfile(checkpoint_tuple, frontload_pos = args.frontload_tags, frontload_neg = args.frontload_neg, noise = args.noise)

            # freeu version of the base model
            if args.use_freeu:
                common.log(f'** patching model with FreeU - b1:{args.freeu_b1}, b2:{args.freeu_b2}, s1:{args.freeu_s1}, s2:{args.freeu_s2}')
                freeuClass = NODE_CLASS_MAPPINGS["FreeU_V2"]()
                model_object = freeuClass.patch(b1=args.freeu_b1, b2=args.freeu_b2, s1=args.freeu_s1, s2=args.freeu_s2, model=model_object)[0]
        
            for image_filename in args.input_filenames:
                child_combinations = len(args.from_step) * len(args.denoise_programs) * len(args.seeds)
                if(current_index + child_combinations <= args.skip):
                    common.log(f'** skipping image {image_filename}')
                    current_index += child_combinations
                    continue

                input_image, metadata = common.load_image_as_image_tensor(image_filename)
                input_image, image_width, image_height = common.scale_image_to_common_size(input_image)

                positive_prompt_without_stem = WeightedList()
                for x in args.frontload_tags:
                    positive_prompt_without_stem.parse(x)
                negative_prompt_without_stem = WeightedList()
                for x in args.frontload_neg:
                    negative_prompt_without_stem.parse(x)
                supplementary_text = ''

                if not args.force_prompt:
                    if not args.disable_tagger:
                        positive_prompt_without_stem = augmentPromptWD14(positive_prompt_without_stem, input_image, args.banlist)
                    if args.use_dtg:
                        positive_prompt_without_stem = augmentPromptDTG(positive_prompt_without_stem, (image_width, image_height), args.dtg_target, args.dtg_rating, args.dtg_temperature, args.banlist)
                    if args.erode_prompt < 1:
                        positive_prompt_without_stem.erode(args.erode_prompt)
                    if args.use_llm:
                        import llm
                        if positive_prompt_without_stem.get_keys():
                            llm_query = args.llm_prompt + f' To assist you, the image is known to contain the following tags: {positive_prompt_without_stem.to_string_weightless()}.'
                        else:
                            llm_query = args.llm_prompt
                        supplementary_text = llm.runLlamaPromptWrapper(llm_query, sys_message = args.llm_sysmessage, image = input_image).strip()                   
                else:
                    positive_prompt_without_stem.parse(args.force_prompt)
                if args.mandatory_tags:
                    positive_prompt_without_stem = augmentPromptMandatory(positive_prompt_without_stem, args.mandatory_tags)
                
                positive_prompt_without_stem.sort()
                if args.shake > 0:
                    positive_prompt_without_stem.shake(args.shake)

                final_positive_prompt = WeightedList(positive_stem)
                final_positive_prompt.extend(positive_prompt_without_stem)
                final_positive_prompt.consolidate_keys(lambda x: max(x))
                
                final_negative_prompt = WeightedList(negative_stem)
                final_negative_prompt.extend(negative_prompt_without_stem)
                final_negative_prompt.consolidate_keys(lambda x: max(x))

                positive_string = final_positive_prompt.to_string()
                if supplementary_text:
                    positive_string += f'. {supplementary_text}'
                negative_string = final_negative_prompt.to_string()

                positive_cond = clipEncoderClass.encode(text = positive_string, clip=clip_object)[0]
                negative_cond = clipEncoderClass.encode(text = negative_string, clip=clip_object)[0]

                # do our initial denoising over the original image from from_step
                common.log(f'{current_index} < {image_filename}')
                common.log(f'{current_index} + {positive_string}')
                common.log(f'{current_index} - {negative_string}')

                for from_step_value in args.from_step:
                    child_combinations = len(args.denoise_programs) * len(args.seeds)
                    if(current_index + child_combinations <= args.skip):
                        common.log(f'** skipping from_step value {from_step_value}')
                        current_index += child_combinations
                        continue

                    if from_step_value == 0:
                        input_latent = latentClass.generate(width=image_width, height=image_height, batch_size=1)[0]
                    else:
                        input_latent = VAEEncoderClass.encode(pixels=input_image, vae=vae)[0]

                    for seed in args.seeds:
                        child_combinations = len(args.denoise_programs)
                        if(current_index + child_combinations <= args.skip):
                            common.log(f'** skipping seed {seed}')
                            current_index += child_combinations
                            continue

                        seed=common.rseed(seed)
                        common.log(f'{current_index} ! seed: {seed}, from step: {from_step_value}, denoise programs: {args.denoise_programs}')

                        first_pass_samples = kSample(model_object, seed, positive_cond, negative_cond, input_latent, args.diffusion_start, args.diffusion_stop, args.hold_file, from_step = from_step_value)
                        original_i2i = vae.decode(first_pass_samples)

                        for (upscale_denoise, upscale_scale) in args.denoise_programs:
                            current_index += 1
                            if(current_index <= args.skip):
                                common.log(f'** skipping denoise program: upscale to {upscale_scale} and denoise at {upscale_denoise}')
                                continue
                            common.log(f'{current_index} ~ denoise program: x{upscale_scale} ~{upscale_denoise}')
                            file_stub = f'{image_filename.stem[:15]}Z_{current_index:0{maximum_index_pad}}_o_{maximum_index}_{checkpoint_shortname}_sd{from_step_value}_ud{upscale_denoise}_us{upscale_scale}'
                            common.log(f'{current_index} > {file_stub}')

                            sampled_images = [original_i2i]

                            if upscale_scale != 1.0:
                                nw = round(first_pass_samples.shape[3] * upscale_scale)
                                nh = round(first_pass_samples.shape[2] * upscale_scale)
                                upscaled_latent = {'samples': comfy.utils.common_upscale(first_pass_samples, nw, nh, 'bicubic', 'disabled')}
                                upscaled_samples = kSample(model_object, seed, positive_cond, negative_cond, upscaled_latent, args.diffusion_start, args.diffusion_stop, args.hold_file, steps = 12, denoise = upscale_denoise)
                                sampled_images.append(vae.decode(upscaled_samples))
                            
                            detailer_images = []

                            if not args.skip_detailing:
                                ultraProviderClass = NODE_CLASS_MAPPINGS["UltralyticsDetectorProvider"]()
                                ultra_provider = ultraProviderClass.doit(model_name="bbox/face_yolov8m.pt")[0]

                                # detailer pipes are just 14-tuples:
                                #  model, clip, vae, positive, negative, wildcard, bbox, 
                                #  segm_detector_opts / None, sam_model_opt / None, detailer_hook / None, 
                                #  refiner_model / None, refiner_clip / None, refiner_pos / None, refiner_neg / None
                                
                                detailer_pipes = [
                                    (model_object, clip_object, vae, positive_cond, negative_cond, '', ultra_provider, None, None, None, None, None, None, None),
                                    (model_object, clip_object, vae, positive_d_cond, negative_d_cond, '', ultra_provider, None, None, None, None, None, None, None),
                                    *([(fdc_model, fdc_clip, fdc_vae, fdc_positive_cond, fdc_negative_cond, '', ultra_provider, None, None, None, None, None, None, None)] if args.fd_checkpoint else []),
                                ]
                                if args.detailer_selector != []:
                                    detailer_pipes = [x for i,x in enumerate(detailer_pipes) if i in args.detailer_selector]
                                    
                                for detailer_index, detailer_pipe in enumerate(detailer_pipes):
                                    detailer_images.extend([runDetailerPipe(detailer_pipe, detailer_index, sampled_images[-1], x, args) for x in [seed] + args.additional_detailer_seeds])
                        
                            output_images = []
                            if not args.skip_original_face or not detailer_images:
                                output_images.extend(sampled_images)
                            output_images.extend(detailer_images)
                            unique = []
                            for t in output_images:
                                if not any(torch.equal(t, ut) for ut in unique):
                                    unique.append(t)

                            common.log(f'** writing {len(unique)} images')
                            for (image_index, image_data) in enumerate(unique):
                                # half of this metadata is wrong but a1111 doesn't really understand multistage generation
                                imageSaverClass.save_files(
                                    filename=f'{file_stub}_%time_%seed_{image_index}',
                                    path=args.output_folder,
                                    extension="png",
                                    steps=25,
                                    cfg=7.5,
                                    modelname=checkpoint,
                                    sampler_name="euler_ancestral",
                                    scheduler="normal",
                                    positive=positive_string,
                                    negative=negative_string,
                                    seed_value=seed,
                                    width=image_data.shape[2],
                                    height=image_data.shape[1],
                                    lossless_webp=True,
                                    quality_jpeg_or_webp=100,
                                    optimize_png=False,
                                    counter=0,
                                    denoise=1,
                                    time_format="%Y-%m-%d-%H%M%S",
                                    save_workflow_as_json=False,
                                    embed_workflow_in_png=False,
                                    strip_a1111_params="nothing",
                                    images=image_data,
                                )
                            
                            if args.save_ora:
                                common.save_images_to_ora(sampled_images[-1], detailer_images, f'{args.output_folder}/ora_{file_stub}_{datetime.now().strftime("%Y-%m-%d-%H%M%S")}_{seed}.ora')

def parse_args() -> argparse.ArgumentParser:
    acceptable_checkpoints = list(checkpoints.everything_d.keys()) + ['*']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', choices=acceptable_checkpoints, default=['*'], help="List of checkpoints. Default is ['*'].")
    parser.add_argument('--input_filenames', type=common.args_valid_file_path, nargs='+', help="List of images to process.")

    parser.add_argument('--use_freeu', action='store_true', default=False, help='Enable freeU patching of model.')
    parser.add_argument('--freeu_b1', type=float, default=1.3)
    parser.add_argument('--freeu_b2', type=float, default=1.4)
    parser.add_argument('--freeu_s1', type=float, default=0.9)
    parser.add_argument('--freeu_s2', type=float, default=0.2)

    parser.add_argument('--from_step', type=int, nargs='+', default=[14], help="Start denoising the original image from this step. At zero, creates a latent from scratch. Default is 14.")

    parser.add_argument('--denoise_programs', type=common.args_parse_float_tuple, nargs='+', default=[(0.0, 1.0)], help="Upscale denoising profiles - the upscaler will denoise at the strength of the first item, after scaling up by the second (e.g. 0.4,1.1). The default is (0.0, 1.0), which does not upscale.")

    parser.add_argument('--noise', type=float, default=0, help="Noise strength, applied to all embeddings. Default 0.0.")
    parser.add_argument('--shake', type=float, default=0, help="Perturb positive prompt tag weights by this standard deviation. Default 0.0 (off).")

    parser.add_argument('--skip_original_face', action='store_true', default=False, help="Don't include original face in the outputs. Default is False.")
    parser.add_argument('--skip_detailing', action='store_true', default=False, help="Skip detailing. Default is False.")
    parser.add_argument('--detailer_selector', nargs='*', type=int, default=[], help="Select individual detailing pipes by zero-index. Defaults to [], which runs all pipes. 0: Detail with full prompt. 1: Detail with stem prompt only. 2: Detail with fd_checkpoint stem prompt (if provided).")

    parser.add_argument('--steps', type=int, default=20, help="Number of steps to take (default 20).")
    parser.add_argument('--denoise', type=float, default=0.35, help="Detailer denoise strength (default 0.4).")
    parser.add_argument('--cycles', type=int, default=2, help="Detailing cycles (default 4).")

    parser.add_argument('--force_prompt', type=str, default=None, help="Forced prompt, displacing the LLM and tagger.")
    parser.add_argument('--disable_tagger', action='store_true', default=False, help="Disable the wd14 tagger.")
    parser.add_argument('--erode_prompt', type=float, default=0.8, help="If less than 1 only retain this percent of detected tags. Default is 0.8, i.e. discard 2 in every 10 tags. Also shuffles prompt.")

    parser.add_argument('--use_dtg', action='store_true', default=False, help = "Enable the DTG tag extension LLM.")
    parser.add_argument('--dtg_rating', default='safe', help="Set desired prompt safety for DTG.")
    parser.add_argument('--dtg_target', choices=['<|very_short|>', '<|short|>', '<|long|>', '<|very_long|>'], default='<|long|>', help="Set desired prompt length for DTG.")
    parser.add_argument('--dtg_temperature', type=float, default=0.5, help="Set inference temperature for DTG.")
    parser.add_argument('--banlist', type=str, nargs='*', default=checkpoints.tag_banlist, help='Tags that will be excluded from wd14 tagging.')
    parser.add_argument('--add_banlist', type=str, nargs='*', default=[], help='Additional tags to ban.')

    parser.add_argument('--use_llm', action='store_true', default=False, help="Seek opinion of local LLM on image contents.")
    parser.add_argument('--llm_prompt', type=str, default='Please describe this image.', help="User prompt for local LLM.")
    parser.add_argument('--llm_sysmessage', type=str, default=None, help="A system prompt for the local LLM. If None, a sensible default is provided.")

    parser.add_argument('--fd_checkpoint', choices=acceptable_checkpoints, default=None, help='fd_checkpoint - an additional checkpoint only used for face detailing. Defaults to None.')
    
    parser.add_argument('--frontload_tags', nargs='+', default=['rating_safe'], help="Frontload tags. Default is ['rating_safe'].")
    parser.add_argument('--frontload_neg', nargs='+', default=[], help="Frontload negative tags. Default is [].")
    parser.add_argument('--mandatory_tags', nargs='+', default=[], help="Mandatory tags. Default is [].")

    parser.add_argument('--seeds', nargs='+', type=int, default=[-1], help="List of seeds. Default is [-1]. -1 is reevaluated each time, use -2 for a 'fixed random' seed.")
    parser.add_argument('--additional_detailer_seeds', nargs='*', type=int, default=[], help="List of additional detailer seeds. Default is [].")

    parser.add_argument('--skip', type=int, default=0, help='Attempt to skip the first n iterations - used for resuming an interrupted job.')

    parser.add_argument('--diffusion_start', type=common.args_validate_time, default=None, help="Diffusion start time in 'HH:MM' format.")
    parser.add_argument('--diffusion_stop', type=common.args_validate_time, default=None, help="Diffusion end time in 'HH:MM' format.")

    parser.add_argument('--output_folder', type=str, default='.', help="Output folder path. Default is '.', or the comfy output folder.")
    parser.add_argument('--hold_file', type=str, default='hold.txt', help="Holdfile path. Default is 'hold.txt' in the CWD.")

    parser.add_argument('--save_ora', action='store_true', default=False, help="Save ORA file after detailing. Default is False.")

    parser.add_argument('--endless', action='store_true', default=False, help="Run forever.")
    parser.add_argument('--endless_sleep', type=float, default=0, help="Run forever delay.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    args.checkpoint_list = [checkpoints.everything_d[x] if x != '*' else random.choice(checkpoints.everything) for x in args.checkpoints]
    
    if args.fd_checkpoint and args.fd_checkpoint != '*':
        fd_checkpoint_tuple = checkpoints.everything_d[args.fd_checkpoint]
    elif args.fd_checkpoint == '*':
        fd_checkpoint_tuple = random.choice(checkpoints.everything)
    else:
        fd_checkpoint_tuple = None
    args.fd_checkpoint_tuple = fd_checkpoint_tuple

    # -2 is a "fixed random seed" which is generated once and then stays the same
    # -1 is the classic random seed which regenerates every time it is called on
    args.seeds = [common.rseed(-1) if x == -2 else x for x in args.seeds]

    # supplement the built in banlist if required
    if args.add_banlist:
        args.banlist.extend(args.add_banlist)

    # prevent comfy from complaining about my args
    sys.argv = [sys.argv[0]]

    common.log(args)

    while True:
        main(args)
        if not args.endless:
            break
        else:
            import time
            common.log(f'** endless sleeping for {args.endless_sleep}')
            time.sleep(args.endless_sleep)
