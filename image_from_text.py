import argparse
import gc
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

    with torch.inference_mode():
        with Silence():
            clipEncoderClass = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            imageSaverClass = NODE_CLASS_MAPPINGS["Image Saver"]()
            latentClass = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
            loraLoaderClass = NODE_CLASS_MAPPINGS["LoraLoader"]()

        # optional fd_checkpoint
        if args.fd_checkpoint_tuple:
            fdc_sname, fdc_name, fdc_model, fdc_clip, fdc_vae, fdc_pos, fdc_neg, fdc_positive_cond, fdc_negative_cond = loadCheckpointProfile(args.fd_checkpoint_tuple, noise = args.noise)

        for bucket_id, bucket_prompts in makeLoraBuckets(args.prompt_list, rerun_lora = args.rerun_lora).items():
            common.log(f'** processing prompts from prompt bucket {bucket_id}')
            for checkpoint_tuple in args.checkpoint_list:
                if not checkpoint_tuple:
                    checkpoint_tuple = random.choice(list(checkpoints.everything_d.values()))
                checkpoint_shortname, checkpoint, model_object, clip_object, vae, positive_stem, negative_stem, positive_d_cond, negative_d_cond = loadCheckpointProfile(checkpoint_tuple, frontload_pos = args.frontload_tags, frontload_neg = args.frontload_neg, noise = args.noise)

                # apply the loras for this bucket
                for lora, ms, cs in bucket_prompts['loras']:
                    common.log(f'** patching model with lora:{lora}:{ms}:{cs}')
                    model_object, clip_object = loraLoaderClass.load_lora(lora_name = lora, strength_model = ms, strength_clip = cs, model = model_object, clip = clip_object)[0:2]

                if bucket_prompts['loras']:
                    # recreate the empty conditionings used for face detailing if necessary
                    positive_d_cond = common.scramble_embedding(clipEncoderClass.encode(text = positive_stem.to_string(suppress_lora=True), clip=clip_object)[0], args.noise)
                    negative_d_cond = common.scramble_embedding(clipEncoderClass.encode(text = negative_stem.to_string(suppress_lora=True), clip=clip_object)[0], args.noise)

                # freeu version of the base model
                if args.use_freeu:
                    common.log(f'** patching model with FreeU - b1:{args.freeu_b1}, b2:{args.freeu_b2}, s1:{args.freeu_s1}, s2:{args.freeu_s2}')
                    freeuClass = NODE_CLASS_MAPPINGS["FreeU_V2"]()
                    model_object = freeuClass.patch(b1=args.freeu_b1, b2=args.freeu_b2, s1=args.freeu_s1, s2=args.freeu_s2, model=model_object)[0]
            
                for (prompt_index, prompt_text) in enumerate(bucket_prompts['prompts']):
                    common.log(f'++ prompt {prompt_index} - {prompt_text}')
                    # dtg uses the aspect ratio as an argument into the model - for that reason, we can't settle the final
                    # version of the prompt until we're inside the following loop:
                    for dimensions in args.sizes:
                        # we add the frontloaded tags here as well so tools like dtg can see them
                        positive_prompt_without_stem = WeightedList()
                        for x in args.frontload_tags:
                            positive_prompt_without_stem.parse(x)
                        positive_prompt_without_stem.parse(prompt_text)

                        if args.use_dtg:
                            positive_prompt_without_stem = augmentPromptDTG(positive_prompt_without_stem, dimensions, args.dtg_target, args.dtg_rating, args.dtg_temperature, args.banlist)

                        if args.mandatory_tags:
                            positive_prompt_without_stem = augmentPromptMandatory(positive_prompt_without_stem, args.mandatory_tags)

                        # final assembly
                        positive_prompt_without_stem.sort()
                        if args.shake > 0:
                            positive_prompt_without_stem.shake(args.shake)
                        final_positive_prompt = WeightedList(positive_stem)
                        final_positive_prompt.extend(positive_prompt_without_stem)
                        final_positive_prompt.consolidate_keys(lambda x: max(x))
                        common.log(f'** positive prompt: {final_positive_prompt.to_string()}')
                        final_positive_string = final_positive_prompt.to_string(suppress_lora=True)
                        if args.supplementary_text:
                            common.log(f'** supplementary text: {args.supplementary_text}')
                            final_positive_string += f'. {args.supplementary_text}'
                        positive_cond = common.scramble_embedding(clipEncoderClass.encode(text=final_positive_string, clip=clip_object)[0], args.noise)

                        input_latent = latentClass.generate(width=dimensions[0], height=dimensions[1], batch_size=1)[0]

                        for seed in args.seeds:
                            seed = common.rseed(seed)
                            first_pass_samples = kSample(model_object, seed, positive_cond, negative_d_cond, input_latent, args.diffusion_start, args.diffusion_stop, args.hold_file)
                            sampled_images = [vae.decode(first_pass_samples)]

                            if args.rescale != 1.0:
                                nw = round(first_pass_samples.shape[3] * args.rescale)
                                nh = round(first_pass_samples.shape[2] * args.rescale)
                                upscaled_latent = {'samples': comfy.utils.common_upscale(first_pass_samples, nw, nh, 'bicubic', 'disabled')}
                                upscaled_samples = kSample(model_object, seed, positive_cond, negative_d_cond, upscaled_latent, args.diffusion_start, args.diffusion_stop, args.hold_file, steps = 12, denoise = args.rescale_denoise)
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
                                    (model_object, clip_object, vae, positive_cond, negative_d_cond, '', ultra_provider, None, None, None, None, None, None, None),
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
                                # half of this metadata is wrong but a1111 doesn't really understand multistage generation anyway
                                imageSaverClass.save_files(
                                    filename=f't2i_{checkpoint_shortname}_{prompt_index}_%time_%seed_{image_index}',
                                    path=args.output_folder,
                                    extension="png",
                                    steps=25,
                                    cfg=7.5,
                                    modelname=checkpoint,
                                    sampler_name="euler_ancestral",
                                    scheduler="normal",
                                    positive=final_positive_string,
                                    negative=negative_stem.to_string(),
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
                                common.save_images_to_ora(sampled_images[-1], detailer_images, f'{args.output_folder}/ora_t2i_{checkpoint_shortname}_{prompt_index}_{datetime.now().strftime("%Y-%m-%d-%H%M%S")}_{seed}_{detailer_index}.ora')
                                
                del model_object, clip_object
                gc.collect()
                torch.cuda.empty_cache()

def parse_args():
    acceptable_checkpoints = list(checkpoints.everything_d.keys()) + ['*', '!']

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', choices=acceptable_checkpoints, default=['*'], help="List of checkpoints. Default is ['*'].")

    parser.add_argument('--use_freeu', action='store_true', default=False, help='Enable freeU patching of model.')
    parser.add_argument('--freeu_b1', type=float, default=1.3)
    parser.add_argument('--freeu_b2', type=float, default=1.4)
    parser.add_argument('--freeu_s1', type=float, default=0.9)
    parser.add_argument('--freeu_s2', type=float, default=0.2)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prompts', nargs='+', help="List of prompts")
    group.add_argument('--prompt_file', type=common.args_read_prompts_from_file, help="File with prompts (one per line)")

    parser.add_argument('--supplementary_text', type=str, default='', help="A supplementary post prompt added after prompt generation.")

    parser.add_argument('--sizes', nargs='+', type=common.args_parse_int_tuple, default=[(1024, 1024), (832, 1216), (1216, 832)], help="List of sizes as 'width,height'. Default is [(1024,1024), (832,1216), (1216,832)]")

    parser.add_argument('--skip_original_face', action='store_true', default=False, help="Don't include original face in the outputs. Default is False.")
    parser.add_argument('--skip_detailing', action='store_true', default=False, help="Skip detailing. Default is False.")
    parser.add_argument('--detailer_selector', nargs='*', type=int, default=[], help="Select individual detailing pipes by zero-index. Defaults to [], which runs all pipes. 0: Detail with full prompt. 1: Detail with stem prompt only. 2: Detail with fd_checkpoint stem prompt (if provided).")
    parser.add_argument('--rerun_lora', action='store_true', default=False, help="Rerun LoRA infected prompts without their LoRA. Default is False.")

    parser.add_argument('--steps', type=int, default=20, help="Number of steps to take (default 20).")
    parser.add_argument('--denoise', type=float, default=0.4, help="Detailer denoise strength (default 0.4).")
    parser.add_argument('--cycles', type=int, default=4, help="Detailing cycles (default 4).")

    parser.add_argument('--use_dtg', action='store_true', default=False, help = "Enable the DTG tag extension LLM.")
    parser.add_argument('--dtg_rating', default='safe', help="Set desired prompt safety for DTG.")
    parser.add_argument('--dtg_target', choices=['<|very_short|>', '<|short|>', '<|long|>', '<|very_long|>'], default='<|long|>', help="Set desired prompt length for DTG.")
    parser.add_argument('--dtg_temperature', type=float, default=0.5, help="Set inference temperature for DTG.")
    parser.add_argument('--banlist', type=str, nargs='*', default=checkpoints.tag_banlist, help='Tags that will be excluded from wd14 tagging.')
    parser.add_argument('--add_banlist', type=str, nargs='*', default=[], help='Additional tags to ban.')

    parser.add_argument('--save_ora', action='store_true', default=False, help="Save ORA file after detailing. Default is False.")

    parser.add_argument('--rescale', type=float, default=1.0, help="HRFix. Defaults to 1.0 (disabled).")
    parser.add_argument('--rescale_denoise', type=float, default=0.4, help="HRFix denoise. Defaults to 0.4 (disabled).")
    parser.add_argument('--noise', type=float, default=0, help="Noise strength, applied to all embeddings. Default 0.0.")
    parser.add_argument('--shake', type=float, default=0, help="Perturb positive prompt tag weights by this standard deviation. Default 0.0 (off).")

    parser.add_argument('--fd_checkpoint', choices=acceptable_checkpoints, default=None, help="fd_checkpoint. Default is None.")

    parser.add_argument('--frontload_tags', nargs='+', default=['rating_safe'], help="Frontload tags. Default is ['rating_safe'].")
    parser.add_argument('--frontload_neg', nargs='+', default=[], help="Frontload negative tags. Default is [].")
    parser.add_argument('--mandatory_tags', nargs='+', default=[], help="Mandatory tags. Default is [].")

    parser.add_argument('--seeds', nargs='+', type=int, default=[-1], help="List of seeds. Default is [-1]. -1 is reevaluated each time, use -2 for a 'fixed random' seed.")
    parser.add_argument('--additional_detailer_seeds', nargs='*', type=int, default=[], help="List of additional detailer seeds. Default is [].")

    parser.add_argument('--diffusion_start', type=common.args_validate_time, default=None, help="Diffusion start time in 'HH:MM' format.")
    parser.add_argument('--diffusion_stop', type=common.args_validate_time, default=None, help="Diffusion end time in 'HH:MM' format.")

    parser.add_argument('--output_folder', type=str, default='.', help="Output folder path. Default is '.', your comfy output directory.")
    parser.add_argument('--hold_file', type=str, default='hold.txt', help="Holdfile path. Default is 'hold.txt'.")

    parser.add_argument('--endless', action='store_true', default=False, help="Run forever.")
    parser.add_argument('--endless_sleep', type=float, default=0, help="Run forever delay.")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.checkpoints == ['!']:
        args.checkpoint_list = checkpoints.everything_d.values()
    else:
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

    if args.prompts:
        prompt_list = args.prompts
    else:
        prompt_list = args.prompt_file
    args.prompt_list = prompt_list

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
