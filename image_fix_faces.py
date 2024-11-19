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

import common
import checkpoints

from nodes import NODE_CLASS_MAPPINGS

from common import Silence
from common import WeightedList

from node_wrappers import *

def produceMetadataString(metadata):
    result = 'Original metadata missing or unrecoverable.'
    if 'parameters' in metadata:
        meta_pieces = metadata['parameters'].split('\n')
        result = f'Original metadata - Positive: {meta_pieces[0]}. ' + ' '.join(x for x in meta_pieces if x.startswith('Negative prompt:'))
    return result

def makeSegments(image, recipes = ['auto'], skip_segments = []):
    segsFromMask = NODE_CLASS_MAPPINGS['MaskToSEGS']()
    segsDetector = NODE_CLASS_MAPPINGS['ImpactSimpleDetectorSEGS']()
    ultraProviderClass = NODE_CLASS_MAPPINGS['UltralyticsDetectorProvider']()

    MASK_VALUE = 1

    results = []
    mask_tensor = torch.full((1, image.shape[1], image.shape[2]), 1 - MASK_VALUE, dtype=torch.float32)
    ultra_provider = None

    with Silence():
        for recipe in recipes:
            if isinstance(recipe, torch.Tensor):
                results.extend(
                    segsFromMask.doit(
                        mask=recipe,
                        combined=True,
                        crop_factor=3.0,
                        bbox_fill=False,
                        drop_size=1,
                        contour_fill=False
                    )[0][1]
                )
            elif recipe != 'auto':
                local_tensor = mask_tensor.clone()
                for l, t, w, h in recipe:
                    local_tensor[0, t:t+h, l:l+w] = MASK_VALUE
                results.extend(
                    segsFromMask.doit(
                        mask=local_tensor,
                        combined=True,
                        crop_factor=3.0,
                        bbox_fill=False,
                        drop_size=1,
                        contour_fill=False
                    )[0][1]
                )
            else:
                if not ultra_provider:
                    ultra_provider = ultraProviderClass.doit(model_name="bbox/face_yolov8m.pt")[0]
                results.extend(
                    segsDetector.detect(
                        bbox_detector = ultra_provider,
                        image = image,
                        bbox_threshold = 0.5,
                        bbox_dilation = 10,
                        crop_factor = 1.5,
                        drop_size = 10,
                        sub_threshold = 0.5,
                        sub_dilation = 0,
                        sub_bbox_expansion = 0,
                        sam_mask_hint_threshold = 0.7,
                        post_dilation = 0,
                        sam_model_opt = None,
                        segm_detector_opt = None,
                        detailer_hook = None
                    )[0][1]
                )

    common.log(f'** detected {len(results)} segments')
    results.sort(key = lambda x: (x.bbox[0], x.bbox[1]))
    if skip_segments and len(results) == 1:
            common.log('** overriding segment skip as only one segment detected')
    elif skip_segments:
        common.log(f'** dropping segments at indexes {skip_segments}')
        results = [x for (i, x) in enumerate(results) if i not in skip_segments]
    return results

def main(args):
    with Silence():
        add_extra_model_paths()
        import_custom_and_start()

    with torch.inference_mode():
        with Silence():
            clipEncoderClass = NODE_CLASS_MAPPINGS['CLIPTextEncode']()
            imageSaverClass = NODE_CLASS_MAPPINGS['Image Saver']()
            loraLoaderClass = NODE_CLASS_MAPPINGS['LoraLoader']()

        # build the segment map
        segment_info = {}
        for input_file_index, input_filename in enumerate(args.input_filenames):
            common.log(f'** working on image {input_file_index}, {input_filename}')
            input_image, metadata = common.load_image_as_image_tensor(input_filename)

            # the segment list starts off with a "bounding box" of sorts of the image dimensions,
            # then a list of the specific segments we wish to detail. this list will be embedded
            # in a larger structure

            segment_info[input_filename] = {
                'segments': [(input_image.shape[1], input_image.shape[2]), makeSegments(input_image, args.segs, args.skip_segment)], 
                'meta_for_file': produceMetadataString(metadata), 
                'input_image': input_image
            }

            prompt_list = []
            for prompt_index, input_prompt in enumerate(args.prompt_list):
                if not input_prompt or input_prompt == 'auto':
                    continue
                input_prompt_chunks = input_prompt.split('|')
                wildcard_body = '[ASC]'
                for input_prompt_chunk in input_prompt_chunks:
                    positive_prompt_without_stem = WeightedList()
                    for x in args.frontload_tags:
                        positive_prompt_without_stem.parse(x)
                    positive_prompt_without_stem.parse(input_prompt_chunk)
                    if args.use_dtg:
                        positive_prompt_without_stem = augmentPromptDTG(positive_prompt_without_stem, (input_image.shape[2], input_image.shape[1]), args.dtg_target, args.dtg_rating, args.dtg_temperature, args.banlist)
                    if args.mandatory_tags:
                        positive_prompt_without_stem = augmentPromptMandatory(positive_prompt_without_stem, args.mandatory_tags)
                    # final assembly
                    positive_prompt_without_stem.sort()
                    if args.shake > 0:
                        positive_prompt_without_stem.shake(args.shake)
                    wildcard_body += f'\n{positive_prompt_without_stem.to_string(suppress_lora=True)} [SEP]'
                prompt_list.append(wildcard_body)
            if not prompt_list or ('' not in prompt_list and not args.prompted_only):
                prompt_list.append('')
            
            segment_info[input_filename]['prompts'] = prompt_list

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

                input_keys = segment_info.keys()
                for input_keys_index, input_keys_filename in enumerate(input_keys):
                    common.log(f'** starting repair of file {input_keys_index+1} of {len(input_keys)}, {input_keys_filename}')
                    
                    segments = segment_info[input_keys_filename]['segments']
                    meta_for_file = segment_info[input_keys_filename]['meta_for_file']
                    input_image = segment_info[input_keys_filename]['input_image']
                    wildcards = segment_info[input_keys_filename]['prompts']
                    common.log(f'** wildcards for this run will be: {wildcards}')

                    basic_pipe = (model_object, clip_object, vae, positive_d_cond, negative_d_cond)
                    detailer_images = []
                    for wildcard_index, wildcard in enumerate(wildcards):
                        detailer_images.extend([(x, wildcard, runBasicPipe(basic_pipe, wildcard_index, input_image, segments, x, args, wildcard)) for x in args.seeds])

                    common.log(f'** writing {len(detailer_images)} images')
                    for (output_index, output_tuple) in enumerate(detailer_images):
                        detailer_seed, detailer_wildcard, output_image = output_tuple
                        # half of this metadata is wrong but a1111 doesn't really understand multistage generation
                        out_height = output_image.shape[1]
                        out_width = output_image.shape[2]
                        imageSaverClass.save_files(
                            filename=f'fr_{checkpoint_shortname}_{input_file_index}_%time_%seed_{prompt_index}_{output_index}',
                            path=args.output_folder,
                            extension="png",
                            steps=25,
                            cfg=7.5,
                            modelname=checkpoint,
                            sampler_name="euler_ancestral",
                            scheduler="normal",
                            positive=detailer_wildcard + ' === ' + meta_for_file,
                            negative=negative_stem.to_string(),
                            seed_value=detailer_seed,
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

                    if args.save_ora:
                        common.save_images_to_ora(input_image, detailer_images, f'{args.output_folder}/ora_t2i_{checkpoint_shortname}_{input_file_index}_{datetime.now().strftime("%Y-%m-%d-%H%M%S")}.ora')
                                    
                del model_object, clip_object
                gc.collect()
                torch.cuda.empty_cache()

def parse_args():
    acceptable_checkpoints = list(checkpoints.everything_d.keys()) + ['*']

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', choices=acceptable_checkpoints, default=['*'], help="List of checkpoints. Default is ['*'].")
    parser.add_argument('--input_filenames', type=common.args_valid_file_path, nargs='+', help="List of images to process.")

    parser.add_argument('--use_freeu', action='store_true', default=False, help='Enable freeU patching of model.')
    parser.add_argument('--freeu_b1', type=float, default=1.3)
    parser.add_argument('--freeu_b2', type=float, default=1.4)
    parser.add_argument('--freeu_s1', type=float, default=0.9)
    parser.add_argument('--freeu_s2', type=float, default=0.2)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prompts', nargs='+', help="List of prompts")
    group.add_argument('--prompt_file', type=common.args_read_prompts_from_file, help="File with prompts (one per line)")
    parser.add_argument('--rerun_lora', action='store_true', default=False, help="Rerun LoRA infected prompts without their LoRA. Default is False.")

    parser.add_argument('--use_dtg', action='store_true', default=False, help = "Enable the DTG tag extension LLM.")
    parser.add_argument('--dtg_rating', default='safe', help="Set desired prompt safety for DTG.")
    parser.add_argument('--dtg_target', choices=['<|very_short|>', '<|short|>', '<|long|>', '<|very_long|>'], default='<|long|>', help="Set desired prompt length for DTG.")
    parser.add_argument('--dtg_temperature', type=float, default=0.5, help="Set inference temperature for DTG.")
    parser.add_argument('--banlist', type=str, nargs='*', default=checkpoints.tag_banlist, help='Tags that will be excluded from wd14 tagging.')
    parser.add_argument('--add_banlist', type=str, nargs='*', default=[], help='Additional tags to ban.')

    parser.add_argument('--prompted_only', action='store_true', default=False, help="Only facefix with prompts. Default is False.")
    parser.add_argument('--segs', nargs='+', type=common.args_parse_bounding_box, default=['auto'], help='List of segments l:t+w+h or \'auto\' (the default).')
    parser.add_argument('--skip_segment', type=int, nargs='+', default=[], help='Segment indexes to skip.')

    parser.add_argument('--save_ora', action='store_true', default=False, help="Save ORA file after detailing. Default is False.")

    parser.add_argument('--steps', type=int, default=20, help="Number of steps to take (default 20).")
    parser.add_argument('--denoise', type=float, default=0.35, help="Detailer denoise strength (default 0.4).")
    parser.add_argument('--cycles', type=int, default=2, help="Detailing cycles (default 4).")

    parser.add_argument('--noise', type=float, default=0, help="Noise strength, applied to all embeddings. Default 0.0.")
    parser.add_argument('--shake', type=float, default=0, help="Perturb positive prompt tag weights by this standard deviation. Default 0.0 (off).")


    parser.add_argument('--frontload_tags', nargs='+', default=['rating_safe'], help="Frontload tags. Default is ['rating_safe'].")
    parser.add_argument('--frontload_neg', nargs='+', default=[], help="Frontload negative tags. Default is [].")
    parser.add_argument('--mandatory_tags', nargs='+', default=[], help="Mandatory tags. Default is [].")

    parser.add_argument('--seeds', nargs='+', type=int, default=[-1], help="List of seeds. Default is [-1]. -1 is reevaluated each time, use -2 for a 'fixed random' seed.")

    parser.add_argument('--diffusion_start', type=common.args_validate_time, default=None, help="Diffusion start time in 'HH:MM' format.")
    parser.add_argument('--diffusion_stop', type=common.args_validate_time, default=None, help="Diffusion end time in 'HH:MM' format.")

    parser.add_argument('--output_folder', type=str, default='.', help="Output folder path. Default is '.', or the comfy output folder.")
    parser.add_argument('--hold_file', type=str, default='hold.txt', help="Holdfile path. Default is 'hold.txt' in the CWD.")

    parser.add_argument('--endless', action='store_true', default=False, help="Run forever.")
    parser.add_argument('--endless_sleep', type=float, default=0, help="Run forever delay.")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    args.checkpoint_list = [checkpoints.everything_d[x] if x != '*' else random.choice(checkpoints.everything) for x in args.checkpoints]    

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
