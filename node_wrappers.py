import common
from common import Silence, WeightedList
from nodes import NODE_CLASS_MAPPINGS

def loadCheckpointProfile(cp_tuple, frontload_pos = [], frontload_neg = [], noise = 0, clip_skip = -2):
    cplsClass = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
    clipEncoderClass = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    vaeLoaderClass = NODE_CLASS_MAPPINGS["VAELoader"]()

    shortname, checkpoint_name, checkpoint_lora, checkpoint_pos_stem, checkpoint_neg_stem, checkpoint_vae_name = cp_tuple

    checkpoint_pos_stem = WeightedList(checkpoint_pos_stem)
    for x in frontload_pos:
        checkpoint_pos_stem.parse(x)
    checkpoint_neg_stem = WeightedList(checkpoint_neg_stem)
    for x in frontload_neg:
        checkpoint_neg_stem.parse(x)

    common.log(f'** loading checkpoint {shortname}')
    with Silence():
        checkpoint_model, checkpoint_clip, checkpoint_vae = cplsClass.load_checkpoint(ckpt_name=checkpoint_name)
        checkpoint_clip.clip_layer(clip_skip)

    if checkpoint_lora:
        model_strength = clip_strength = 0.7
        common.log(f'** patching checkpoint with lora:{checkpoint_lora}:{model_strength}:{clip_strength}')
        with Silence():
            loraLoaderClass = NODE_CLASS_MAPPINGS["LoraLoader"]()
            lora_result = loraLoaderClass.load_lora(lora_name = checkpoint_lora, strength_model = model_strength, strength_clip = clip_strength, model = checkpoint_model, clip = checkpoint_clip)
            checkpoint_model, checkpoint_clip = lora_result[0:2]

    with Silence():
        checkpoint_pos_cond = common.scramble_embedding(clipEncoderClass.encode(text = checkpoint_pos_stem.to_string(), clip=checkpoint_clip)[0], noise)
        checkpoint_neg_cond = common.scramble_embedding(clipEncoderClass.encode(text = checkpoint_neg_stem.to_string(), clip=checkpoint_clip)[0], noise)

    if checkpoint_vae_name != 'baked':
        checkpoint_vae = vaeLoaderClass.load_vae(vae_name=checkpoint_vae_name)[0]
    
    return (shortname, checkpoint_name, checkpoint_model, checkpoint_clip, checkpoint_vae, checkpoint_pos_stem, checkpoint_neg_stem, checkpoint_pos_cond, checkpoint_neg_cond)

def augmentPromptWD14(prompt_object, image, banlist = [], threshold = 0.35, character_threshold = 0.85, model = 'wd-v1-4-moat-tagger-v2'):
    wd14TaggerClass = NODE_CLASS_MAPPINGS["WD14Tagger|pysssss"]()
    with Silence():
        tagger_tags = wd14TaggerClass.tag(
            model=model,
            threshold=threshold, 
            character_threshold=character_threshold,
            replace_underscore=False,
            trailing_comma=False,
            exclude_tags=','.join(x.replace(' ', '_') for x in banlist),
            image=image,
        )['result'][0][0]
    npo = WeightedList(prompt_object)
    npo.parse(tagger_tags)
    common.log(f'** wd14 returned these tags: {npo.to_string()}')
    return npo

def augmentPromptDTG(prompt_object, dimensions = [1,1], dtg_target = '<|long|>', dtg_rating = 'safe', dtg_temperature = 0.5, banlist = []):
    import llm
    # sort high priority keys to the front
    npo = WeightedList(prompt_object)
    npo.sort()
    common.log(f'** invoking dtg - {dtg_target}/{dtg_rating}/{dtg_temperature}')
    with Silence():
        dtg_results = llm.runDTGPromptWrapper(
            npo.get_keys(suppress_lora=True), 
            dtg_rating, 
            dimensions[0], 
            dimensions[1], 
            dtg_target, 
            banlist,
            dtg_temperature
        )
    common.log(f'** dtg returned these additional tags: {dtg_results[2]}')
    npo.parse(', '.join(dtg_results[2]))
    return npo

def augmentPromptMandatory(prompt_object, mandatory):
    npo = WeightedList(prompt_object)
    for tag in mandatory:
        if '|' in tag:
            trigger, tag_body = tag.split('|', maxsplit = 1)
            if trigger in npo:
                common.log(f'** mandatory term trigger: \'{trigger}\' seen in prompt, adding \'{tag_body}\'')
                npo.parse(tag_body)
        else:
            npo.parse(tag)
    return npo

def kSample(model, seed, p_cond, n_cond, latent, d_start, d_stop, hfile, steps = 25, denoise = 1.0, from_step = 0):
    kSamplerClass = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]()
    if(d_start and d_stop): common.sleep_while_outside(d_start, d_stop)
    common.sleep_while_holdfile(hfile)
    if seed == -1: seed = common.rseed(seed)
    return kSamplerClass.sample(
        model=model,
        add_noise='enable',
        noise_seed=seed,
        steps=steps,
        cfg=6.5,
        sampler_name="euler_ancestral",
        scheduler="normal",
        positive=p_cond,
        negative=n_cond,
        latent_image=latent,
        start_at_step=from_step,
        end_at_step=10000,
        return_with_leftover_noise='disabled',
        denoise=denoise
    )[0]['samples']

def makeLoraBuckets(prompt_list, rerun_lora = False):
    # the intermingling of loras and prompts has been a disaster for the human race
    prompt_buckets = {'base': {'loras': [], 'prompts': []}}
    for prompt in prompt_list:
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
    return prompt_buckets

def runDetailerPipe(pipe, index, image, seed, args):
    fdpClass = NODE_CLASS_MAPPINGS['FaceDetailerPipe']()
    if(args.diffusion_start and args.diffusion_stop): common.sleep_while_outside(args.diffusion_start, args.diffusion_stop)
    common.sleep_while_holdfile(args.hold_file)
    seed = common.rseed(seed)
    common.log(f'** running detailer pipe {index} with seed {seed}')
    return fdpClass.doit(
        guide_size=384,
        guide_size_for=True,
        max_size=1024,
        seed=seed,
        steps=args.steps,
        cfg=7.5,
        sampler_name="euler_ancestral",
        scheduler="normal",
        denoise=args.denoise,
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
        cycle=args.cycles,
        inpaint_model=False,
        noise_mask_feather=0,
        image=image,
        detailer_pipe=pipe,
    )[0]

def runBasicPipe(pipe, index, image, segments, seed, args, wildcard = ''):
    fdpClass = NODE_CLASS_MAPPINGS['DetailerForEachPipe']()
    if(args.diffusion_start and args.diffusion_stop): common.sleep_while_outside(args.diffusion_start, args.diffusion_stop)
    common.sleep_while_holdfile(args.holdfile_path)
    common.log(f'** running detailer pipe {index} with seed {seed}')
    if(wildcard):
        common.log(f'** wildcard is:\n{wildcard}')
    return fdpClass.doit(
        image = image,
        segs = segments,
        guide_size = 1024,
        guide_size_for='bbox',
        max_size=1024,
        seed=seed,
        steps=args.steps,
        cfg=7.5,
        sampler_name='euler_ancestral',
        scheduler='normal',
        denoise=args.denoise,
        feather=5,
        noise_mask=True,
        force_inpaint=True,
        basic_pipe=pipe,
        wildcard=wildcard,
        refiner_ratio=None,
        detailer_hook=None,
        refiner_basic_pipe_opt=None,
        cycle=args.cycles,
        inpaint_model=False,
        noise_mask_feather=20,
        scheduler_func_opt=None
    )[0]
