# comfyui-python-workflows
 Workflow scripts demonstrating some uses of [pydn's comfyui-to-python-extension](https://github.com/pydn/ComfyUI-to-Python-Extension).

 These scripts reflect certain assumptions about my own workflows (including that they are very SDXL focused) - they may not match up with your needs exactly, but being released under [the MIT License](LICENSE), you should feel free to modify them to suit your needs. At a minimum however, you should look at `checkpoints.py` to set the paths for your checkpoints, and if you use the tagging functionality in `image_remaker.py`, you will probably want to consider setting the list of banned tags in `common.py` depending on what your preferred tagging models tend to overdetect / what you do not want to see.

# recent changes:
The `tag_banlist` configuration item has moved from `common.py` to `checkpoints.py`. Although these are not technically checkpoint related, it was decided to try and centralise all of the file related configuration in one file.

# dependencies
These scripts assume the presence of certain ComfyUI nodes, which they load as part of their execution. These include:
* [Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
* [Image Saver](https://github.com/alexopus/ComfyUI-Image-Saver)

If you want to use the image remaker script, you will want:
* [WD14Tagger](https://github.com/pythongosssss/ComfyUI-WD14-Tagger)

If you want to use llama to interrogate your images as part of the remaker script you will need [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) installed and working. llama-cpp-python is also required to make use of [DanTagGen](https://huggingface.co/KBlueLeaf/DanTagGen-delta-rev2) for prompt supplementation. If you use either, make sure to adjust the paths of the relevant gguf model files in `checkpoints.py`. 

If you want to use layered image saving support (i.e. saving each face detailer result over the original image in its own layer and exporting an ORA image file), you will need the [layeredimage](https://github.com/FHPythonUtils/LayeredImage/) library installed in your comfy environment. This has caused some users conflicts depending on which other nodes they are using.

The scripts assume an intermediate level of python capability - you may need to tailor them to your own environment, a task I cannot assist with.

# using the scripts
 These scripts should be run using the same Python environment you use to run Comfy, from the folder containing your ComfyUI folder. 
 
 An example session with the scripts might look like this - you start by running the script to generate some raw images. In this case, you disable face detailing because you want to look through the images first and only detail the most promising ones.
 ```PowerShell
PS 02/11/2024 16:50:46> ..\python_embeded\python.exe C:\Users\curious\Documents\GitHub\comfy_python\image_from_text.py --checkpoints sxzluma_real --prompts "1boy, solo, a wizard is drinking coffee in a cafe" --sizes 832,1024 --skip_detailing --rescale 1.1 --rescale_denoise 0.4 --seeds -1 -1 -1 -1 --output_folder raw
```

You end up with eight images in a subfolder `raw` in your ComfyUI output folder - two each for each seed, representing the before and after of the rescaling process. You pick through them and select two or three that you like, and delete the others. In this case, you will leave the task of identifying the faces up to an ultralytics provider, but you can manually specify the bounding boxes if you prefer (or if you are detailing something that is not a face).

```PowerShell
PS 02/11/2024 17:07:11> ..\python_embeded\python.exe C:\Users\curious\Documents\GitHub\comfy_python\image_fix_faces.py --checkpoints sxzluma_real --input_filenames (Get-Item output\raw\*.png).FullName --prompts "green_eyes" --steps 20 --denoise 0.4 --cycles 4 --noise 0.1 --seeds -1 -1 -1 -1 --output_folder detailed
```

This loads our images, detects their faces, and rediffuses them with a four pass, 20 step process. The face detailer script uses both the prompts passed in on the command line and a 'default prompt' (containing only the preamble specified in the checkpoint record - this default prompt can be disabled from the command line) - so in this case you get eight images back, two for each seed.

Finally, we cherrypick a result we like.

![image](https://github.com/user-attachments/assets/40ef9599-4554-4581-b42b-b79eeb859bc7)

Later, you decide to take this original image and iterate on it. You put it in a folder named "step1", and run the following command:

```PowerShell
PS 10/11/2024 01:01:27> foreach ($i in 1..10) { ..\python_embeded\python.exe C:\Users\curious\Documents\GitHub\comfy_python\image_remaker.py --checkpoints * --degrade_prompt 0.5 --use_dtg --dtg_rating safe --dtg_target "<|long|>" --dtg_temperature 0.5 --from_step 4 --frontload_tags rating_safe --mandatory_tags "hat|witches_hat" --output_folder "output\step$($i+1)\" --input_filenames (Get-Item "output\step$i\*.png").FullName  --skip_original_face --single_detailer 3 }
```

Here, we take our source image, run a wd14 tagger over it, throw away 50% of the resulting tags, and then use the DTG llm to 'refill' our prompt back to approximately 40 tags (what DTG calls the '<|long|>' target). If we see the tag 'hat' in the results, we will add the tag 'witches_hat' to it. We can see this tag mutation process in action:
```
20241110 01:06:51: ** degrading 21 tags down to 10
20241110 01:06:51: ** - tagger tags were looking_at_viewer, black_hair, 1boy, holding, jewelry, sitting, upper_body, outdoors, solo_focus, day, necklace, tree, cup, facial_hair, umbrella, chair, table, plant, holding_cup, mug, realistic
20241110 01:06:51: ** - tagger tags now looking_at_viewer, black_hair, solo_focus, day, necklace, tree, facial_hair, chair, plant, realistic
20241110 01:06:55: ** dtg returned the following tags, which will replace the originals: ['looking at viewer', 'black hair', 'solo focus', 'day', 'necklace', 'tree', 'facial hair', 'chair', 'plant', 'realistic', 'pencil skirt', 'smile', 'table', 'holding', 'crossed arms', 'potted plant', 'beard', 'brown eyes', 'glasses', 'hand on own cheek', 'crossed legs', 'long hair', 'open book', 'outdoors', 'old man', 'old', 'wrinkled skin', 'grey hair', 'blonde hair', 'hand on own face', 'building', 'old woman', 'jewelry', 'grey eyes', 'mustache', 'book', 'indoors', 'skirt', 'holding book', 'sitting', 'window']
```

(Note, the loss of 1boy in the original tags means the model will now almost certainly drift to producing a female character due to training bias.) We then rediffuse the original image using a random checkpoint ('*'), starting at step 4. We also detail the face, using the fourth of the the default detailer pipes (`--single_detailer 3` - these pipes are zero indexed). When it comes time to save the file, we only save the detailed version and skip the original.

Finally, we then loop over this process ten times, each time loading from and saving to the next folder in the chain. 

![remake](https://github.com/user-attachments/assets/cda6370f-4dde-4f80-85d8-cce6da766b32)

Each of the three scripts has a large number of command line switches that let you tune their behaviour, which the rest of this readme file deals with.

# image_from_text.py
## checkpoint specification
`--checkpoints` - must be one or more of the shortnames used to define checkpoint tuples in `checkpoints.py` or `*`, which will choose one at random (which is the default.)

`--fd_checkpoint` - takes the same values as `--checkpoints`, but can only be a single value. If set, provides a fifth default image detailing pipe (see `--single_detailer` below). Can be helpful when you have a model (including stable diffusion 1.5 models) which does particularly good faces.

## prompts and lora
`--prompts` or `--prompt_file` - either a list of strings which form your positive prompts or a text file containing one prompt per line. In this second case, lines beginning with `#` are skipped. This (and pretty much only this) supports inline specification of loras using the familiar `<lora:path_to_lora_without_safetensors_extension:model_str:clip_str>`. Prompts will be grouped together into 'buckets' by the lora they use and their strength, and then processed sequentially, bucket by bucket. The `<lora:>` tag itself is hidden when generating the embedding of the prompt but will be present in image metadata.

_On inline lora support generally_: in-prompt lora handling is fiddly and low on my priorities to fix - I tend to instead create a new checkpoint tuple to handle these. While lora are parsed out of prompts for `image_from_text`, they do not work in the following places:
* A checkpoint specified in `--fd_checkpoint` ignores lora from the checkpoint specification.
* Prompts specified with `--force_prompt` in `image_remaker`.
* Prompts specified in `image_fix_faces`.

`--rerun_lora` - if this is true, prompts for a given lora bucket are also run in the `base` bucket (without any lora loaded). This can let you test the strength or effectiveness of a given lora.

`--frontload_tags` and `--frontload_neg` - final prompts are created by merging the positive and negative stems specified for a given checkpoint with the tags specified in these 'frontload' arguments, and then adding in the user's prompt (or the prompt generated by other processes, depending on the script run). These arguments provide a convenient way to bulk set a tag without specifying it in each prompt.

`--use_dtg` the [DanTagGen](https://huggingface.co/KBlueLeaf/DanTagGen-delta-rev2) LLM can be used to augment prompt tags with other tags that would be contextually relevant. If you use this flag, you must have llama-cpp-python installed, have downloaded a gguf checkpoint from the linked repository, and have set the key in your checkpoints file appropriately. DTG has several other relevant options - `--dtg_rating`, `--dtg_target` and `--dtg_temperature`. Check the linked model card for more information on how these work and the values they can take.

`--mandatory_tags` - when a prompt has been assembled from the checkpoint stem, `frontload` arguments and other inputs (like dtg), these items are added to it. By default, this only affects the position of tag in the final prompt, but it is also possible to specify conditional mandatory tags by use of a pipe - `--mandatory_tags sky|(clouds:1.2)` will add the `(clouds:1.2)` tag to the prompt, but *only* if it included the `sky` tag at the time the mandatory tags were processed. As mandatory tags are processed in the specified order, you can use this behaviour to chain them together.

`--noise` adds random noise to the embeddings generated from your prompts. The results are incoherent from a math perspective because the regular embedding and its pooled representation have their noise calculated independently, but it can still produce some interesting effects at low strengths, say 0.1.

_On prompt ordering and tag duplication_: prior to being clipped into a conditioning, prompt objects are generally consolidated (multiple copies of a tag are replaced with a single copy at the earliest position, with a weight equal to the greatest observed weight amongst the copies) and sorted (certain tags, such as tags relating to image quality or subjects are moved to the front of the prompt).

## sizes, hqfix, seeds
`--sizes` - a list of comma separated integer sizes. The default is a list of three sizes - `1024,1024`, `832,1216`, and `1216,832`.

`--rescale`, `--rescale_denoise` - a form of high-resolution fixing, the image will be subject to bicubic latent upscaling by the rescale factor and then re-diffused for 12 steps at the provided denoising strength. Generally, both the original and upscaled image will be saved as outputs, but only the second image will be detailed if detailing is enabled.

`--seeds` - a list of seed values. -1 generates a seed in the range of 1 to 1125899906842624. -2 does the same, but the seed is generated and then locked at the time the arguments are parsed, i.e. it is not rerandomised as the process continues / checkpoints are switched, etc etc.

## face detailing
`--skip_original_face` - if detailing is turned on, the original, undetailed face will not be saved.
`--skip_detailing` - do not perform any detailing. Detailing is on by default.
`--single_detailer` - by default, the script carries out four different detailing pipelines (five if an `fd_checkpoint` is specified) 
- 0: the unmodified model object paired with the provided positive prompt, 
- 1: the unmodified model object paired with the checkpoint's default positive prompt
- 2: the model modified by FreeU_V2 paired with the provided positive prompt,
- 3: the model modified by FreeU_V2 paired with the checkpoint's default positive prompt
- 4: the fd_checkpoint model if provided

Pipeline 2 is 'closest' to what will have been used to generate the original image, but you may find that very rich prompts lead to bad faces (as the image attempts to diffuse in e.g. tagged background elements). By providing an index like 0, 1, 2, 3, or 4, only that detailing process will be executed. The default is -1, which means that all the pipelines will be run and saved. I tend to use 3.

`--additional_detailer_seeds` - the detailer processes described above are generally run once each for each image, but you can provide additional seeds here to generate further alternative face candidates.

## file saving
`--save_ora` - if you are detailing, this option will also save an .ORA file containing the base image and then the differences for each detailer process stacked on top as layers. Can be useful for picking and choosing faces between different generations. GIMP can edit these files. If enabled, always saves a file even if no detailing was done (due to a lack of face detections or all faces detected being above the threshold size).

## pausing diffusion
`--diff_start` and `--diff_stop` - the program will pause from diffusing images outside of the time period specified by these arguments. For limiting power use during very large scale diffusion runs.
`--hold_file` - the program will pause from diffusing images while this file exists - to be used to pause and restart the process, including remotely via a mounted filesystem.

`--endless` and `--endless_sleep` - continue re-executing the program's arguments forever until interrupted, pausing for `--endless_sleep` seconds between each iteration (no delay by default).

# image_fix_faces.py
Arguments here with the same names as in `image_from_text.py` will have the same function, with the following exceptions. Run with `--help` to see the full list.

Because of the complexity of handling pipe separated per-segment prompts, none of the lora loading, deduplication or sorting prompt functionality is available here.

## inputs
`--input_filenames` - specify the images to be read. Instead of providing functionality to e.g. select random items from a folder, this can be offloaded to the shell with commands like `((Get-Item "folder\\*.png").FullName | Get-Random -Count 5)`.

`--prompts` and `--prompt_file` - If a prompt is `auto` it will be treated as a blank detailing prompt. If a prompt contains pipes (`|`) it will be split up and each piece will be assigned to a separate segment in the normal Impact ordering (top to bottom, left to right).

`--prompted_only` - Similarly to `image_from_text` which provides four or five default detailing pipelines, the program will attempt to detail with both the prompt provided and an 'empty prompt' made from the checkpoint positive stem. If you specify `--prompted_only`, this empty prompt pipeline will be disabled.

## segment management
`--segs` - allows the user to specify bounding boxes manually (this can be used to either avoid a face-like object, to specify one that seems to avoid autodetection, or to merge together segments that should be detailed together as one piece). Segments should be specified in the `left:top+width+height` format.

`--skip_segment` - delete the segment at the provided index from the segment map. Can be used to avoid an autodetected segment.

# image_remaker.py
Reads in images, attempts to determine their content, and rediffuses them. Arguments here with the same names as in `image_from_text.py` will have the same function, with the following exceptions. Run with `--help` to see the full list. Prompts can also be specified directly - as noted above, this argument does not support inline lora specification.

## inputs
`--input_filenames` - specify the images to be read. Images will be scaled if they have a pixel count more than 10% away from the common SDXL pixel counts of 1,048,576 (1024x1024) or 1,011,712 (1216x832).

## the denoising process
`--from_step` - the diffusion process is 25 steps - this argument specifies which step to start from when using a latent created by encoding the input image. The default is 14, but you can provide a list of values and each will be generated. The lower the value, the more of the original image / pose / composition will be lost. At a value of 0, the original image is entirely discarded and a new empty latent substituted in its place.

`--denoise_programs` - a set of tuples of floats, which are effectively pairs representing the `--rescale_denoise` and `--rescale` arguments to `image_from_text.py`. For example, if you pass `--denoise_programs 0.4,1.1 0.7,1.1` the image will be modified twice - upscaled to 110% of its original size and then denoised from `from_step` at a strength of 0.4, and from `from_step` at a strength of 0.7.

## generating the prompt
`--disable_tagger` - by default, images are tagged using the wd14 node. This can be disabled with this switch. 

`--banlist` - passed to the WD14 tagger (and the DTG llm) to specify these tags should not be included in generated prompts - useful if your tagger overdetects things like moles. By default, takes its value from a list in the `common.py` file.

`--degrade_prompt` - as the wd14 tagger's models are deterministic, they always generate the same prompt for the same image. If you would like some more randomness, this setting will remove all but a certain percentage of the detected tags. By default, this is set to 80%, i.e. 20% of detected tags are discarded. Only applies to tags generated by the tagger.

`--force_prompt` - the output of both the LLM and tagger can be overridden with this option, if you already know what your prompt should be but want to use this tool to assist in e.g. scene composition.

`--use_dtg` - disabled by default, DTG and its related arguments are available here in the same way as in `image_from_text`. If enabled, it will run on both tagger generated and `--force_prompt` forced prompts.

`--use_llm` - disabled by default, determines whether to include an LLM's intepretation of the input image in the prompt. The LLM's response is added after the tag section of the prompt.

`--llm_prompt`, `--llm_sysmessage` - configuration for the LLM. Sensible defaults are provided. Note - if the process already has tags developed due to the tagger, `--force_prompt` or `--use_dtg`, those tags are provided to the LLM.

## managing long running processes
`--skip` - attempts to skip over the first `n` items - used to resume an interrupted batch job.
