# comfyui-python-workflows
 Workflow scripts demonstrating some uses of [pydn's comfyui-to-python-extension](https://github.com/pydn/ComfyUI-to-Python-Extension).

 These scripts reflect certain assumptions about my own workflows (including that they are very SDXL focused) - they may not match up with your needs exactly, but being released under [the MIT License](LICENSE.md), you should feel free to modify them to suit your needs. At a minimum however, you should look at `checkpoints.py` to set the paths for your checkpoints, and if you use the tagging functionality in `image_remaker.py`, you will probably want to consider setting the list of banned tags in `common.py` depending on what your preferred models tend to overdetect / what you do not want to see.

# dependencies
These scripts assume the presence of certain ComfyUI nodes, which they load as part of their execution. These include:
* [Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
* [Image Saver](https://github.com/alexopus/ComfyUI-Image-Saver)

If you want to use the image remaker script, you will want:
* [WD14Tagger](https://github.com/pythongosssss/ComfyUI-WD14-Tagger)

If you want to use llama to interrogate your images in the remaker you will need [llama_cpp-python](https://github.com/abetlen/llama-cpp-python) installed and working. This is also required to make use of [dtg](https://huggingface.co/KBlueLeaf/DanTagGen-delta-rev2) for prompt supplementation. If you use either, make sure to adjust the paths to the relevant gguf models in `checkpoints.py`. 

If you want to use the layered image saving support (saving each detailer result over the original image in its own layer as an ORA image file), you will need the [layeredimage](https://github.com/FHPythonUtils/LayeredImage/) Python library installed in your comfy environment. This has caused some users conflicts depending on which other nodes they are using.

The scripts assume an intermediate level of python capability - you may need to tailor them to your own environment, a task I cannot assist with.

# using the scripts
 These scripts should be run using the same Python environment you use to run Comfy, from the folder containing your ComfyUI folder. 
 
 An example session with the scripts might look like this - you start by running the script to generate some raw images. In this case, you disable face detailing because you want to look through the images first and only send the promising ones to that next stage.
 ```PowerShell
PS 02/11/2024 16:50:46> ..\python_embeded\python.exe C:\Users\curious\Documents\GitHub\comfy_python\image_from_text.py --checkpoints sxzluma_real --prompts "1boy, solo, a wizard is drinking coffee in a cafe" --sizes 832,1024 --skip_detailing --rescale 1.1 --rescale_denoise 0.4 --seeds -1 -1 -1 -1 --output_folder raw
```

You end up with eight images in a subfolder `raw` in your ComfyUI output folder - two each for each seed, representing the before and after of the rescaling process. You pick through them and select two or three that you like, and decide to detail their faces. In this case, you will leave the task of identifying the faces up to an ultralytics provider. You delete the files you don't want to continue with, so only your good candidates remain in the folder.

```PowerShell
PS 02/11/2024 17:07:11> ..\python_embeded\python.exe C:\Users\curious\Documents\GitHub\comfy_python\image_fix_faces.py --checkpoints sxzluma_real --input_filenames (Get-Item output\raw\*.png).FullName --prompts "green_eyes" --steps 20 --denoise 0.4 --cycles 4 --noise 0.1 --seeds -1 -1 -1 -1 --output_folder detailed
```

This loads our images, detects their faces, and rediffuses them with a four stage, 20 step process. In this case, you get two images out for each image in - one using the provided prompt, `green_eyes`, and another with a blank prompt.

Finally, we cherrypick a result we like.

![image](https://github.com/user-attachments/assets/40ef9599-4554-4581-b42b-b79eeb859bc7)

Later, you decide to take this original image and iterate on it. You put it in a folder named "step1", and run the following command:

```PowerShell
PS 10/11/2024 01:01:27> foreach ($i in 1..10) { ..\python_embeded\python.exe C:\Users\curious\Documents\GitHub\comfy_python\image_remaker.py --checkpoints * --degrade_prompt 0.5 --use_dtg --dtg_rating safe --dtg_target "<|long|>" --dtg_temperature 0.5 --from_step 4 --frontload_tags rating_safe --mandatory_tags "hat|witches_hat" --output_folder "output\step$($i+1)\" --input_filenames (Get-Item "output\step$i\*.png").FullName  --skip_original_face --single_detailer 3 }
```

Here, we take our source image, run a wd14 tagger over it, throw away 50% of the resulting tags, and then use the DTG llm to 'refill' our prompt back to approximately 40 tags (what DTG calls the '<|long|>' target). If we see the tag 'hat' in the results, we will add the tag 'witches_hat'. We can see this tag mutation process in action:
```
20241110 01:06:51: ** degrading 21 tags down to 10
20241110 01:06:51: ** - tagger tags were looking_at_viewer, black_hair, 1boy, holding, jewelry, sitting, upper_body, outdoors, solo_focus, day, necklace, tree, cup, facial_hair, umbrella, chair, table, plant, holding_cup, mug, realistic
20241110 01:06:51: ** - tagger tags now looking_at_viewer, black_hair, solo_focus, day, necklace, tree, facial_hair, chair, plant, realistic
20241110 01:06:55: ** dtg returned the following tags, which will replace the originals: ['looking at viewer', 'black hair', 'solo focus', 'day', 'necklace', 'tree', 'facial hair', 'chair', 'plant', 'realistic', 'pencil skirt', 'smile', 'table', 'holding', 'crossed arms', 'potted plant', 'beard', 'brown eyes', 'glasses', 'hand on own cheek', 'crossed legs', 'long hair', 'open book', 'outdoors', 'old man', 'old', 'wrinkled skin', 'grey hair', 'blonde hair', 'hand on own face', 'building', 'old woman', 'jewelry', 'grey eyes', 'mustache', 'book', 'indoors', 'skirt', 'holding book', 'sitting', 'window']
```

(Note, the loss of 1boy in the original tags means the model will now almost certainly drift to producing a female character due to training bias.) We then rediffuse the original image from step 4, detailing the face with a single detailer pipe. We then loop over this process ten times, choosing a new checkpoint each time. 

Each of the three scripts has a large number of command line switches that let you tune their behaviour, which the rest of this readme file deals with.

# image_from_text.py
`--checkpoints` - must be one or more of the shortnames used to define checkpoint tuples in `checkpoints.py` or `*`, which will choose one at random (this is the default.)

`--fd_checkpoint` - as for `--checkpoints`, but can only be a single value. If provided, the image will be detailed with this model in addition to whatever was used to create it. Can be helpful when you have a stable diffusion 1.5 model which does particularly good faces.

`--prompts` or `--prompt_file` - either a list of strings which form your positive prompts or a text file containing one prompt per line. Lines beginning with `#` are skipped. Prompts support inline specification of loras using the familiar `<lora:lname:model_str:clip_str>` syntax - this works in most places you would expect it to, but not all (e.g. in detailer prompts) - in-prompt lora handling is very fiddly and it is low on my list of priorities to fix. If you do use loras in this way, your prompts will be grouped together by the loras they use and then processed in "buckets". These loras will be applied in addition to anything else already defined in the checkpoint tuple. 

Prompts are modified by the default positive stem provided in the checkpoint tuple, as well as any values set by `--frontload_tags`, `--frontload_neg`, or `--mandatory_tags`. The frontload_tags flags are added between the checkpoint defaults and your prompts, the items in mandatory tags are added towards the end. Mandatory tags can be given a triggering condition using a pipe symbol: `--mandatory_tags "sky|(clouds:1.2)"` will add the clouds tag to the prompt only if the sky tag is present when it is processed. Mandatory tags are processed sequentially, so you can chain them together.

`--use_dtg` the [dtg](https://huggingface.co/KBlueLeaf/DanTagGen-delta-rev2) LLM can be used to augment your prompt with other tags that would be contextually relevant. If you use this flag, you need to have llama_cpp_python installed, have downloaded a gguf checkpoint from the linked repository, and set the key in your checkpoints file appropriately. DTG has several other relevant options - `--dtg_rating`, `--dtg_target` and `--dtg_temperature`. Check the linked model card for more information on how these work.

`--noise` adds random noise to the embeddings generated from your prompts. Inconsitent because the noise added to the regular embedding and its pooled output are independent, but can still produce some interesting effects at low strengths, say 0.1.

`--rerun_lora` - reruns any prompts that include lora strings with and without the lora loaded, used for testing strength/effectiveness.

`--sizes` - a list of comma separated integer sizes. The default is a list of three sizes - `1024,1024`, `832,1216`, and `1216,832`.

`--seeds` - a list of seed values. -1 generates a seed in the range of 1 to 1125899906842624. -2 does the same, but the seed is generated and then locked at the time the arguments are parsed, i.e. it is not rerandomised as the process continues / checkpoints are switched, etc etc.

`--rescale`, `--rescale_denoise` - a form of high-resolution fixing, the image will be subject to bicubic latent upscaling and then re-diffused for 12 steps at the provided denoising strength. Generally, both the original and upscaled image are included in the output, but only the second image will be detailed if detailing is requested.

`--skip_original_face` - if detailing is turned on, the original, undetailed face will not be saved.
`--skip_detailing` - do not perform any detailing. Detailing is otherwise on by default.
`--single_detailer` - by default, the script carries out four different detailing processes (five if an `fd_checkpoint` is specified) 
- 0: the unmodified model object paired with the provided positive prompt, 
- 1:  unmodified model object paired with the checkpoint's default positive prompt
- 2: the model modified by FreeU_V2 paired with the provided positive prompt,
- 3: the model modified by FreeU_V2 paired with the checkpoint's default positive prompt

By providing an index like 0, 1, 2, or 3, only that detailing process will be executed. The default is -1, which means all the processes are run.

`--additional_detailer_seeds` - the detailer processes described above generally run once for each seed, but you can provide additional seeds here to generate further alternative face candidates.

`--save_ora` - if you are detailing, also saves an .ORA file containing the base image and then the differences from it for each detailer process stacked on top as layers. Can be useful for picking and choosing faces between different generations.

`--diff_start` and `--diff_stop` - hold off on diffusing images outside of these times.
`--hold_file` - hold off on diffusing images while this file exists - to be used to pause and restart the process, including remotely via a mounted filesystem.

`--endless` and `--endless_sleep` - continue re-executing the program's arguments forever until interrupted, pausing for `--endless_sleep` seconds between each iteration (no delay by default).

# image_fix_faces.py
Arguments here with the same names as in `image_from_text.py` will have the same function, with the following exceptions. Run with `--help` to see the full list.

`--input_filenames` - specify the images to be read.

`--prompts` and `--prompt_file` - If a prompt is the string `auto` it will be treated as a blank detailing prompt. If a prompt contains pipes (`|`) it will be split up and each piece will be assigned to a separate segment in the normal Impact ordering (top to bottom, left to right). Like in `image_from_text.py`, the program will attempt to detail with both empty prompts and provided prompts - the `--prompted_only` switch changes this so only the provided prompt is run.

`--segs` - allows the user to specify bounding boxes manually (this can be used to avoid a face-like object, or to merge two segments that should be detailed together into one). Specified in the `left:top+width+height` format.

`--skip_segment` - delete the segment at the provided index from the segment map. Can be used to avoid an autodetected segment.

# image_remaker.py
Reads in images, attempts to determine their content, and rediffuses them. Arguments here with the same names as in `image_from_text.py` will have the same function, with the following exceptions. Run with `--help` to see the full list.

`--input_filenames` - specify the images to be read. Images are automatically scaled if they have a pixel count more than 10% out from common SDXL resolutions like 1024x1024.

`--from_step` - the diffusion process is 25 steps - this argument specifies which step to start from when using a latent created by encoding the input image. The default is 14, but you can provide a list of values and each will be generated. The lower the value, the more of the original image / pose / composition will be lost. At 0, the original image is entirely discarded and a new empty latent substituted in its place.

`--denoise_programs` - a set of tuples of floats, which are effectively pairs representing the `--rescale_denoise` and `--rescale` arguments to `image_from_text.py`. For example, if you pass `--denoise_programs 0.4,1.1 0.7,1.1` the image will be modified twice - upscaled to 110% of its original size and then denoised from `from_step` at a strength of 0.4, and from `from_step` at a strength of 0.7.

`--use_dtg` - DTG is available here in addition to `image_from_text`. If enabled, it will run on both tagger generated and `--force_prompt` forced prompts.

`--use_llm`, `--llm_prompt`, `--llm_sysmessage` - whether or not to include an LLM's opinion on the image in the prompt, and its configuration if so. Defaults exist for `llm_prompt` and `llm_sysmessage`, or you can tailor them to deal with specific contexts. By default the LLM is disabled. When used, it generates "supplementary text" which is added after any tags developed by the tagger, `--force_prompt` or `--use_dtg`. If you have tags from any of those sources, they will be provided to the LLM to assist it.

`--disable_tagger` - by default, images are tagged using the wd14 node. This can be disabled with this switch. The contents of `--banlist` can be used to avoid tags you don't want to be detected in your image - by default, this loads a list from the `common.py` file.

`--force_prompt` - the output of both the LLM and tagger can be overridden with this option, if you already know what your prompt should be but want to use this tool to assist in e.g. scene composition.

`--degrade_prompt` - as the wd14 tagger's models are deterministic, they always generate the same prompt for the same image. If you would like some randomness, this setting will remove all but a certain percentage of the detected tags. By default, this is set to 80%, i.e. 20% of detected tags are discarded.

`--skip` - attempts to skip over the first `n` items - used to resume an interrupted batch job.
