import checkpoints
import common
import multiprocessing
import random
import re

# functions to invoke llama models, either in their visual mode, or using the
# dtg danbooru tag generator llm - https://huggingface.co/KBlueLeaf/DanTagGen-delta-rev2

def runWrappedLLMFunction(messages, fpointer):
    parent_connector, child_connector = multiprocessing.Pipe()
    process = multiprocessing.Process(target=fpointer, args=(messages, child_connector))
    process.start()
    process.join()
    results = parent_connector.recv()
    return results

def createLlamaPrompt(prompt, sys_message = None, image_tensor = None, temperature = 0.4):
    messages = [
        {'role': 'system', 'content': sys_message or 'You are an assistant who describes the content and composition of images. Describe only what you see in the image, not what you think the image is about. Be factual and literal. Do not use metaphors or similes. Be concise.'},
        {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]},
        temperature,
    ]
    if image_tensor != None:
        image_url = common.image_tensor_to_png_url(image_tensor)
        messages[1]['content'].append({'type': 'image_url', 'image_url': image_url})
    return messages

def runLlamaPromptThread(messages, connection):
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    
    llm = Llama( 
        model_path = checkpoints.llama_model_path,
        chat_handler = Llava15ChatHandler(clip_model_path = checkpoints.llama_clip_path),
        n_gpu_layers = -1, 
        n_ctx = 2048,
        chat_format = "llava-1-5", 
        seed = -1,
        logits_all = True, 
        verbose = False
    )

    results = llm.create_chat_completion(max_tokens=200, messages=messages[:-1], temperature=messages[-1])
    result = random.choice(results['choices'])['message']['content']
    connection.send(result)
    connection.close()

def runLlamaPromptWrapper(prompt, sys_message = None, image = None):
    return runWrappedLLMFunction(
        createLlamaPrompt(prompt, sys_message, image, temperature=0.4), 
        runLlamaPromptThread
    )

def createDTGPrompt(tags: list, dtg_rating = 'safe', artist = '<|empty|>', characters = '<|empty|>', copyrights = '<|empty|>', width = 1.0, height = 1.0, target = '<|very_long|>', temperature = 0.7):
    tags = [t.strip().replace('_', ' ') for t in tags if t.strip()]
    target_tag_count = {'<|very_short|>': 10, '<|short|>': 20, '<|long|>': 40, '<|very_long|>': 60}.get(target, 40)
    prompt = f'''quality: masterpiece
rating: {dtg_rating}
artist: {artist}
characters: {characters}
copyrights: {copyrights}
aspect ratio: {width / height:.1f}
target: {target}
general: {', '.join(tags)}<|input_end|>'''
    messages = {'prompt': prompt, 'tags': tags, 'target_tag_count': target_tag_count, 'temperature': temperature}
    return messages

def runDTGThread(messages, connection):
    from llama_cpp import Llama, LLAMA_SPLIT_MODE_NONE
    llm = Llama(
        model_path = checkpoints.dtg_model_path,
        n_ctx = 384,
        split_mode = LLAMA_SPLIT_MODE_NONE,
        n_gpu_layers = -1,
        verbose = False
    )
    
    original_tags = messages['tags']
    additions_tags = []
    text = messages['prompt']
    temperature = messages['temperature']

    attempts = 10

    while True:
        # carry out a completion
        results = llm.create_completion(
            text,
            temperature = temperature,
            top_p = 0.95,
            top_k = 100,
            max_tokens = 256,
            repeat_penalty = 1
        )
        text += results['choices'][0]['text']
        text = text.replace('</s>', '').replace('<s>', '')

        def process_tag(x):
            x = x.strip()
            x = re.sub(r'(?<!\\)([()])', r'\\\1', x)
            if x in common.tag_banlist or x.replace(' ', '_') in common.tag_banlist:
                return None 
            return x

        # find the new parts of the text, convert to tags, clean, and add back in
        additions = text.split('<|input_end|>')[-1].strip().strip(',')
        additions_tags = list(set([t for t in (process_tag(tag) for tag in additions.split(',')) if t]))
        text = text.replace(additions, ', '.join(additions_tags))
        text = text.strip().replace('  <|', ' <|')
        
        if len(original_tags) + len(additions_tags) > messages['target_tag_count']:
            break

        # the original logic was a bit difficult for me to follow here - if we're below the
        # desired token cap, we reach for more, but if the LLM keeps returning the same number
        # of tokens each time, we desist after _n_ attempts (defaults to 5). however max_retry
        # is greater than 1 this never happens as retry is set to max_retry in all branches
        # immediately after the test. I am just going to do something simpler
        if attempts == 0:
            break
        attempts -= 1
    
    result_tuple = (text, original_tags + additions_tags, additions_tags)
    connection.send(result_tuple)
    connection.close()

def runDTGPromptWrapper(tags: list, dtg_rating, image_width, image_height, target, temperature):
    return runWrappedLLMFunction(
        createDTGPrompt(tags, dtg_rating = dtg_rating, width = image_width, height = image_height, target = target, temperature = temperature),
        runDTGThread
    )
