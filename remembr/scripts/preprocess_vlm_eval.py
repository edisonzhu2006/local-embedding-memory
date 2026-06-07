import json
import numpy as np
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from time import strftime, localtime
import numpy as np
import tqdm
import re
import time
import uuid
import sys
import os, sys
import pickle as pkl
from PIL import Image as PILImage
import glob
from pathlib import Path
from dataclasses import asdict
import argparse
import traceback
import cv2 
# load this directory
sys.path.append(sys.path[0] + '/..')
from remembr.agents.remembr_agent_vlm_emb import ReMEmbRAgentVLM
from remembr.agents.vlm_non_agent import VLMNonAgent
from remembr.memory.memory import VLMMemoryItem
from remembr.memory.milvus_memory_vlm import MilvusVLMMemory
from remembr.embedder.embedders import VLMEmbeddings
from remembr.memory.memory_factory import MemoryFactory
from remembr.memory.video_memory import VideoMemory
from remembr.memory.memory import MemoryItem
# logging info
# from contextlib import redirect_stdout, redirect_stderr
# import datetime
# LOG_FILE_DIR = "./scripts/logs"
# os.makedirs(LOG_FILE_DIR, exist_ok=True)
# LOG_FILE_NAME = f"preprocess_vlm_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
# LOG_FILE_PATH = os.path.join(LOG_FILE_DIR, LOG_FILE_NAME)
# print(f"Logging to {LOG_FILE_PATH}")

DEFAULT_QUESTION_MESSAGE_NAVQA = """```json
        {{
            "type_reasoning": "-input your reasoning in here for the type of question-", 
            "type": "-input the type of answer that is expected based only on the question: position, binary, time, or text. Be sure to then fill in that selected category.",
            "answer_reasoning", "-input your reasoning in here for the answer. If you do not know the answer, provide your best guess for the answer type you provide.-", 
            "text": "--a text answer here--",
            "binary": "yes/no",
            "position": "[x,y,z]",
            "orientation": "[-.92]", 
            "time": 5.3,
            "duration": 2.4,
        }}
        ```
        """

def hms_to_seconds(ts: str) -> float:
    parts = ts.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 1:   # SS(.sss)
        return parts[0]
    elif len(parts) == 2: # MM:SS(.sss)
        return parts[0]*60 + parts[1]
    elif len(parts) == 3: # HH:MM:SS(.sss)
        return parts[0]*3600 + parts[1]*60 + parts[2]
    else:
        raise ValueError(f"Invalid timestamp {ts}")

def parse_json(string):
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    return eval(parsed)

# we can have binary, position-based, time-based, or description-based. let's answer accordingly
def evaluate_output(qa_instance, predicted):

    out_error = {}

    q_type = qa_instance['type']
    # import ipdb; ipdb.set_trace()
    if 'position' in q_type:

        answer = np.array(qa_instance['answers']['position'])

        # compute L2 loss between predicted['binary'] and answer
        if type(predicted['position']) == str:
            predicted['position'] = eval(predicted['position'])
        pred_pos = np.array(predicted['position'])

        dist = np.linalg.norm(answer - pred_pos)

        out_error['position_error'] = dist

    elif 'binary' in q_type:

        answer = qa_instance['answers']['text'][1] # we made this assumption in other examples that binary answer is the second one

        if 'binary' in predicted and (predicted['binary'].lower() == "yes" or predicted['binary'].lower() == "no"):
            # get correct/incorrect label
            if predicted['binary'].lower() == answer.lower():
                correct = 1
            else:
                correct = 0

            out_error['binary_iscorrect'] = correct

    elif 'time' in q_type:

        answer = np.array(qa_instance['answers']['time']) # answer and dist are in minutes
        # import ipdb; ipdb.set_trace()
        ## answer = array(-0.29)
        ## predicted['time'] = None
        ## type(predicted['time']) = <class 'NoneType'>
        # compute L2 loss between predicted['binary'] and answer
        if predicted['time'] is None:
            predicted['time'] = float('inf')
        elif type(predicted['time']) == str:
            # predicted['time'] = eval(predicted['time'])
            # predicted['time'] = '03:27:30'
            ## eval(predicted['time']) = 124650.0
            predicted_time_seconds = float(hms_to_seconds(predicted['time']))
            
            # Extract current time from the question text
            # Question format: "The current time is 2023-01-16 11:03:02"
            time_match = re.search(r'The current time is \d{4}-\d{2}-\d{2} (\d{2}):(\d{2}):(\d{2})', qa_instance['question'])
            if time_match:
                current_hour, current_minute, current_second = map(int, time_match.groups())
                curr_time_str = f"{current_hour}:{current_minute}:{current_second}"
                curr_time_seconds = float(hms_to_seconds(curr_time_str))
                
                # Convert predicted absolute time to relative time (minutes)
                predicted_relative_minutes = (predicted_time_seconds - curr_time_seconds) / 60.0
                predicted['time'] = predicted_relative_minutes
            else:
                raise ValueError("Could not find current time in the question")

        pred_time = np.array(predicted['time'])

        dist = abs(answer - pred_time)

        out_error['time_error'] = dist

    elif 'duration' in q_type:

        answer = np.array(qa_instance['answers']['duration'])  # answer is in minutes

        # compute L2 loss between predicted['binary'] and answer
        if type(predicted['duration']) == str:
            # import ipdb; ipdb.set_trace()
            predicted_duration_seconds = eval(predicted['duration'])
            # Convert seconds to minutes for comparison
            predicted['duration'] = predicted_duration_seconds / 60.0
        pred_time = np.array(predicted['duration'])

        dist = abs(answer - pred_time)

        out_error['duration_error'] = dist

    elif 'text' in q_type:
        answer = qa_instance['answers']['text']
        out_error = {'answer': answer}

    else:
        raise Exception("We do not support question type " + q_type)

    return out_error

def answer_question(model, question, qa_instance):

    print(f'Question: {question}')

    parsed = None
    while True:
        try:

            start_time = time.time()
            response = model.query(question)
            end_time = time.time()

            elapsed = end_time - start_time

            parsed = asdict(response)
            out_error = evaluate_output(qa_instance, parsed)

            print("Time elapsed", elapsed)

        except Exception as e:
            print(parsed)
            print(e)
            traceback.print_exception(*sys.exc_info()) 
            continue

        return_dict = {"response": parsed}
        return_dict.update(parsed)
        return_dict['error'] = out_error
        return_dict['elapsed'] = elapsed

        return return_dict

def load_memory(args, qa_instance, embedder:VLMEmbeddings|None, unit='seconds'):
    # Here we load everything needed to load a MilvusDB instance neatly
    start_time = qa_instance['start_time']
    end_time = qa_instance['end_time']
    convert_file = args.convert_file

    if args.framework == 'remembr':
        memory = MemoryFactory.create_memory(
            backend=args.memory_backend,
            db_collection_name="eval_memory_for_vlm",
            embedder=embedder,
            storage_path=args.memory_storage,
            use_vlm_embedding=True,
            time_offset=start_time if unit=='seconds' else start_time/1e6,
            dim=args.emb_dim,
            retriever_k=args.top_k,
            respond_with_score=args.add_score_info,
        )
    elif args.framework == 'vlm_only':
        memory = VideoMemory(start_time=start_time if unit=='seconds' else start_time/1e6)
    else:
        raise Exception("We only support [remembr, vlm_only] for now")
    memory.reset()
    with open(convert_file, 'r') as f:
        out = json.load(f)

    outputs = []

    # Compute start idx
    all_start_times = np.array([float(x['start_time']) for x in out])
    diff = all_start_times - start_time
    start_idx = np.argmin(np.abs(diff))

    # Compute end idx
    all_end_times = np.array([float(x['end_time']) for x in out])
    diff = all_end_times - end_time
    end_idx = np.argmin(np.abs(diff))
    
    # load all pkl files from seq_id
    pkl_files = glob.glob(os.path.join(args.coda_dir, str(args.sequence_id), '*.pkl'))
    pkl_files.sort(key=lambda x: float(x.split('/')[-1][:-4]))

    dict_fns_to_emb = {}
    if args.backend == 'ol' and out:
        cache_dir = os.path.join(args.coda_dir, str(args.sequence_id), ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_emb = os.path.join(cache_dir, f"{args.online_model_nickname}_image_embeds.npy")
        cache_list = os.path.join(cache_dir, f"{args.online_model_nickname}_image_filenames.json")
        if os.path.exists(cache_emb) and os.path.exists(cache_list):
            with open(cache_list, "r", encoding="utf-8") as f:
                cached_fns = json.load(f)
            print(f"Loading {len(cached_fns)} embeddings from {cache_emb}...")
            arr = np.load(cache_emb)
            for i, fn in enumerate(cached_fns):
                dict_fns_to_emb[fn] = arr[i]
        else:
            print(f"Caching {len(out)} image embeddings to memory. This may take a while...")
            for entity in out:
                if 'image_filenames' in entity and entity['image_filenames'] is not None:
                    print(f"Caching {len(entity['image_filenames'])} image embeddings to memory. This may take a while...")
                    embeddings_temp = embedder.embed_documents(["[IMG]" + img_path for img_path in entity['image_filenames']])
                    mean_embeddings = [sum(embedding) / len(embedding) for embedding in zip(*embeddings_temp)]
                    dict_fns_to_emb[Path(entity['image_filenames'][0]).name] = mean_embeddings
                    print(f"Finished caching {len(entity['image_filenames'])} image embeddings")
            with open(cache_list, "w", encoding="utf-8") as f:
                json.dump(list(dict_fns_to_emb.keys()), f)
                print(f"Finished dumping {len(dict_fns_to_emb.keys())} embeddings to {cache_list}")
            np.save(cache_emb, np.array(list(dict_fns_to_emb.values())))
            print(f"Finished saving {len(dict_fns_to_emb.values())} embeddings to {cache_emb}")
    elif args.hf_model_id == 'youzexue/QQMM-embed-v2' and out:
        cache_dir = os.path.join(args.coda_dir, str(args.sequence_id), ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_emb = os.path.join(cache_dir, f"QQMM-embed-v2_image_embeds.npy")
        cache_list = os.path.join(cache_dir, f"QQMM-embed-v2_image_filenames.json")
        if os.path.exists(cache_emb) and os.path.exists(cache_list):
            with open(cache_list, "r", encoding="utf-8") as f:
                cached_fns = json.load(f)   
            arr = np.load(cache_emb)
            print(f"Loading {len(cached_fns)} QQMM embeddings from cache...")
            for i, fn in enumerate(cached_fns):
                dict_fns_to_emb[fn] = arr[i]
            print(f"Finished loading {len(cached_fns)} QQMM embeddings from cache!")
        else:
            print(f"Caching {len(out)} image embeddings to memory. This may take a while...")
            for entity in out:
                if 'image_filenames' in entity and entity['image_filenames'] is not None:
                    print(f"Caching {len(entity['image_filenames'])} image embeddings to memory. This may take a while...")
                    embeddings_temp = embedder.embed_documents(["[IMG]" + img_path for img_path in entity['image_filenames']])
                    mean_embeddings = [sum(embedding) / len(embedding) for embedding in zip(*embeddings_temp)]
                    dict_fns_to_emb[Path(entity['image_filenames'][0]).name] = mean_embeddings
                    print(f"Finished caching {len(entity['image_filenames'])} image embeddings")
            with open(cache_list, "w", encoding="utf-8") as f:
                json.dump(list(dict_fns_to_emb.keys()), f)
                print(f"Finished dumping {len(dict_fns_to_emb.keys())} embeddings to {cache_list}")
            np.save(cache_emb, np.array(list(dict_fns_to_emb.values())))
            print(f"Finished saving {len(dict_fns_to_emb.values())} embeddings to {cache_emb}")

    for i in range(start_idx, end_idx+1):
        item = out[i]
        entity = {
            'position': item['position'],
            'time': item['time'] if unit == "seconds" else item['time'] / 1e6,
            'caption': item['caption'],
            'theta': item['theta'],
            'image_file_path': item['image_file_path'] if 'image_file_path' in item else None,
            'image_filenames': item['image_filenames']
        }
        outputs.append(entity)
        entity = VLMMemoryItem.from_dict(entity)

        if args.framework == 'remembr':
            memory.insert(entity, vlm_embedding=dict_fns_to_emb[Path(item['image_filenames'][0]).name] if dict_fns_to_emb else None) # will make up embeddings inside
        elif args.framework == 'vlm_only':
            memory.insert(entity)
        else:
            raise Exception("We only support [remembr, vlm_only] for now")
    return memory, outputs


def main(args, embedder=None):
    if args.framework == 'remembr':
        base_llm = args.base_llm
        agent = ReMEmbRAgentVLM(llm_type=base_llm, num_ctx=args.num_ctx, temperature=args.temperature, debug=args.debug, prompt_dir="prompts/navqa_prompts")
    elif args.framework == 'vlm_only':
        agent = VLMNonAgent(llm_type=args.base_llm, num_ctx=args.num_ctx, temperature=args.temperature, question_message=args.question_message, prompt_dir="prompts/navqa_prompts")
    else:
        raise Exception("We only support remembr for now")
    # import ipdb; ipdb.set_trace()
    data = json.load(open(args.qa_file, 'r'))
    # data is question data
    # import ipdb; ipdb.set_trace()
    data = data['data']

    running_successes = 0
    num_binary = 0

    running_pos_error = 0
    num_position = 0

    running_time_error = 0
    num_time = 0

    running_duration_error = 0
    num_duration = 0
    
    responses = []

    for i in tqdm.tqdm(range(0, len(data)), total=len(data)):

        print(f"Evaluating {i} out of {len(data)}")

        qa_instance = data[i]
        question = qa_instance['question']
        context = qa_instance['context']
        start_time = qa_instance['start_time']
        answers = qa_instance['answers']
        id = qa_instance['id']
        
        if (qa_instance['type'] == 'text'):
            print("Skipping text questions for now")
            responses.append({}) # this means skipped!
            continue
        memory, instance_captions = load_memory(args=args, qa_instance=data[i], embedder=embedder)
        # import ipdb; ipdb.set_trace()
        if len(instance_captions) == 0: # ISSUE
            print("Length of Instance Captions is 0. It should not be")
            import pdb; pdb.set_trace()

        print("HISTORY LENGTH", len(instance_captions))

        agent.set_memory(memory)

        out_dict = answer_question(agent, question, qa_instance)

        out_dict['question'] = qa_instance['question']
        out_dict['id'] = id
        error_dict = out_dict['error']

        # keep track of how many of each. usually all CSVs are one type only
        if qa_instance['type'] == 'position':
            num_position += 1
            if 'position_error' in error_dict:
                running_pos_error += error_dict['position_error']
        elif qa_instance['type'] == 'binary':
            num_binary += 1
            if 'binary_iscorrect' in error_dict:
                running_successes += error_dict['binary_iscorrect']
        elif qa_instance['type'] == 'time':
            num_time += 1
            if 'time_error' in error_dict:
                running_time_error += error_dict['time_error']
        elif qa_instance['type'] == 'duration':
            num_duration += 1
            if 'duration_error' in error_dict:
                running_duration_error += error_dict['duration_error']
        
        print("Question:", question)
        if 'response' in out_dict:
            print("Response:", out_dict['response'])
        print("Running Binary QA accuracy", running_successes/(num_binary+1))
        print("Running Spatial Error", running_pos_error/(num_position+1))
        print("Running Temporal Error", running_time_error/(num_time+1))
        print("Running Duration Error", running_duration_error/(num_duration+1))

        responses.append(out_dict)


    # save all_questions into json
    out_json = {
        "version": 0.1,
        "responses": responses
    }

    # save the outputs
    os.makedirs(args.out_dir, exist_ok=True)
    if "QQMM" in args.model:
        name = 'remembr+gemini-2.5-flash+hf+QQMM-embed-v2+0+3584'+'__'+ Path(args.convert_file).name[:-5] + "__" +  Path(args.qa_file).name[:-5] + args.postfix
    else:
        name = args.model+'__'+ Path(args.convert_file).name[:-5] + "__" +  Path(args.qa_file).name[:-5] + args.postfix
    with open(os.path.join(args.out_dir, f'{name}.json'), 'w') as f:
        # to_save = json.dumps(out_json, indent=4)
        json.dump(out_json, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='Long Horizon Robot QA',
                        description='Runs various LLMs on the QA dataset',)
    
    # data-specific args
    parser.add_argument("--model", type=str, default="remembr+gpt-4o+oc+ViT-B-32+laion2b_s34b_b79k+512")

    # choice 1: using input folder flag: doing video batch 
    # parser.add_argument("--input_folder", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./data_vlm/")
    parser.add_argument("--coda_dir", type=str, default="./coda_data/")
    parser.add_argument("--sequence_id", type=int, default=0)
    # choice 2: using single video - give two flags
    parser.add_argument("--qa_file", type=str, default=None)
    parser.add_argument("--convert_file", type=str, default=None)

    parser.add_argument("--out_dir", type=str, default="./output/")
    parser.add_argument("--postfix", type=str, default='_0')
    parser.add_argument("--question_message", type=str, default=DEFAULT_QUESTION_MESSAGE_NAVQA)
    # llm-specific args
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_ctx", type=int, default=8192*8)

    # remembr specific args
    parser.add_argument("--db_ip", type=str, default='127.0.0.1')
    parser.add_argument("--debug", action='store_true', help="If true, prints out debug info")

    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fp_16", action='store_true', help="If true, uses fp16 for hf model")
    parser.add_argument("--vlm_text_prompts", type=str, default="")
    parser.add_argument("--vlm_image_prompts", type=str, default="Imagine the scene.")

    # NEGLECT THESE FLAGS; DONNOT USE THEM
    # embedding-related flags
    parser.add_argument("--framework", type=str, default='remembr', choices=["remembr", "vlm_only"]) 
    parser.add_argument("--base_llm", type=str, default='gpt-4o')
    parser.add_argument("--backend", type=str, default='oc', choices=["oc", "hf", "ol"])
    parser.add_argument("--oc_model", type=str, default='ViT-SO400M-14-SigLIP-384')
    parser.add_argument("--oc_pretrained", type=str, default='webli')
    parser.add_argument("--hf_model_id", type=str, default='google/paligemma2-3b-mix-224')
    parser.add_argument("--vlm_layer", type=int, default=1)
    parser.add_argument("--emb_dim", type=int, default=1024)
    parser.add_argument("--online_model_nickname", type=str, default=None, help="If backend is 'ol', provide the online model nickname here. Supported models: google-multimodal-embedding-model, seed-clip-vit-l-14")


    parser.add_argument("--memory_backend", type=str, default='milvus', choices=["milvus", "faiss"])
    parser.add_argument("--memory_storage", type=str, default='./output/memory_storage')

    # ablation study controls
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--add_score_info", action='store_true', help="If true, tool call will respond with the cosine similarity score between the query and the retrieved frame")

    # qqmm 4-bit quantization flag
    parser.add_argument("--quant", action='store_true', help="If true, uses 4-bit quantization for QQMM")

    args = parser.parse_args()
    if args.model:
        # model example: remembr+gpt-4o+oc+ViT-SO400M-14-SigLIP-384+webli+1024+google/paligemma2-3b-mix-224+1
        ## gpt-4o is the base llm
        ## oc is the backend
        ## ViT-SO400M-14-SigLIP-384 is the oc model
        ## webli is the oc pretrained
        ## 1024 is the emb_dim
        ## google/paligemma2-3b-mix-224 is the hf model id
        ## 1 is the vlm layer
        # model format: framework+base_llm+backend+oc_model+oc_pretrained+emb_dim+hf_model_id+vlm_layer
        components = args.model.split('+')
        args.framework = components[0]
        args.base_llm = components[1]

        args.backend = components[2] if len(components) > 2 else None
        if args.backend == 'oc':
            args.oc_model = components[3]
            args.oc_pretrained = components[4]
            args.emb_dim = int(components[5])
        elif args.backend == 'hf':
            args.hf_model_id = components[3]
            args.vlm_layer = int(components[4])
            args.emb_dim = int(components[5])
        elif args.backend == 'ol':
            args.online_model_nickname = components[3]
            args.emb_dim = int(components[4])
        elif args.backend is None:
            # using default mxbai embedder dimension.
            args.emb_dim = 1024
        else:
            raise Exception("Backend must be oc or hf")

    if args.framework == 'remembr':
        embedder = VLMEmbeddings(
            args.device,
            backend=args.backend,
            oc_model=args.oc_model,
            oc_pretrained=args.oc_pretrained,
            hf_model_id=args.hf_model_id,
            vlm_layer=args.vlm_layer,
            batch_size=args.batch_size,
            fp_16=args.fp_16,
            emb_dim=args.emb_dim,
            vlm_text_prompts=args.vlm_text_prompts,
            vlm_image_prompts=args.vlm_image_prompts,
            online_model_nickname=args.online_model_nickname,
            quant=args.quant
        )
    else:
        embedder = None

    if not args.qa_file and not args.convert_file:
        # choice 1
        # get all files in the data directory, it is seq by seq
        qa_files = glob.glob(os.path.join(args.data_dir, 'questions', str(args.sequence_id), 'human_qa.json'))
        convert_files = glob.glob(os.path.join(args.data_dir, 'convert', str(args.sequence_id), '*.json'))
        print(f"Loaded {len(qa_files)} QA files of {qa_files} and {len(convert_files)} convert files of {convert_files}")
        qa_files = sorted(qa_files)
        convert_files = sorted(convert_files)
        if len(qa_files) != len(convert_files):
            print("Warning: The number of QA files and convert files do not match!")
        for qa_file, convert_file in zip(qa_files, convert_files):
            # assert qa_file.split('/')[-1].split('_qa.json')[0] == convert_file.split('/')[-1].split('_frames.json')[0], f"QA file {qa_file} and convert file {convert_file} do not match!"
            
            args.qa_file = qa_file
            args.convert_file = convert_file

            name = os.path.join(args.out_dir, f'{args.model}__{Path(args.convert_file).name[:-5]}__{Path(args.qa_file).name[-5]}{args.postfix}.json')
            
            if os.path.exists(name):
                print(f"Output file {name} already exists! Skipping...")
                continue
            
            print(f"Processing {qa_file} and {convert_file}")          
            main(args, embedder)
    else:
        print(f"Processing {args.qa_file} and {args.convert_file}")
        main(args, embedder)
# For remembr+VLM+image embedding pipeline: gemini-2.5-pro + oc + ViT-B-32 + laion2b_s34b_b79k + 512
"""
python -m scripts.preprocess_vlm_eval \
  --model remembr+gemini-2.5-pro+oc+ViT-B-32+laion2b_s34b_b79k+512 \
  --data_dir ./data_vlm/ \
  --coda_dir ./coda_data/ \
  --sequence_id 0 \
  --out_dir ./data_vlm/test_result/gemini-2.5-pro+oc+ViT-B-32+laion2b_s34b_b79k+512 \
  --postfix _0 \
  --temperature 0.7 \
  --num_ctx 16384 \
  --memory_backend faiss \
  --debug \
  --top_k 5 \
  --add_score_info
"""
# For remembr+VLM+image embedding pipeline: gpt-4o + oc + ViT-B-32 + laion2b_s34b_b79k + 512
"""
python -m scripts.preprocess_vlm_eval \
  --model remembr+gpt-4o+oc+ViT-B-32+laion2b_s34b_b79k+512 \
  --data_dir ./data_vlm/ \
  --coda_dir ./coda_data/ \
  --sequence_id 0 \
  --out_dir ./data_vlm/test_result/gpt-4o+oc+ViT-B-32+laion2b_s34b_b79k+512 \
  --postfix _0 \
  --temperature 0.7 \
  --num_ctx 16384 \
  --memory_backend faiss \
  --debug \
  --top_k 5 \
  --add_score_info
"""
# For remembr+VLM+image embedding pipeline: gemini-2.5-flash + oc + ViT-SO400M-14-SigLIP-384 + webli + 1152
"""
python -m scripts.preprocess_vlm_eval \
  --model remembr+gemini-2.5-flash+oc+ViT-SO400M-14-SigLIP-384+webli+1152 \
  --data_dir ./data_vlm/ \
  --coda_dir ./coda_data/ \
  --sequence_id 0 \
  --out_dir ./data_vlm/test_result/gemini-2.5-flash+oc+ViT-SO400M-14-SigLIP-384+webli+1152 \
  --postfix _0 \
  --temperature 0.7 \
  --num_ctx 16384 \
  --memory_backend faiss \
  --debug \
  --top_k 5 \
  --add_score_info
"""
# For remembr+VLM+image embedding pipeline: gemini-2.5-flash + ol + seed1.6 + 2048
"""
python -m scripts.preprocess_vlm_eval \
  --model remembr+gemini-2.5-flash+ol+seed1.6+2048 \
  --data_dir ./data_vlm/ \
  --coda_dir ./coda_data/ \
  --sequence_id 0 \
  --out_dir ./data_vlm/test_result/gemini-2.5-flash+ol+seed1.6+2048 \
  --postfix _0 \
  --temperature 0.7 \
  --num_ctx 16384 \
  --memory_backend faiss \
  --debug \
  --top_k 5 \
  --add_score_info
  
"""
# For vlm_only pipeline: gemini-2.5-flash
"""
python -m scripts.preprocess_vlm_eval \
    --model vlm_only+gemini-2.5-flash \
    --data_dir ./data_vlm/ \
    --coda_dir ./coda_data/ \
    --sequence_id 0 \
    --out_dir ./data_vlm/test_result/vlm-only+gemini-2.5-flash \
    --postfix _0 \
    --temperature 0.0001 \
    --num_ctx 999999 \
    --debug
"""
# For qqmm 4-bit quantization on 5070ti
"""
python -m scripts.preprocess_vlm_eval \
    --model remembr+gemini-2.5-flash+hf+youzexue/QQMM-embed-v2+0+3584 \
    --data_dir ./data_vlm/ \
    --coda_dir ./coda_data/ \
    --sequence_id 0 \
    --out_dir ./data_vlm/test_result/gemini-2.5-flash_qqmm_k_5 \
    --postfix _0 \
    --temperature 0.7 \
    --num_ctx 9999999 \
    --debug \
    --memory_backend faiss \
    --top_k 5 \
    --add_score_info \
    --quant
"""
