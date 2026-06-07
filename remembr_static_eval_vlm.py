import json
import numpy as np
import tqdm
import time
import sys
import os, sys
import glob
import gc
import torch
from pathlib import Path
from dataclasses import asdict
import argparse
import traceback 
# load this directory
sys.path.append(sys.path[0] + '/..')
from langchain_huggingface import HuggingFaceEmbeddings
from remembr.agents.remembr_agent_vlm_emb import ReMEmbRAgentVLM
from remembr.agents.remembr_agent import ReMEmbRAgent
from remembr.agents.vlm_non_agent import VLMNonAgent
from remembr.agents.embedder_only_agent import EmbedderOnlyAgent
from remembr.memory.memory import VLMMemoryItem
# from remembr.memory.milvus_memory_vlm import MilvusVLMMemory
from remembr.embedder.embedders import VLMEmbeddings
from remembr.memory.memory_factory import MemoryFactory
from remembr.memory.video_memory import VideoMemory
from remembr.memory.memory import MemoryItem

def answer_question(model, question):

    print(f'Question: {question}')

    parsed = None
    while True:
        try:

            start_time = time.time()
            response = model.query(question)
            end_time = time.time()

            elapsed = end_time - start_time

            parsed = asdict(response)

            print("Time elapsed", elapsed)

        except Exception as e:
            print(parsed)
            print(e)
            traceback.print_exception(*sys.exc_info()) 
            continue


        return_dict = {"response": parsed}
        return_dict.update(parsed)
        return_dict['elapsed'] = elapsed

        return return_dict


def load_memory(args, embedder:VLMEmbeddings|HuggingFaceEmbeddings|None, start_time, end_time, caption_file, unit='microseconds'):
    if args.framework == 'remembr':
        # Here we load everything needed to load a MilvusDB instance neatly
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
    elif args.framework == 'remembr_caption_only':
        memory = MemoryFactory.create_memory(
            backend=args.memory_backend,
            db_collection_name="eval_memory_for_vlm_caption_only",
            embedder=embedder,
            storage_path=args.memory_storage,
            use_vlm_embedding=False,
            time_offset=start_time if unit=='seconds' else start_time/1e6,
            dim=args.emb_dim
        )
    elif args.framework == 'embedder_only':
        # Embedder-only: pure retrieval without LLM reasoning
        memory = MemoryFactory.create_memory(
            backend=args.memory_backend,
            db_collection_name="eval_memory_embedder_only",
            embedder=embedder,
            storage_path=args.memory_storage,
            use_vlm_embedding=True,  # Use VLM embeddings for retrieval
            time_offset=start_time if unit=='seconds' else start_time/1e6,
            dim=args.emb_dim,
            retriever_k=args.top_k,
            respond_with_score=True,  # Always return similarity scores
        )
    else:
        raise Exception("We only support [remembr, vlm_only, remembr_caption_only, embedder_only] for now")
    
    memory.reset()
    # import ipdb; ipdb.set_trace()
    with open(caption_file, 'r') as f:
        out = json.load(f)


    outputs = []

    if "start_time" in out[0]:
        # Compute start idx
        all_start_times = np.array([float(x['start_time']) for x in out])
        diff = all_start_times - start_time
        start_idx = np.argmin(np.abs(diff))

        # Compute end idx
        all_end_times = np.array([float(x['end_time']) for x in out])
        diff = all_end_times - end_time
        end_idx = np.argmin(np.abs(diff))
    else:
        # for caption only case, defaultly using all
        start_idx = 0
        end_idx = len(out) - 1



    dict_fn_to_emb = {}
    if args.backend == 'ol' and out:
        cache_dir = os.path.join(Path(out[0]["image_file_path"]).parent.parent, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_emb = os.path.join(cache_dir, f"{args.online_model_nickname}_image_embeds.npy")
        cache_list = os.path.join(cache_dir, f"{args.online_model_nickname}_image_filenames.json")
        if os.path.exists(cache_emb) and os.path.exists(cache_list):
            with open(cache_list, "r", encoding="utf-8") as f:
                cached_fns = json.load(f)
            arr = np.load(cache_emb)
            for i, fn in enumerate(cached_fns):
                dict_fn_to_emb[fn] = arr[i]
        else:
            for entity in out:
                if 'image_file_path' in entity and entity['image_file_path'] is not None:
                    print("Caching image embeddings to memory. This may take a while...")
                    dict_fn_to_emb[Path(entity['image_file_path']).name] = embedder.embed_query("[IMG]" + entity['image_file_path'])
            with open(cache_list, "w", encoding="utf-8") as f:
                json.dump(list(dict_fn_to_emb.keys()), f)
            np.save(cache_emb, np.array(list(dict_fn_to_emb.values())))

    elif args.hf_model_id == 'youzexue/QQMM-embed-v2' and out:
        cache_dir = os.path.join(Path(out[0]["image_file_path"]).parent.parent, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_emb = os.path.join(cache_dir, f"QQMM-embed-v2_image_embeds.npy")
        cache_list = os.path.join(cache_dir, f"QQMM-embed-v2_image_filenames.json")
        if os.path.exists(cache_emb) and os.path.exists(cache_list):
            with open(cache_list, "r", encoding="utf-8") as f:
                cached_fns = json.load(f)   
            arr = np.load(cache_emb)
            print(f"Loading {len(cached_fns)} QQMM embeddings from cache...")
            for i, fn in enumerate(cached_fns):
                dict_fn_to_emb[fn] = arr[i]
            print(f"Finished loading {len(cached_fns)} QQMM embeddings from cache!")
        else:
            print(f"Caching {len(out)} image embeddings to memory. This may take a while...")
            for idx, entity in enumerate(out):
                if 'image_file_path' in entity and entity['image_file_path'] is not None:
                    print(f"Caching image embedding {idx+1}/{len(out)}...")
                    # Clear cache before each embedding to reduce fragmentation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    dict_fn_to_emb[Path(entity['image_file_path']).name] = embedder.embed_query("[IMG]" + entity['image_file_path'])
                    # Clear cache after embedding to free intermediate tensors
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
            with open(cache_list, "w", encoding="utf-8") as f:
                json.dump(list(dict_fn_to_emb.keys()), f)
                print(f"Finished dumping {len(dict_fn_to_emb.keys())} embeddings to {cache_list}")
            np.save(cache_emb, np.array(list(dict_fn_to_emb.values())))
            print(f"Finished saving {len(dict_fn_to_emb.values())} embeddings to {cache_emb}")
        
    for i in range(start_idx, end_idx+1):

        item = out[i]
        # Extract position from caption file if available
        position = item.get('position', [0., 0., 0.])
        # Handle case where position might be ["unknown"] or a list of strings
        if isinstance(position, list) and len(position) > 0:
            if isinstance(position[0], str) and position[0] == "unknown":
                position = [0., 0., 0.]
            elif not isinstance(position[0], (int, float)):
                position = [0., 0., 0.]
        else:
            position = [0., 0., 0.]
        
        entity = {
            'position': position,  # Use actual position from caption file
            'time': item['time'] if unit == "seconds" else item['time'] / 1e6,
            'caption': item['caption'],
            'theta': 3.14, # we don't have theta info in the captions right now
            'image_file_path': item['image_file_path'] if 'image_file_path' in item else None
        }

        outputs.append(entity)

        
        entity = VLMMemoryItem.from_dict(entity) if args.framework not in ['remembr_caption_only'] else MemoryItem.from_dict(entity)

        if args.framework == 'remembr':
            memory.insert(entity, vlm_embedding=dict_fn_to_emb[Path(item['image_file_path']).name] if dict_fn_to_emb else None) # will make up embeddings inside
        elif args.framework == 'vlm_only':
            memory.insert(entity)
        elif args.framework == 'remembr_caption_only':
            memory.insert(entity, text_embedding=item['text_embedding'])
        elif args.framework == 'embedder_only':
            memory.insert(entity, vlm_embedding=dict_fn_to_emb[Path(item['image_file_path']).name] if dict_fn_to_emb else None)
        else:
            raise Exception("We only support [remembr, vlm_only, remembr_caption_only, embedder_only] for now")
        
    return memory, outputs

def main(args, embedder=None):

    if args.framework == 'remembr':
        base_llm = args.base_llm
        agent = ReMEmbRAgentVLM(llm_type=base_llm, num_ctx=args.num_ctx, temperature=args.temperature, debug=args.debug, prompt_dir=args.input_prompt_folder)
    elif args.framework == 'vlm_only':
        agent = VLMNonAgent(llm_type=args.base_llm, num_ctx=args.num_ctx, temperature=args.temperature, prompt_dir=args.input_prompt_folder)
    elif args.framework == 'remembr_caption_only':
        base_llm = args.base_llm
        agent = ReMEmbRAgent(llm_type=base_llm, num_ctx=args.num_ctx, temperature=args.temperature, debug=args.debug, prompt_dir=args.input_prompt_folder)
    elif args.framework == 'embedder_only':
        # Embedder-only agent: pure retrieval without LLM reasoning
        agent = EmbedderOnlyAgent(debug=args.debug)
    else:
        raise Exception("We only support [remembr, vlm_only, remembr_caption_only, embedder_only]")

    data = json.load(open(args.qa_file, 'r'))
    start_time = data['start_time']
    end_time = data['end_time']

    data = data['data']

    responses = []
    

    memory, instance_captions = load_memory(args, embedder, start_time, end_time, args.caption_file)
    
    agent.set_memory(memory)
    
    if len(instance_captions) == 0: # ISSUE
        print("Length of Instance Captions is 0. It should not be")
        import pdb; pdb.set_trace()

    print("HISTORY LENGTH", len(instance_captions))

    for i in tqdm.tqdm(range(0, len(data)), total=len(data)):
        agent.retrieval_logs.append("[RST]")

        print(f"Evaluating {i} out of {len(data)}")

        qa_instance = data[i]
        question = qa_instance['question']
        id = qa_instance['id']
        
        out_dict = answer_question(agent, question)

        out_dict['question'] = qa_instance['question']
        out_dict['id'] = id

        print("Question:", question)
        if 'response' in out_dict:
            print("Response:", out_dict['response'])

        responses.append(out_dict)


    # save all_questions into json
    out_json = {
        "version": 0.1,
        "responses": responses,
        "debug_logs": agent.retrieval_logs
    }

    # save the outputs
    os.makedirs(args.out_dir, exist_ok=True)
    if "QQMM" in args.model:
        name = 'remembr+'+args.base_llm+'+hf+QQMM-embed-v2+0+3584'+'__'+ Path(args.caption_file).name[:-5] + "__" +  Path(args.qa_file).name[:-5] + args.postfix
    else:
        name = args.model+'__'+ Path(args.caption_file).name[:-5] + "__" +  Path(args.qa_file).name[:-5] + args.postfix
    if "/" in name:
        name = name.replace("/", "_")
    with open(os.path.join(args.out_dir, f'{name}.json'), 'w') as f:
        # to_save = json.dumps(out_json, indent=4)
        json.dump(out_json, f, indent=4)
    
    # Clean up memory to prevent OOM when processing multiple datasets
    del memory
    del agent
    del instance_captions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='Long Horizon Robot QA',
                        description='Runs various LLMs on the QA dataset',)
    
    # data-specific args
    parser.add_argument("--model", type=str, default="remembr+gpt-4o+oc+ViT-B-32+laion2b_s34b_b79k+512")

    # choice 1: using input folder flag: doing video batch 
    parser.add_argument("--input_folder", type=str, default=None)
    parser.add_argument("--input_prompt_folder", type=str, default='prompts', help="The folder containing the prompt files for the evaluation")
    # choice 2: using single video - give two flags
    parser.add_argument("--qa_file", type=str, default=None)
    parser.add_argument("--caption_file", type=str, default=None)

    parser.add_argument("--out_dir", type=str, default="./output/")
    parser.add_argument("--postfix", type=str, default='_0')

    # llm-specific args
    parser.add_argument("--temperature", type=float, default=0.00001)
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
    parser.add_argument("--framework", type=str, default='remembr') 
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
    
    # caption baseline text embedding model
    parser.add_argument("--text_emb_model", type=str, default='original', help="Specify the caption baseline text embedding model", choices=['original', 'sota'])

    # QQMM specific args
    parser.add_argument("--coda_dir", type=str, default=None)
    parser.add_argument("--sequence_id", type=int, default=None)
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

    if args.framework == 'remembr_caption_only':
        if args.text_emb_model == 'original':
            embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
        elif args.text_emb_model == 'sota':
            embedder = VLMEmbeddings(
                args.device,
                backend="ol",
                oc_model=None,
                oc_pretrained=None,
                hf_model_id=None,
                vlm_layer=None,
                batch_size=args.batch_size,
                fp_16=args.fp_16,
                emb_dim=2048,
                vlm_text_prompts=args.vlm_text_prompts,
                vlm_image_prompts=args.vlm_image_prompts,
                online_model_nickname="seed1.6"
            )
        else:
            raise Exception("Text embedding model must be 'original' or 'sota'")
    elif args.framework == 'remembr':
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
    elif args.framework == 'embedder_only':
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

    # for k in range(1, 9):
    #     args.top_k = k
    #     args.postfix = f'_topk_{k}'
    #     print(f"Top k = {args.top_k}")
    if not args.qa_file and not args.caption_file:
        # choice 1
        # get all files in the input folder
        qa_files = sorted(glob.glob(os.path.join(args.input_folder, '*_qa.json')))
        caption_files = sorted(glob.glob(os.path.join(args.input_folder, '*_frames.json')))
        if len(qa_files) != len(caption_files):
            print("Warning: The number of QA files and caption files do not match!")
        for qa_file, caption_file in zip(qa_files, caption_files):
            assert qa_file.split('/')[-1].split('_qa.json')[0] == caption_file.split('/')[-1].split('_frames.json')[0], f"QA file {qa_file} and caption file {caption_file} do not match!"
            args.qa_file = qa_file
            args.caption_file = caption_file

            name = os.path.join(args.out_dir, (f'{args.model}__{Path(args.caption_file).name[:-5]}__{Path(args.qa_file).name[:-5]}{args.postfix}.json').replace("/", "_"))
            
            if os.path.exists(name):
                print(f"Output file {name} already exists! Skipping...")
                continue
            
            print(f"Processing {qa_file} and {caption_file}")          

            main(args, embedder)
            
            # Clean up GPU memory between datasets
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"GPU memory cleared. Free memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved, {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
        args.qa_file = None
        args.caption_file = None
    else:
        print(f"Processing {args.qa_file} and {args.caption_file}")          
        main(args, embedder)