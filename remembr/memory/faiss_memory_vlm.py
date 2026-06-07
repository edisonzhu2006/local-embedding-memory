from dataclasses import dataclass, asdict
import datetime, time
import json
import os
import pickle
from time import strftime, localtime
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document
import numpy as np
import faiss

from remembr.memory.memory import Memory, VLMMemoryItem
from ..embedder.embedders import VLMEmbeddings

FIXED_SUBTRACT = 1721761000  # this is just a large value that brings us close to 1970


class FAISSVLMWrapper:
    """FAISS-based vector storage wrapper for VLM embeddings"""
    
    def __init__(self, collection_name='test', storage_path='./faiss_storage', dim=1024, drop_collection=False, cache=False, l2_metric=False):
        self.collection_name = collection_name
        self.storage_path = storage_path
        self.dim = dim
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # File paths for different indices and metadata
        self.vlm_index_path = os.path.join(storage_path, f"{collection_name}_vlm.index")
        self.position_index_path = os.path.join(storage_path, f"{collection_name}_position.index")
        self.time_index_path = os.path.join(storage_path, f"{collection_name}_time.index")
        self.metadata_path = os.path.join(storage_path, f"{collection_name}_metadata.json")
        self.id_counter_path = os.path.join(storage_path, f"{collection_name}_id_counter.json")
        
        # Initialize indices
        self.vlm_index = None
        self.position_index = None
        self.time_index = None
        self.metadata = []
        self.id_counter = 0
        
        # Load existing data or create new
        if drop_collection:
            self.drop_collection()

        self.cache = cache
        self.l2_metric = l2_metric
        self._load_or_create_indices(cache)
    
    def drop_collection(self):
        """Remove all files for this collection"""
        for path in [self.vlm_index_path, self.position_index_path, self.time_index_path, 
                    self.metadata_path, self.id_counter_path]:
            if os.path.exists(path):
                os.remove(path)
    
    def _load_or_create_indices(self, load=False):
        """Load existing indices or create new ones"""
        # Load metadata
        if load and os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []
        
        # Load ID counter
        if load and os.path.exists(self.id_counter_path):
            with open(self.id_counter_path, 'r') as f:
                self.id_counter = json.load(f)
        else:
            self.id_counter = 0
        
        # Create or load FAISS indices
        self._create_or_load_index('vlm', self.vlm_index_path, self.dim, load)
        self._create_or_load_index('position', self.position_index_path, 4, load)  # 4D: x, y, z, orientation
        self._create_or_load_index('time', self.time_index_path, 2, load)
    
    def _create_or_load_index(self, index_type, path, dim, load=False):
        """Create new FAISS index or load existing one"""
        if load and os.path.exists(path):
            # Load existing index
            if index_type == 'vlm':
                self.vlm_index = faiss.read_index(path)
            elif index_type == 'position':
                self.position_index = faiss.read_index(path)
            elif index_type == 'time':
                self.time_index = faiss.read_index(path)
        else:
            # Create new index
            if index_type == 'vlm':
                self.vlm_index = faiss.IndexFlatL2(dim) if self.l2_metric else faiss.IndexFlatIP(dim)
            elif index_type == 'position':
                self.position_index = faiss.IndexFlatL2(dim)
            elif index_type == 'time':
                self.time_index = faiss.IndexFlatL2(dim)
    
    def _save_indices(self):
        """Save all indices to disk"""
        if self.vlm_index is not None:
            faiss.write_index(self.vlm_index, self.vlm_index_path)
        if self.position_index is not None:
            faiss.write_index(self.position_index, self.position_index_path)
        if self.time_index is not None:
            faiss.write_index(self.time_index, self.time_index_path)
        
        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
        
        # Save ID counter
        with open(self.id_counter_path, 'w') as f:
            json.dump(self.id_counter, f)
    
    def insert(self, data_list):
        """Insert data into all indices"""
        for data in data_list:
            # Generate unique ID
            data['id'] = str(self.id_counter)
            self.id_counter += 1
            
            # Add to metadata
            self.metadata.append(data)
            
            # Add vectors to indices
            if 'vlm_embedding' in data:
                vlm_vector = np.array([data['vlm_embedding']], dtype=np.float32)
                self.vlm_index.add(vlm_vector)
            
            if 'position' in data:
                # Pad position to 4 elements (x, y, z, orientation) for FAISS index
                # Some datasets have 3 elements, pad with 0.0 for orientation
                pos = list(data['position'])
                while len(pos) < 4:
                    pos.append(0.0)
                position_vector = np.array([pos[:4]], dtype=np.float32)
                self.position_index.add(position_vector)
            
            if 'time' in data:
                time_vector = np.array([data['time']], dtype=np.float32)
                self.time_index.add(time_vector)
        
        # Save to disk
        if self.cache:
            self._save_indices()
    
    def search_by_vlm_embedding(self, query_vector, k):
        """Search by VLM embedding vector"""
        if self.vlm_index.ntotal == 0:
            return []

        query_vector = np.array([query_vector], dtype=np.float32)
        distances, indices = self.vlm_index.search(query_vector, min(k, self.vlm_index.ntotal))
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                metadata = self.metadata[idx].copy()
                # Create Document object similar to Milvus results
                doc = Document(
                    page_content=metadata.get('caption', ''),
                    metadata=metadata
                )
                results.append((doc, float(dist)))
        
        return results
    
    def search_by_position(self, position_vector, k):
        """Search by position vector"""
        if self.position_index.ntotal == 0:
            return []
        
        # Pad to 4 elements if needed (x, y, z, orientation)
        pos = list(position_vector)
        while len(pos) < 4:
            pos.append(0.0)
        query_vector = np.array([pos[:4]], dtype=np.float32)
        distances, indices = self.position_index.search(query_vector, min(k, self.position_index.ntotal))
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                metadata = self.metadata[idx].copy()
                doc = Document(
                    page_content=metadata.get('caption', ''),
                    metadata=metadata
                )
                results.append((doc, float(dist)))
        
        return results
    
    def search_by_time(self, time_vector, k):
        """Search by time vector"""
        if self.time_index.ntotal == 0:
            return []
        
        query_vector = np.array([time_vector], dtype=np.float32)
        distances, indices = self.time_index.search(query_vector, min(k, self.time_index.ntotal))
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                metadata = self.metadata[idx].copy()
                doc = Document(
                    page_content=metadata.get('caption', ''),
                    metadata=metadata
                )
                results.append((doc, float(dist)))
        
        return results


class FAISSVLMMemory(Memory):
    """FAISS-based VLM memory implementation that replaces MilvusVLMMemory"""
    
    def __init__(self, db_collection_name: str, embedder: VLMEmbeddings, storage_path='./faiss_storage', 
                 time_offset=FIXED_SUBTRACT, dim=1024, retriever_k=5, respond_with_score=False):
        self.dim = dim
        self.retriever_k = retriever_k
        self.db_collection_name = db_collection_name
        self.storage_path = storage_path
        self.time_offset = time_offset
        self.embedder = embedder
        self.working_memory = []
        self.respond_with_score = respond_with_score
        
        print("Using FAISS storage for VLM memory")
        self.reset(drop_collection=False)

    def insert(self, item: VLMMemoryItem, vlm_embedding=None):
        """Insert a VLM memory item"""
        memory_dict = asdict(item)
        memory_dict['id'] = str(time.time())

        if vlm_embedding is None:
            vlm_embedding = self.embedder.embed_query("[IMG]" + memory_dict['image_file_path'])

        memory_dict['time'] = [(memory_dict['time'] - self.time_offset), 0.0]
        memory_dict['vlm_embedding'] = vlm_embedding

        self.faiss_wrapper.insert([memory_dict])

    def get_working_memory(self) -> list[VLMMemoryItem]:
        return self.working_memory

    def reset(self, drop_collection=True):
        """Reset the memory storage"""
        if drop_collection:
            print("Resetting FAISS VLM memory. Dropping current collection")

        self.faiss_wrapper = FAISSVLMWrapper(
            collection_name=self.db_collection_name,
            storage_path=self.storage_path,
            dim=self.dim,
            drop_collection=drop_collection
        )

    def search_by_position(self, query: tuple) -> str:
        """Search memories by position"""
        docs_with_scores = self.faiss_wrapper.search_by_position(np.array(query).astype(float), self.retriever_k)
        docs = [doc for doc, _ in docs_with_scores]
        
        self.working_memory += docs
        docs = self.memory_to_string_vlm(docs)
        return docs

    def search_by_time(self, hms_time: str) -> str:
        """Search memories by time"""
        # Convert time string to searchable format
        t = localtime(self.time_offset)
        mdy_date = strftime('%m/%d/%Y', t)
        template = "%m/%d/%Y %H:%M:%S"

        # Check if time is already in correct format
        try:
            res = bool(datetime.datetime.strptime(hms_time, template))
        except ValueError:
            res = False

        hms_time = hms_time.strip()
        if not res:
            hms_time = mdy_date + ' ' + hms_time

        query = time.mktime(datetime.datetime.strptime(hms_time, template).timetuple()) - self.time_offset
        query_vector = np.array([query, 0])

        docs_with_scores = self.faiss_wrapper.search_by_time(query_vector, self.retriever_k)
        docs = [doc for doc, _ in docs_with_scores]
        
        self.working_memory += docs
        docs = self.memory_to_string_vlm(docs)
        return docs

    def search_by_text(self, query: str) -> str:
        """Search memories by text using VLM embedding"""
        # Generate VLM embedding for the query
        query_embedding = self.embedder.embed_query("[TXT]" + query)
        
        docs_with_scores = self.faiss_wrapper.search_by_vlm_embedding(query_embedding, self.retriever_k)
        docs = [doc for doc, _ in docs_with_scores]
        
        self.working_memory += docs
        docs = self.memory_to_string_vlm(docs, text_score_list=[score for _, score in docs_with_scores] if self.respond_with_score else None)
        return docs

    def memory_to_string_vlm(self, memory_list: list[Document], ref_time: float = None, text_score_list=None):
        """Convert VLM memory documents to string format"""
        if ref_time is None:
            ref_time = self.time_offset

        out_string = "Image Retrieval Results:\n"
        for i, doc in enumerate(memory_list):
            score = text_score_list[i] if text_score_list is not None else None
            if len(doc.metadata['time']) == 2:
                t = doc.metadata['time'][0]
            else:
                t = doc.metadata['time']
            
            if ref_time:
                t += ref_time
            t = localtime(t)
            t = strftime('%Y-%m-%d %H:%M:%S', t)

            s = f"({i+1}) At time={t}, the robot was at an average position of {np.array(doc.metadata['position']).round(3).tolist()}. The robot saw the following: "
            if score is not None:
                s += f"(Cosine similarity between the query and the frame: {score:.3f}) "
            s += f"[IMG]{doc.metadata['image_file_path']}[/IMG]\n\n"
            out_string += s
        return out_string
