"""
Memory factory for creating memory instances with different backends.
Supports both Milvus and FAISS backends.
"""

from typing import Optional, Union
from .memory import Memory, MemoryItem, VLMMemoryItem
from .faiss_memory import FAISSMemory
from .faiss_memory_vlm import FAISSVLMMemory
from ..embedder.embedders import VLMEmbeddings


class MemoryFactory:
    """Factory class for creating memory instances with different backends"""
    
    @staticmethod
    def create_memory(
        backend: str = "faiss",  # Default to FAISS
        db_collection_name: str = "test",
        db_ip: str = "127.0.0.1",
        db_port: int = 19530,
        storage_path: str = "./faiss_storage",
        time_offset: float = 1721761000,
        use_vlm_embedding: bool = False,
        vlm_model_path: Optional[str] = None,
        dim: int = 1024,
        retriever_k: int = 5,
        embedder: Optional[VLMEmbeddings] = None,
        respond_with_score: bool = False,
        **kwargs
    ) -> Memory:
        """
        Create a memory instance with the specified backend.
        
        Args:
            backend: "milvus" or "faiss"
            db_collection_name: Name of the collection/database
            db_ip: IP address for Milvus (ignored for FAISS)
            db_port: Port for Milvus (ignored for FAISS)
            storage_path: Path for FAISS storage (ignored for Milvus)
            time_offset: Time offset for timestamp calculations
            use_vlm_embedding: Whether to use VLM embeddings
            vlm_model_path: Path to VLM model (for Milvus)
            dim: Embedding dimension
            retriever_k: Number of results to retrieve
            embedder: VLM embedder instance (for VLM backends)
            **kwargs: Additional arguments passed to memory constructors
        
        Returns:
            Memory instance
        """
        if backend.lower() == "milvus":
            from .milvus_memory import MilvusMemory
            from .milvus_memory_vlm import MilvusVLMMemory
            if use_vlm_embedding and embedder is not None:
                return MilvusVLMMemory(
                    db_collection_name=db_collection_name,
                    embedder=embedder,
                    db_ip=db_ip,
                    db_port=db_port,
                    time_offset=time_offset,
                    dim=dim,
                    retriever_k=retriever_k,
                    **kwargs
                )
            else:
                return MilvusMemory(
                    db_collection_name=db_collection_name,
                    db_ip=db_ip,
                    db_port=db_port,
                    time_offset=time_offset,
                    use_vlm_embedding=use_vlm_embedding,
                    vlm_model_path=vlm_model_path,
                    **kwargs
                )
        
        elif backend.lower() == "faiss":
            assert embedder is not None, "Embedder must be provided for FAISS backend"
            if use_vlm_embedding:
                return FAISSVLMMemory(
                    db_collection_name=db_collection_name,
                    embedder=embedder,
                    storage_path=storage_path,
                    time_offset=time_offset,
                    dim=dim,
                    retriever_k=retriever_k,
                    respond_with_score=respond_with_score,
                    **kwargs
                )
            else:
                return FAISSMemory(
                    db_collection_name=db_collection_name,
                    embedder=embedder,
                    storage_path=storage_path,
                    time_offset=time_offset,
                    use_vlm_embedding=use_vlm_embedding,
                    vlm_model_path=vlm_model_path,
                    dim=dim,
                    retriever_k=retriever_k,
                    **kwargs
                )
        
        else:
            raise ValueError(f"Unsupported backend: {backend}. Supported backends are 'milvus' and 'faiss'")
    
    @staticmethod
    def get_available_backends() -> list[str]:
        """Get list of available backends"""
        return ["milvus", "faiss"]
    
    @staticmethod
    def get_backend_info() -> dict:
        """Get information about available backends"""
        return {
            "milvus": {
                "description": "Milvus vector database backend",
                "requires_docker": True,
                "requires_network": True,
                "persistent": True,
                "scalable": True,
                "features": ["Vector search", "Metadata filtering", "Distributed storage"]
            },
            "faiss": {
                "description": "FAISS local file-based backend",
                "requires_docker": False,
                "requires_network": False,
                "persistent": True,
                "scalable": False,
                "features": ["Vector search", "Local storage", "No dependencies"]
            }
        }


def create_memory_from_config(config: dict) -> Memory:
    """
    Create memory instance from configuration dictionary.
    
    Example config:
    {
        "backend": "faiss",
        "db_collection_name": "my_memory",
        "storage_path": "./my_storage",
        "use_vlm_embedding": True,
        "embedder": embedder_instance
    }
    """
    return MemoryFactory.create_memory(**config)
