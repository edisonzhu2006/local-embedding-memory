"""
Embedder-Only Agent: Direct retrieval without LLM reasoning.

This agent performs pure embedding-based retrieval without any LLM post-processing.
It simply returns the position/time of the most similar frame based on embedding similarity.
"""

import json
import numpy as np
import sys
import os

sys.path.append(sys.path[0] + '/..')

from remembr.agents.agent import Agent, AgentOutput
from remembr.memory.memory import Memory


class EmbedderOnlyAgent(Agent):
    """
    Agent that performs direct embedding retrieval without LLM reasoning.

    This is the simplest possible agent - it just:
    1. Embeds the query question
    2. Retrieves top-K most similar frames
    3. Returns the position/time of the best match

    No LLM reasoning, no tool calling, just pure retrieval.
    """

    def __init__(self, debug=False):
        """
        Initialize the embedder-only agent.

        Args:
            debug: Whether to print debug information
        """
        self.retrieval_logs = []
        self.memory = None
        self.debug = debug

    def set_memory(self, memory: Memory):
        """Set the memory backend for retrieval."""
        self.memory = memory

    def query(self, question: str) -> AgentOutput:
        """
        Query the memory using pure embedding retrieval.

        Args:
            question: The question to answer

        Returns:
            AgentOutput with position/time from the most similar frame
        """
        if self.memory is None:
            raise ValueError("Memory not set. Call set_memory() first.")

        if self.debug:
            print(f"[EmbedderOnly] Query: {question}")

        # Log the query for debugging
        self.retrieval_logs.append([f"Query: {question}"])

        # Perform direct retrieval - no LLM reasoning
        # Use the memory's faiss_wrapper directly to get docs with scores
        try:
            # Generate VLM embedding for the query and search
            query_embedding = self.memory.embedder.embed_query("[TXT]" + question)
            docs_with_scores = self.memory.faiss_wrapper.search_by_vlm_embedding(query_embedding, k=5)

            if self.debug:
                print(f"[EmbedderOnly] Retrieved {len(docs_with_scores)} results")

            # Log retrieved results and collect frame info
            result_info = []
            retrieved_frames = []
            for i, (doc, score) in enumerate(docs_with_scores):
                frame_path = doc.metadata.get('image_file_path', '')
                result_info.append(f"Result {i+1}: score={score:.3f} frame={frame_path}")
                frame_pos = doc.metadata.get('position', [0.0, 0.0, 0.0])
                if isinstance(frame_pos, np.ndarray):
                    frame_pos = frame_pos.tolist()
                retrieved_frames.append({
                    "rank": i + 1,
                    "score": float(score),
                    "position": frame_pos,
                    "image_file_path": doc.metadata.get('image_file_path', ''),
                })
            self.retrieval_logs.append(result_info)

            if not docs_with_scores:
                if self.debug:
                    print("[EmbedderOnly] No results found, returning default")
                return AgentOutput(
                    type="text",
                    text="No matching frames found",
                    binary=None,
                    position=None,
                    time=None,
                    orientation=None,
                    duration=None
                )

            # Take the highest scoring result (first result)
            best_doc, best_score = docs_with_scores[0]

            # Extract position and time from document metadata
            position = best_doc.metadata.get('position', [0.0, 0.0, 0.0])
            time_val = best_doc.metadata.get('time', None)

            # Handle time field which can be a list or single value
            if isinstance(time_val, list) and len(time_val) > 0:
                time_val = time_val[0]

            if self.debug:
                print(f"[EmbedderOnly] Best match: position={position}, time={time_val}, score={best_score:.3f}")

            # Format position as a list if it's a numpy array
            if position is not None and isinstance(position, np.ndarray):
                position = position.tolist()

            # Return position-based answer with retrieved frame info
            return AgentOutput(
                type="position",
                position=position if position is not None else [0.0, 0.0, 0.0],
                text=f"Found at position {position} (similarity score: {best_score:.3f})" if best_score is not None else f"Found at position {position}",
                binary=None,
                time=time_val,
                orientation=None,
                duration=None,
                retrieved_frames=retrieved_frames
            )

        except Exception as e:
            print(f"[EmbedderOnly] Error during retrieval: {e}")
            import traceback
            traceback.print_exc()

            # Return a default response on error
            return AgentOutput(
                type="text",
                text=f"Error during retrieval: {str(e)}",
                binary=None,
                position=None,
                time=None,
                orientation=None,
                duration=None
            )
