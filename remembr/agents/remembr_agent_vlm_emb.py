import sys, re
import base64
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage



import sys, os
sys.path.append(sys.path[0] + '/..')


from remembr.utils.util import file_to_string
from remembr.tools.tools import *
from remembr.agents.remembr_agent import ReMEmbRAgent


class ReMEmbRAgentVLM(ReMEmbRAgent):

    def __init__(self, llm_type='gpt-4o', num_ctx=8192, temperature=0, debug=False, prompt_dir="unknown"):
        super().__init__(llm_type, num_ctx, temperature, debug, prompt_dir)
        self.tool_descriptions = {
            "retrieve_from_text":
                "Search and return multimodal information from your video memory in the form of a list of closest images with their similarity scores",
            "retrieve_from_position":
                "Search and return multimodal information from your video memory by using a position array such as (x,y,z)",
            "retrieve_from_time":
                "Search and return multimodal information from your video memory by using an H:M:S time."
        }


    ### Modified Nodes
    def agent_msg_decorator(self, messages):
        # Convert all ToolMessages into AI Messages since Ollama cann't handle ToolMessage
        i = 0
        pattern = r"\{'type': 'image_url', 'image_url': \{'url': 'data:image/jpeg;base64[^']*'\}\}"
        replacement = "{'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64'}}"
        message_dump = [re.sub(pattern, replacement, str(message)) for message in messages]
        self.retrieval_logs.append(message_dump)
        if self.debug:
            print(f"\n========================Start Processing IMG - {len(messages)}=======================")
            print("message dump:", message_dump)
        
        image_messages = []
        tool_call_num = 0
        while i < len(messages):
            if type(messages[i]) == ToolMessage and "[IMG]" in messages[i].content:
                # match the image file path between [IMG] and [/IMG]
                old_content = messages[i].content
                if not isinstance(old_content, str):
                    print(f"Coming into Ollama Message Processing, type(old_content): {type(old_content)}, content: {old_content}")
                    if ('gpt' not in self.llm_type) and ('nim' not in self.llm_type):
                        # format for ollama
                        print(f"Turn ToolMessage to AIMessage...")
                        messages[i] = AIMessage(id=messages[i].id, content=old_content) # ignore tool_call_id
                    continue
                ls_content = old_content.split("\n\n")
                updated_contents = []
                tool_call_num += 1
                for content in ls_content:
                    if "[IMG]" in content:
                        search_result = re.search(r"\[IMG\](.*?)\[/IMG\]", content)
                        image_file_path = search_result.group(1)
                        updated_content = f"Tool call {tool_call_num}: " + content.replace(search_result.group(0), "")
                        if '.pkl' in image_file_path:
                            from remembr.embedder.embedders import file_to_data_url
                            data_uri = file_to_data_url(image_file_path)
                        else:
                            with open(image_file_path, "rb") as f:
                                img_base64 = base64.b64encode(f.read()).decode("utf-8")
                            data_uri = f"data:image/jpeg;base64,{img_base64}"
                        updated_content = [
                            {"type": "text", "text": updated_content},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ]
                        updated_contents.extend(updated_content)
                messages[i].content = f"Tool call {tool_call_num} results will be shown by the human user."
                image_messages.append(HumanMessage(id=messages[i].id, content=updated_contents))
            i += 1
        if image_messages:
            messages.append(AIMessage(content="I see. Could the user provide more details?"))
            messages.extend(image_messages)

        if self.debug:
            print(f"========================End Processing IMG - {len(messages)}=======================")
        return messages

    def gen_msg_decorator(self, messages):
        return messages[:]