"""Custom Quantized Model implementation using SmolAgents framework.
It extends the TransformersModel to support quantized models using
BitsAndBytesConfig from Hugging Face Transformers. It also includes
functionality to parse tool calls from model outputs based on a specified pattern.
"""

import re
import json
import uuid
from typing import Any
from rich.text import Text
# Transformer import for LLM interactions
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
# SmolAgents imports for agent framework
from smolagents import TransformersModel, ChatMessage, Tool
from smolagents.models import MessageRole, TokenUsage, ChatMessageToolCallFunction, ChatMessageToolCall
from smolagents.monitoring import LogLevel
import torch

def remove_content_after_stop_sequences(content: str, stop_sequences: list[str]) -> str:
    """Remove content after any stop sequence is encountered."""
    for stop_seq in stop_sequences:
        split = content.split(stop_seq)
        content = split[0]
    return content

class QuantModel(TransformersModel):
    """Custom Quantized Model that extends TransformersModel to support quantized models.
    It uses BitsAndBytesConfig for loading models in 8-bit or 4-bit precision.
    Additionally, it includes functionality to parse tool calls from model outputs
    based on a specified pattern.
    Args:
        model_id (str | None): The identifier of the pre-trained model.
        tools (list[Tool] | None): List of Tool instances available to the model.
        tool_call_pattern (str | None): Regex pattern to identify tool calls in model output.
        tool_name_key (str): Key name for tool name in parsed JSON.
        tool_params_key (str): Key name for tool parameters in parsed JSON.
        sys_prompt_file (str | None): Optional file path to a system prompt template.
        device_map (str | None): Device map for model loading.
        torch_dtype (str | None): Torch data type for model loading.
        trust_remote_code (bool): Whether to trust remote code when loading the model.
        model_kwargs (dict[str, Any] | None): Additional keyword arguments for model loading.
        max_new_tokens (int): Maximum number of new tokens to generate.
        max_tokens (int | None): Maximum total tokens (prompt + generation).
        load_in_8bit (bool): Whether to load the model in 8-bit precision.
        load_in_4bit (bool): Whether to load the model in 4-bit precision.
        temperature (float): Sampling temperature for generation.
        top_k (float): Top-k sampling parameter.
        top_p (float): Top-p sampling parameter.
        do_sample (bool): Whether to use sampling for generation.
        repetition_penalty (float): Repetition penalty for generation.
    """
    def __init__(self,
        model_id: str | None = None,
        tools: list[Tool] | None = None,
        tool_call_pattern: str | None = None,
        tool_name_key: str = "name",
        tool_params_key: str = "arguments",
        sys_prompt_file: str | None = None,
        device_map: str | None = None,
        torch_dtype: str | None = None,
        trust_remote_code: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        max_new_tokens: int = 256,
        max_tokens: int | None = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        temperature: float = 0.1,
        top_k: float = 50,
        top_p: float = 0.5,
        do_sample:  bool = False,
        repetition_penalty: float = 1.0,
        enable_thinking: bool = False,
        **kwargs):
    
        if model_id is None:
            raise ValueError("model_id must be provided")
        
        self.max_new_tokens = max_tokens if max_tokens is not None else max_new_tokens

        if device_map is None:
            device_map = "cuda" if torch.cuda.is_available() else "cpu"

        self._is_vlm = False

        self.model_kwargs = model_kwargs or {}

        super().__init__(model_id,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                **self.model_kwargs,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                quantization_config=bnb_config,
                trust_remote_code=trust_remote_code,
                **self.model_kwargs,
            )
        elif load_in_4bit:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                quantization_config=bnb_config,
                trust_remote_code=trust_remote_code,
                **self.model_kwargs,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                **self.model_kwargs,
            )
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore
        
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.tools = tools or []
        self.tool_call_pattern = tool_call_pattern
        self.tool_name_key = tool_name_key
        self.tool_params_key = tool_params_key
        self.sys_prompt_file = sys_prompt_file
        self.do_sample = do_sample
        self.enable_thinking = enable_thinking
    
    def _prepare_completion_args(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Prepare parameters required for model invocation.
        """
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            tool_choice=None,
            **kwargs,
        )

        messages = completion_kwargs.pop("messages")
        stop_sequences = completion_kwargs.pop("stop", None)
        completion_kwargs.pop("tools", None)


        # Format tool schema for inclusion in system prompt
        tools_schemas = []
        for tool in tools_to_call_from:
            tools_schemas.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputs": tool.inputs
                }
            )
        
        prompt_tensor = (self.processor if hasattr(self, "processor") else self.tokenizer).apply_chat_template(
            messages,
            tools=tools_schemas,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=self.enable_thinking
        )
        
        print(f"System prompt:\n{self.tokenizer.apply_chat_template(
            messages,
            tools=tools_schemas,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.enable_thinking
        )}")

        # Compute attention mask 
        attention_mask = (prompt_tensor != (self.processor if hasattr(self, "processor") else self.tokenizer).pad_token_id).long()
        attention_mask = attention_mask.to(self.model.device)
        prompt_tensor = prompt_tensor.to(self.model.device)  # type: ignore
        if hasattr(prompt_tensor, "input_ids"):
            prompt_tensor = prompt_tensor["input_ids"]

        model_tokenizer = self.processor.tokenizer if hasattr(self, "processor") else self.tokenizer
        stopping_criteria = (
            self.make_stopping_criteria(stop_sequences, tokenizer=model_tokenizer) if stop_sequences else None
        )

        completion_kwargs.update({
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample
        })

        return dict(
            inputs=prompt_tensor,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            **completion_kwargs,
        )

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """Generate a response from the model based on input messages.
        Parameters:
            messages (list[ChatMessage | dict]): The input messages for the model.
            stop_sequences (list[str] | None): Optional list of stop sequences to end generation.
            tools_to_call_from (list[Tool] | None): Optional list of tools available for calling.
            **kwargs: Additional keyword arguments for generation.
        Returns:
            ChatMessage: The generated response message from the model.
        """
        generation_kwargs = self._prepare_completion_args(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        count_prompt_tokens = generation_kwargs["inputs"].shape[1]  # type: ignore

        out = self.model.generate(
            **generation_kwargs,
        )
        generated_tokens = out[0, count_prompt_tokens:]
        if hasattr(self, "processor"):
            output_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
        else:
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if stop_sequences is not None:
            output_text = remove_content_after_stop_sequences(output_text, stop_sequences)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=output_text,
            raw={
                "out": output_text,
                "completion_kwargs": {key: value for key, value in generation_kwargs.items() if key != "inputs"},
            },
            token_usage=TokenUsage(
                input_tokens=count_prompt_tokens,
                output_tokens=len(generated_tokens),
            ),
        )
    
    def parse_tool_calls(
        self,
        message: ChatMessage,
    ) -> ChatMessage:
        """Parse tool calls from the model output message based on the specified tool_call_pattern.
        Parameters:
            message (ChatMessage): The model output message to parse.
        Returns:
            ChatMessage: The message with parsed tool calls.
        """
        if self.tool_call_pattern is None:
            raise ValueError("tool_call_pattern must be provided to parse tool calls." + 
                             "For example: r'<tool_call>(.*?)</tool_call>' or 'Calling tools:'")
        # Try to parse tool calls from the message content based on a call pattern
        content = message.content.strip()
        tool_call_match = re.search(self.tool_call_pattern, content, re.DOTALL)
        message.tool_calls = [
                    ChatMessageToolCall(
                        id=str(uuid.uuid4()),
                        type="function",
                        function=ChatMessageToolCallFunction(name="final_answer", arguments={"answer": content}),
                    )   
                ]
        if tool_call_match:
            parsed_response = tool_call_match.group(1).strip()
            # Create JSON object from the parsed response
            action = json.loads(parsed_response)
            # Extract tool name and parameters
            try:
                tool_name = action[self.tool_name_key]
                tool_arguments = action[self.tool_params_key]
                message.tool_calls = [
                    ChatMessageToolCall(
                        id=str(uuid.uuid4()),
                        type="function",
                        function=ChatMessageToolCallFunction(name=tool_name, arguments=tool_arguments),
                    )   
                ]
            except KeyError as e:
                raise ValueError(f"Missing key in tool call JSON: {e}")
        else:
            print(f"No tool call found in the message content. Using full content as final answer.")
        return message
        