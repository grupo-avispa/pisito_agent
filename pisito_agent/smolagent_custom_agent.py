"""Custom Agent implementation using SmolAgents framework and FastMCP for tool calling.
It works as ToolCallingAgent, integrating with a FastMCP server to utilize defined tools.
Additionally, it expects a quantized model for efficient inference and initializes 
with a system prompt if filepath is provided.
"""

from rich.text import Text
from rich.live import Live
from rich.markdown import Markdown
from collections.abc import Generator
# SmolAgents imports for agent framework
from smolagents import ToolCallingAgent, ChatMessage, Tool, MCPClient
from smolagents.monitoring import LogLevel
from smolagents.models import parse_json_if_needed, ChatMessageStreamDelta, agglomerate_stream_deltas
from smolagents.agents import ToolOutput, ActionOutput
from smolagents.utils import (
    AgentExecutionError,
    AgentGenerationError,
    AgentParsingError,
    AgentToolExecutionError,
)
from smolagents.memory import ActionStep, ToolCall
# MCP client import for tool calling
from fastmcp import Client
import torch
from smolagent_custom_model import QuantModel



class CustomAgent(ToolCallingAgent):
    """Custom Agent that integrates with FastMCP server for tool calling and uses a quantized model.
    This agent uses JSON-like tool calls, using method 'model.parse_tool_calls' to parse tool calls from model output.
    
    Args:
        tools (list[Tool]): List of Tool instances available to the agent.
        model (QuantModel): The quantized language model used by the agent.
        sys_prompt_file (str | None): Optional file path to a system prompt template.
        planning_interval (int | None): Optional interval for planning steps.
        stream_outputs (bool): Whether to stream outputs during generation.
        max_tool_threads (int | None): Maximum number of threads for tool execution.
        **kwargs: Additional keyword arguments for the base ToolCallingAgent.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: QuantModel,
        sys_prompt_file: str | None = None,
        planning_interval: int | None = None,
        stream_outputs: bool = False,
        max_tool_threads: int | None = None,
        **kwargs,
    ):
        self.sys_prompt_file = sys_prompt_file
        super().__init__(
            tools=tools,
            model=model,
            planning_interval=planning_interval,
            stream_outputs=stream_outputs,
            max_tool_threads=max_tool_threads,
            **kwargs,
        )

    def initialize_system_prompt(self) -> str:
        """
        Initializes the system prompt for the agent.
        If a system prompt file is provided, it reads the content from the file.
        Otherwise, it uses a default prompt.
        """
        system_prompt = "You are a helpful assistant."
        if self.sys_prompt_file:
            with open(self.sys_prompt_file, "r") as f:
                system_prompt = f.read()
        return system_prompt
    
    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()

        # Add new step in logs
        memory_step.model_input_messages = input_messages

        try:
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                output_stream = self.model.generate_stream(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],
                    tools_to_call_from=self.tools_and_managed_agents,
                )

                chat_message_stream_deltas: list[ChatMessageStreamDelta] = []
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        chat_message_stream_deltas.append(event)
                        live.update(
                            Markdown(agglomerate_stream_deltas(chat_message_stream_deltas).render_as_markdown())
                        )
                        yield event
                chat_message = agglomerate_stream_deltas(chat_message_stream_deltas)
            else:
                chat_message: ChatMessage = self.model.generate(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],
                    tools_to_call_from=self.tools_and_managed_agents,
                )
                self.logger.log_markdown(
                    content=str(chat_message.content or chat_message.raw or ""),
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )
                self.logger.log(
                    Text(f"LLM raw output:\n{chat_message.content or chat_message.raw or ''}",
                        style=f"green"),
                    level=LogLevel.INFO,
                )
                # Record model output
                memory_step.model_output_message = chat_message
                memory_step.model_output = chat_message.content
                memory_step.token_usage = chat_message.token_usage

        except Exception as e:
            raise AgentGenerationError(f"Error while generating output:\n{e}", self.logger) from e

        if chat_message.tool_calls is None or len(chat_message.tool_calls) == 0:
            try:
                chat_message = self.model.parse_tool_calls(chat_message)
            except Exception as e:
                raise AgentParsingError(f"Error while parsing tool call from model output: {e}", self.logger)
        else:
            for tool_call in chat_message.tool_calls:
                tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        final_answer, got_final_answer = None, False
        for output in self.process_tool_calls(chat_message, memory_step):
            yield output
            if isinstance(output, ToolOutput):
                if output.is_final_answer:
                    if len(chat_message.tool_calls) > 1:
                        raise AgentExecutionError(
                            "If you want to return an answer, please do not perform any other tool calls than the final answer tool call!",
                            self.logger,
                        )
                    if got_final_answer:
                        raise AgentToolExecutionError(
                            "You returned multiple final answers. Please return only one single final answer!",
                            self.logger,
                        )
                    final_answer = output.output
                    got_final_answer = True

                    # Manage state variables
                    if isinstance(final_answer, str) and final_answer in self.state.keys():
                        final_answer = self.state[final_answer]
        yield ActionOutput(
            output=final_answer,
            is_final_answer=got_final_answer,
        )

if __name__ == "__main__":

    server_parameters = {"url": "http://localhost:3002/mcp"}

    try:
        mcp_client = MCPClient(server_parameters)
        tools = mcp_client.get_tools()

        # Use the tools with your agent
        model = QuantModel(
            model_id="Qwen/Qwen3-0.6B",
            tools=tools,  # Add your Tool instances here
            tool_call_pattern="<tool_call>(.*?)</tool_call>",
            do_sample=False,
            max_new_tokens=256,
            enable_thinking=False
        )
        
        agent = CustomAgent(
            tools=tools,  # Add your Tool instances here
            model=model,
            sys_prompt_file="../templates/system_prompt.jinja"
        )
        result = agent.run("Turn off the bedroom ", max_steps=8)

        # Process the result as needed
        print(f"Agent response:\n{result}")
    finally:
        if 'mcp_client' in locals():
            mcp_client.disconnect()