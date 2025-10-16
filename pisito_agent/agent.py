"""
ROS2 node implementing a conversational agent using a Huggingface LLM.
The agent subscribes to user queries and publishes generated responses.
The agent's behavior is configurable via ROS2 parameters.
"""

# General imports 
import re
import torch
import json
import asyncio
from typing import Tuple

# Transformer import for LLM interactions
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# MCP client import for tool calling
from fastmcp import Client

# ROS2 imports for subscriber and publisher implementation
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String

# ============= ROS2 NODE =============

class Agent(Node):
    # ============= INITIALIZATION =============

    def __init__(self):
        """
        Initialize the Agent node.

        Sets up the ROS2 node, retrieves parameters, initializes the
        LLM object, and creates the subscriber and publisher.

        Returns:
            None
        """
        super().__init__('hf_agent')
        # Retrieve ROS2 parameters (topic names, MCP servers, system prompt)
        self.get_params()

        # Create event loop for asynchronous operations
        self.loop = asyncio.new_event_loop()

        # Initialize the LLM object
        self.loop.run_until_complete(self.initialize_llm())

        # Create the subscriber to listen for user queries
        self.group = ReentrantCallbackGroup()
        self.query_sub = self.create_subscription(
            String,
            self.query_topic,
            self.agent_callback,
            10,
            callback_group=self.group)
        # Create the publisher to send agent responses
        self.response_pub = self.create_publisher(
            String,
            self.response_topic,
            10,
            callback_group=self.group)

        self.get_logger().info('Agent node initialized.')
    
    # ============= METHODS =============

    def agent_callback(self, msg: String) -> None:
        """
        Callback function for processing user queries.

        Receives a user query message, processes it using the LLM,
        and publishes the generated response.

        Parameters:
            msg (String): The incoming user query message.
        Returns:
            None
        """
        user_query = msg.data
        self.get_logger().info(f'Received user query: {user_query}')

        # Prepare the messages for the LLM
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": user_query}
        ]
        # Prepare the chat template with tools, system prompt and user query
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            tools=self.tools, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )

        # Move inputs to the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        # Generate the response using the LLM
        with torch.no_grad():
            llm_output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                top_k=self.top_k,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(llm_output[0][len(inputs["input_ids"][0]):], 
                                         skip_special_tokens=False)
        # Post-process the response to extract tool call between <tool> tags if any
        response_msg = String()
        response_msg.data = self.loop.run_until_complete(self.process_response(response))
        # Publish the response
        self.response_pub.publish(response_msg)

    async def process_response(self, llm_response: str) -> str:
        """
        Process the LLM response to handle tool calls.

        If the LLM response contains a tool call, it extracts the tool name and parameters,
        invokes the tool via the MCP client, and publishes the tool's output.
        If no tool call is present, it publishes the raw response.

        Parameters:
            llm_response (str): The raw response from the LLM.
        Returns:
            str: The final response including LLM and tool outputs.
        """
        # Default response if parsing fails
        tool_response = ("Error parsing tool call, the resulting JSON must have 'name' " + 
        "and 'parameters' fields. Also must be contained inside <tool_call> </tool_call> tags.")
        parsed_response = llm_response.strip()
        # Post-process the response to extract tool call between <tool> tags if any
        tool_call_match = re.search(self.tool_call_pattern, llm_response, re.DOTALL)
        if tool_call_match:
            parsed_response = tool_call_match.group(1).strip()
            # Create JSON object from the parsed response
            action = json.loads(parsed_response)
            # Extract tool name and parameters
            tool_name = action['name']
            tool_params = action['arguments']
            async with self.client:
                try:
                    tool_response = await self.client.call_tool(tool_name, 
                                                                tool_params)
                    tool_response = tool_response.content[0].text
                except Exception as e:
                    tool_response = f"Error executing tool '{tool_name}': {str(e)}"
        final_response = f"- LLM response: {parsed_response}\n - Tool response: {tool_response}"
        self.get_logger().info(f'Final response:\n {final_response}')
        return final_response

    async def initialize_llm(self)-> None:
        """
        Initialize the LLM model and tokenizer.

        Loads the specified LLM model and tokenizer using the transformers library.
        Configures the model for efficient inference.

        Parameters:
            None
        Returns:
            None
        """
        
        # Load the tokenizer and model from the specified LLM model name
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model, use_fast=True)
        # Load model with 8-bit quantization if specified
        if self.use_int8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf4",
                bnb_8bit_compute_type=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model,
                quantization_config=quantization_config,
                device_map="auto"
            )
            self.get_logger().info(f'LLM model {self.llm_model} loaded in 8 bit quantization.')
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model,
                device_map="auto",
                torch_dtype=torch.float16
            )
            self.get_logger().info(f'LLM model {self.llm_model} loaded.')
        
        # Set system prompt
        with open(self.system_prompt_file, 'r') as f:
            self.sys_prompt = f.read()

        # MCP servers configuration
        with open(self.mcp_servers, 'r') as f:
            mcp_servers_config = json.load(f)
        print(mcp_servers_config)

        # Initialize MCP client for retrieving RAG and other tools
        self.client = Client(mcp_servers_config)

        # Retrieve available tools from MCP server
        self.tools = []
        async with self.client:
            try:
                tool_list = await self.client.list_tools()
                self.get_logger().info(f'Successfully loaded {len(tool_list)} tools from MCP server')
                # Convert list to JSON schema
                for tool in tool_list:
                    schema = {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema,
                        "outputSchema": tool.outputSchema
                    }
                    self.tools.append(schema)
            except Exception as e:
                self.get_logger().info(f'WARNING: Failed to connect to MCP server: {str(e)}')
                self.get_logger().info('Continuing with no tools available')


    def get_params(self) -> None:
        """
        Retrieve and configure ROS2 parameters.

        Declares and retrieves parameters from the ROS2 parameter server.
        Logs each parameter value for verification.

        Parameters:
            None

        Returns:
            None
        """
        # Declare and retrieve topic parameters
        self.declare_parameter('query_topic', 'user_query')
        self.query_topic = self.get_parameter(
            'query_topic').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter query_topic is set to: [{self.query_topic}]')
        
        self.declare_parameter('response_topic', 'llm_response')
        self.response_topic = self.get_parameter(
            'response_topic').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter response_topic is set to: [{self.response_topic}]')

        # Declare and retrieve MCP servers parameter
        self.declare_parameter('mcp_servers', 'mcp.json')
        self.mcp_servers = self.get_parameter(
            'mcp_servers').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter mcp_servers is set to: [{self.mcp_servers}]')

        # Declare and retrieve system prompt template path parameter
        self.declare_parameter('system_prompt_file', 'system_prompt.jinja')
        self.system_prompt_file = self.get_parameter(
            'system_prompt_file').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter system_prompt_file is set to: [{self.system_prompt_file}]')
        
        # Declare and retrieve LLM model name parameter
        self.declare_parameter('llm_model', 'Qwen/Qwen2.5-0.5B-Instruct')
        self.llm_model = self.get_parameter(
            'llm_model').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter llm_model is set to: [{self.llm_model}]')
        
        # Declare tool call regex pattern to extract tool calls from LLM response
        self.declare_parameter('tool_call_pattern', r'<tool>(.*?)/<tool>')
        self.tool_call_pattern = self.get_parameter(
            'tool_call_pattern').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter tool_call_pattern is set to: [{self.tool_call_pattern}]')
        
        # Declare and retrieve bolean parameter for int 8 quantization
        self.declare_parameter('use_int8', True)
        self.use_int8 = self.get_parameter(
            'use_int8').get_parameter_value().bool_value
        self.get_logger().info(
            f'The parameter use_int8 is set to: [{self.use_int8}]')
        
        # Declare and retrieve LLM generation parameters
        self.declare_parameter('max_new_tokens', 512)
        self.max_new_tokens = self.get_parameter(
            'max_new_tokens').get_parameter_value().integer_value
        self.get_logger().info(
            f'The parameter max_new_tokens is set to: [{self.max_new_tokens}]')
        
        self.declare_parameter('top_k', 10)
        self.top_k = self.get_parameter(
            'top_k').get_parameter_value().integer_value
        self.get_logger().info(
            f'The parameter top_k is set to: [{self.top_k}]')
        
        self.declare_parameter('temperature', 0.1)
        self.temperature = self.get_parameter(
            'temperature').get_parameter_value().double_value
        self.get_logger().info(
            f'The parameter temperature is set to: [{self.temperature}]')
        
        self.declare_parameter('repetition_penalty', 1.1)
        self.repetition_penalty = self.get_parameter(
            'repetition_penalty').get_parameter_value().double_value
        self.get_logger().info(
            f'The parameter repetition_penalty is set to: [{self.repetition_penalty}]')


def main(args=None) -> None:
    """
    Run the ROS2 agent.

    Initialize the ROS2 context, create the agent node, and spin
    until shutdown is requested.

    Parameters:
        args: Command-line arguments (optional).

    Returns:
        None
    """
    rclpy.init(args=args)
    
    try:
        # Create the agent node
        agent = Agent()

        # Use a MultiThreadedExecutor to allow concurrent callback execution
        executor = MultiThreadedExecutor()
        executor.add_node(agent)

        # Spin the node to process callbacks
        executor.spin()
    except (KeyboardInterrupt, Exception, ExternalShutdownException) as e:
        print(f'Shutting down agent node due to: {e}')


if __name__ == '__main__':
    main()
