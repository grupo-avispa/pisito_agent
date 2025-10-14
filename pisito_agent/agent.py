"""
ROS2 node implementing a conversational agent using a Huggingface LLM.
The agent subscribes to user queries and publishes generated responses.
The agent's behavior is configurable via ROS2 parameters.
"""

# General imports 
import re
import torch
import json

# Transformer import for LLM interactions
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

        # Initialize the LLM object
        self.initialize_llm()

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


    def initialize_llm(self):
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

    def get_params(self) -> None:
        """
        Retrieve and configure ROS2 parameters.

        Declares and retrieves parameters from the ROS2 parameter server,
        including topic names, MCP servers, and system prompt template path.
        Logs each parameter value for verification.

        Parameters:
            None

        Returns:
            None

        Note:
            Sets self.query_topic, self.response_topic, self.mcp_servers, and self.system_prompt.
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
        self.declare_parameter('system_prompt', 'system_prompt.jinja')
        self.system_prompt = self.get_parameter(
            'system_prompt').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter system_prompt is set to: [{self.system_prompt}]')
        
        # Declare and retrieve LLM model name parameter
        self.declare_parameter('llm_model', 'Qwen/Qwen2.5-0.5B-Instruct')
        self.llm_model = self.get_parameter(
            'llm_model').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter llm_model is set to: [{self.llm_model}]')
        
        # Declare and retrieve bolean parameter for int 8 quantization
        self.declare_parameter('use_int8', True)
        self.use_int8 = self.get_parameter(
            'use_int8').get_parameter_value().bool_value
        self.get_logger().info(
            f'The parameter use_int8 is set to: [{self.use_int8}]')


def main(args=None) -> None:
    """
    Run the LangGraph ROS2 action server.

    Initialize the ROS2 context, create the agent node, and spin
    until shutdown is requested.

    Parameters:
        args: Command-line arguments (optional).

    Returns:
        None
    """
    try:
        with rclpy.init(args=args):
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
