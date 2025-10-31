"""
"""

# Time import for measuring LLM call duration
import time
# asyncio for async operations
import asyncio
# JSON import for reading MCP server configurations
import json

# Custom graph manager class
from pisito_agent.langgraph_functions import LangGraphManager
# Custom Ollama agent and message utilities
from pisito_agent.ollama_utils import Ollama, Messages
# MCP client for tool retrieval
from fastmcp import Client

# ROS2 imports for subscriber and publisher implementation
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String

# ============= ROS2 SERVER =============


class RosAgent(Node):
    """
    ROS2 Action Server for LangGraph-based conversational AI.
    This class sets up a ROS2 node that listens for user queries, processes them
    using a LangGraph workflow, and publishes the generated responses.
    Attributes:
        query_topic (str): Topic name for receiving user queries.
        response_topic (str): Topic name for publishing LLM responses.
        mcp_servers (str): MCP server configuration for tool retrieval.
        system_prompt (str): System prompt template for LLM interactions.
        loop: Asyncio event loop for asynchronous operations.
        ollama_agent (Ollama): Instance of the Ollama agent for LLM interactions.
        graph_manager (LangGraphManager): Instance of the LangGraph manager for workflow control.
    """

    # ============= INITIALIZATION =============

    def __init__(self):
        """
        Initialize the LangGraph Action Server.
        """
        super().__init__('langgraph_agent_node')

        # Retrieve ROS2 parameters (topic name, MCP servers, system prompt, etc.)
        self.get_params()

        # Create a persistent event loop for async operations
        # Using asyncio.new_event_loop() to avoid deprecation warning
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Initialize ollama agent
        try:
            self.loop.run_until_complete(self.initialize_ollama_agent())
        except Exception as e:
            self.get_logger().error(f'Failed to initialize Ollama agent: {e}')
            raise

        # Initialize LangGraph manager with ROS2 logger
        self.graph_manager = LangGraphManager(logger=self.get_logger(),
                                              ollama_agent=self.ollama_agent,
                                              max_steps=self.max_steps)

        # Initialize and compile the LangGraph workflow
        try:
            self.loop.run_until_complete(self.graph_manager.make_graph())
        except Exception as e:
            self.get_logger().error(f'Failed to create LangGraph workflow: {e}')
            raise

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

        self.get_logger().info('langgraph_agent_node node initialized.')


    # ============= ACTION SERVER CALLBACKS =============

    def agent_callback(self, msg: String) -> None:
        """
        Callback function for processing user queries.

        Receives a user query message, processes it using the agent graph,
        and publishes the generated response.

        Parameters:
            msg (String): The incoming user query message.
        Returns:
            None
        """

        user_query = msg.data
        self.get_logger().info(f'Received user query: {user_query}')

        init_time = time.time()

        # Prepare the initial conversation state with system prompt and user query
        initial_state: Messages = {
            'messages': [
                self.ollama_agent.create_message(
                    role='system',
                    content=self.system_prompt
                )
            ]
        }
        initial_state['messages'].append(
            self.ollama_agent.create_message(
                role='user',
                content=user_query
            )
        )
        # Run the agent graph asynchronously
        result = self.loop.run_until_complete(
            self.graph_manager.graph.ainvoke(initial_state)
        )

        # Log processing time and generated response
        self.get_logger().info(f'Agent processing time: {time.time() - init_time:.3f} seconds')
        self.get_logger().info(f'Generated response: {result["messages"][-1].get("content", "")}')

        # Publish the response
        response_msg = String()
        response_msg.data = result['messages'][-1].get('content', '')
        self.response_pub.publish(response_msg)

    async def initialize_ollama_agent(self) -> None:
        """
        Initialize the mcp client and Ollama agent with the required parameters.

        Returns:
            None
        """
        # Initialize MCP client
        try:
            self.get_logger().info(f'Loading MCP servers from: {self.mcp_servers}')
            with open(self.mcp_servers, 'r') as f:
                mcp_servers_config = json.load(f)
            self.get_logger().info(f'MCP servers config: {mcp_servers_config}')
        except FileNotFoundError:
            self.get_logger().error(f'MCP servers file not found: {self.mcp_servers}')
            raise
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Invalid JSON in MCP servers file: {e}')
            raise
        
        self.mcp_client = Client(mcp_servers_config)
        
        # Retrieve available tools from MCP
        self.get_logger().info('Retrieving tools from MCP servers...')
        async with self.mcp_client:
            mcp_tools = await self.mcp_client.list_tools()
        self.tools = []
        # Prepare tool definitions for Ollama agent
        for tool in mcp_tools:
            self.tools.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
            })
        self.get_logger().info(f'Retrieved {len(self.tools)} tools from MCP servers')
        
        # Get system prompt template content
        try:
            self.get_logger().info(f'Loading system prompt from: {self.system_prompt_file}')
            with open(self.system_prompt_file, 'r') as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
            self.get_logger().error(f'System prompt file not found: {self.system_prompt_file}')
            raise
        
        # Initialize Ollama agent with retrieved tools and parameters
        self.get_logger().info(f'Initializing Ollama agent with model: {self.llm_model}')
        self.ollama_agent = Ollama(
            model=self.llm_model,
            tools=self.tools,
            tool_call_pattern=self.tool_call_pattern,
            mcp_client=self.mcp_client,
            think=self.enable_thinking,
            raw=self.raw_mode,
            temperature=self.temperature,
            jinja_template_path=self.chat_template_path,
            system_prompt=self.system_prompt,
            debug=self.debug_mode,
            repeat_penalty=self.repeat_penalty,
            top_k=self.top_k,
            top_p=self.top_p,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
        )
        self.get_logger().info('Ollama agent initialized successfully')

    def get_params(self) -> None:
        """
        Retrieve and configure ROS2 parameters.

        Declares and retrieves parameters from the ROS2 parameter server,
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
        
        # Declare and retrieve model chat template file path parameter
        self.declare_parameter('model_chat_template_file', 'qwen3.jinja')
        self.chat_template_path = self.get_parameter(
            'model_chat_template_file').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter model_chat_template_file is set to: [{self.chat_template_path}]')
        
        # Declare and retrieve LLM model name parameter
        self.declare_parameter('llm_model', 'qwen3:0.6b')
        self.llm_model = self.get_parameter(
            'llm_model').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter llm_model is set to: [{self.llm_model}]')
        
        # Declare tool call regex pattern to extract tool calls from LLM response
        self.declare_parameter('tool_call_pattern', '<tool_call>(.*?)</tool_call>')
        self.tool_call_pattern = self.get_parameter(
            'tool_call_pattern').get_parameter_value().string_value
        self.get_logger().info(
            f'The parameter tool_call_pattern is set to: [{self.tool_call_pattern}]')
        
        # Declare and retrieve Ollama generation parameters
        self.declare_parameter('raw_mode', True)
        self.raw_mode = self.get_parameter(
            'raw_mode').get_parameter_value().bool_value
        self.get_logger().info(
            f'The parameter raw_mode is set to: [{self.raw_mode}]')
        
        self.declare_parameter('debug_mode', False)
        self.debug_mode = self.get_parameter(
            'debug_mode').get_parameter_value().bool_value
        self.get_logger().info(
            f'The parameter debug_mode is set to: [{self.debug_mode}]')
        
        self.declare_parameter('temperature', 0.0)
        self.temperature = self.get_parameter(
            'temperature').get_parameter_value().double_value
        self.get_logger().info(
            f'The parameter temperature is set to: [{self.temperature}]')
        
        self.declare_parameter('repeat_penalty', 1.1)
        self.repeat_penalty = self.get_parameter(
            'repeat_penalty').get_parameter_value().double_value
        self.get_logger().info(
            f'The parameter repeat_penalty is set to: [{self.repeat_penalty}]')
        
        self.declare_parameter('top_k', 10)
        self.top_k = self.get_parameter(
            'top_k').get_parameter_value().integer_value
        self.get_logger().info(
            f'The parameter top_k is set to: [{self.top_k}]')
        
        self.declare_parameter('top_p', 0.25)
        self.top_p = self.get_parameter(
            'top_p').get_parameter_value().double_value
        self.get_logger().info(
            f'The parameter top_p is set to: [{self.top_p}]')
        
        self.declare_parameter('num_ctx', 8192)
        self.num_ctx = self.get_parameter(
            'num_ctx').get_parameter_value().integer_value
        self.get_logger().info(
            f'The parameter num_ctx is set to: [{self.num_ctx}]')
        
        self.declare_parameter('num_predict', 256)
        self.num_predict = self.get_parameter(
            'num_predict').get_parameter_value().integer_value
        self.get_logger().info(
            f'The parameter num_predict is set to: [{self.num_predict}]')
        
        # Declare and retrieve LangGraph workflow parameters
        self.declare_parameter('max_steps', 5)
        self.max_steps = self.get_parameter(
            'max_steps').get_parameter_value().integer_value
        self.get_logger().info(
            f'The parameter max_steps is set to: [{self.max_steps}]')
        
        self.declare_parameter('enable_thinking', False)
        self.enable_thinking = self.get_parameter(
            'enable_thinking').get_parameter_value().bool_value
        self.get_logger().info(
            f'The parameter enable_thinking is set to: [{self.enable_thinking}]')
        


def main(args=None) -> None:
    """
    Run the ROS2 agent.

    Initialize the ROS2 context, create the agent node, and spin
    until shutdown is requested. Uses a MultiThreadedExecutor 
    for concurrent callbacks.

    Parameters:
        args: Command-line arguments (optional).

    Returns:
        None
    """
    rclpy.init(args=args)
    
    try:
        # Create the agent node
        agent = RosAgent()

        # Use a MultiThreadedExecutor to allow concurrent callback execution
        executor = MultiThreadedExecutor()
        executor.add_node(agent)

        # Spin the node to process callbacks
        executor.spin()
    except (KeyboardInterrupt, Exception, ExternalShutdownException) as e:
        print(f'Shutting down agent node due to: {e}')


if __name__ == '__main__':
    main()
