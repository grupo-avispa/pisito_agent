"""
ROS2 node implementing a conversational agent using a Huggingface LLM.
The agent subscribes to user queries and publishes generated responses.
The agent's behavior is configurable via ROS2 parameters.
"""

# General imports 
import json
from typing import Tuple

# Transformer import for LLM interactions
from smolagents import MCPClient

# Import for customm model and agent objects
from custom_model import QuantModel
from custom_agent import CustomAgent

# ROS2 imports for subscriber and publisher implementation
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String

# ============= ROS2 NODE =============

class RosAgent(Node):
    # ============= INITIALIZATION =============

    def __init__(self):
        """
        Initialize the RosAgent node.

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

        self.get_logger().info('RosAgent node initialized.')
    
    # ============= METHODS =============

    def agent_callback(self, msg: String) -> None:
        """
        Callback function for processing user queries.

        Receives a user query message, processes it using the agent,
        and publishes the generated response.

        Parameters:
            msg (String): The incoming user query message.
        Returns:
            None
        """
        user_query = msg.data
        self.get_logger().info(f'Received user query: {user_query}')

        # Call the agent to process the query and generate a response
        result = self.agent.run(user_query, max_steps=self.max_steps)
        # Proces result for pretty string output
        steps_summary = f"User query:{user_query}\n"
        if self.return_full_result:
            for step in result.steps:
                steps_summary += (
                    "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Step {step.get('step_number', '')}\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Model output:\n{step.get('model_output', '')}\n\n"
                    f"Tool calls:\n{step.get('tool_calls', [])}\n\n"
                    f"Observations:\n{step.get('observations', [])}\n"
                )
            steps_summary += f"Final Answer:\n{result.output}\n"
        else:
            steps_summary += f"Final Answer:\n{result}\n"
        response_msg = String()
        response_msg.data = steps_summary
        # Publish the response
        self.response_pub.publish(response_msg)
        self.get_logger().info(f'Published agent response.')

    def initialize_llm(self)-> None:
        """
        Initialize the LLM model and ReAct agent.
        Also initializes the MCP client and retrieves available tools.

        Parameters:
            None
        Returns:
            None
        """
        tools = []
        try:
            with open(self.mcp_servers, 'r') as f:
                server_parameters = json.load(f)
            mcp_client = MCPClient(server_parameters)
            tools = mcp_client.get_tools()
            self.get_logger().info(f'Successfully loaded {len(tools)} tools from MCP server')
        except Exception as e:
            self.get_logger().error(f'Error initializing tools: {str(e)}')
            self.get_logger().warning('Continuing with no tools available')
            
        # Use the tools with your agent
        model = QuantModel(
                model_id=self.llm_model,
                tools=tools,  
                tool_call_pattern=self.tool_call_pattern,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                load_in_8bit=self.use_int8,
                do_sample=self.do_sample
            )

        self.agent = CustomAgent(
            tools=tools,  
            model=model,
            sys_prompt_file=self.system_prompt_file,
            max_steps=self.max_steps,
            return_full_result=self.return_full_result
        )
        self.get_logger().info(f'Successfully initialized LLM model and ReAct agent.')


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
        self.declare_parameter('max_new_tokens', 256)
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
        
        self.declare_parameter('do_sample', True)
        self.do_sample = self.get_parameter(
            'do_sample').get_parameter_value().bool_value
        self.get_logger().info(
            f'The parameter do_sample is set to: [{self.do_sample}]')
        
        self.declare_parameter('return_full_result', False)
        self.return_full_result = self.get_parameter(
            'return_full_result').get_parameter_value().bool_value
        self.get_logger().info(
            f'The parameter return_full_result is set to: [{self.return_full_result}]')
        
        self.declare_parameter('max_steps', 5)
        self.max_steps = self.get_parameter(
            'max_steps').get_parameter_value().integer_value
        self.get_logger().info(
            f'The parameter max_steps is set to: [{self.max_steps}]')


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
