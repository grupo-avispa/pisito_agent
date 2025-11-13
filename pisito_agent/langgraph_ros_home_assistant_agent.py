"""
ROS2 server for LangGraph-based conversational AI.
"""

# Time import for measuring LLM call duration
import time

# Custom graph manager class
from pisito_agent.langgraph_home_assistant import LangGraphManager
# Custom Ollama agent and message utilities
from langgraph_base_ros.ollama_utils import Messages
# Base class import
from langgraph_base_ros.langgraph_ros_base import LangGraphRosBase

# ROS2 imports for subscriber and server implementation
import rclpy
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.callback_groups import ReentrantCallbackGroup
from llm_interactions_msgs.srv import UserQueryResponse

# ============= ROS2 SERVER =============

class RosHomeAssistantAgent(LangGraphRosBase):
    """
    ROS2 server for LangGraph-based conversational AI.
    This class sets up a ROS2 node that listens for user queries, processes them
    using a LangGraph workflow, and publishes the generated responses.
    This implements a home assistant use case.
    Attributes:
        graph_manager (LangGraphManager): The LangGraph manager instance for workflow control.
    Methods:
        agent_callback(msg: String) -> None:
            Callback function for processing user queries.
    """

    # ============= INITIALIZATION =============

    def __init__(self):
        """
        Initialize the LangGraph ROS node.
        """
        # Call the base class initializer
        super().__init__()

        # Initialize LangGraph manager with ROS2 logger
        self.graph_manager = LangGraphManager(logger=self.get_logger(),
                                              ollama_agent=self.ollama_agent,
                                              max_steps=self.max_steps)
        
        # Build the LangGraph workflow
        self.build_graph()

        # Create the subscriber to listen for user queries
        self.group = ReentrantCallbackGroup()
        self.agent_srv = self.create_service(
            srv_type=UserQueryResponse,
            srv_name=self.service_name,
            callback=self.agent_callback,
            callback_group=self.group
        )

        self.get_logger().info('langgraph_agent_node node initialized.')
    
    def build_graph(self) -> None:
        """
        Initialize and compile the LangGraph workflow.
        Parameters:
            graph_manager (LangGraphManager): The LangGraph manager instance to use.
        Returns:
            None
        """
        # Initialize and compile the LangGraph workflow
        try:
            self.loop.run_until_complete(self.graph_manager.make_graph())
        except Exception as e:
            self.get_logger().error(f'Failed to create LangGraph workflow: {e}')
            raise

        self.get_logger().info('LangGraphManager graph created successfully...')


    # ============= SERVICE CALLBACKS =============

    def agent_callback(self, request, response):
        """
        Callback function for processing user queries.

        Receives a user request, processes it using the agent graph,
        and returns the generated response.

        Parameters:
            request: The UserQueryResponse request containing user query details.
            response: The UserQueryResponse response to be populated with the agent's reply.
        Returns:
            The populated response with the agent's generated reply.
        """

        user_query = request.user_query
        user_name = request.user_name
        if user_name:
            user_query = f'User named {user_name} says: {user_query}'
        
        self.get_logger().info(f'Received user query:\n{user_query}')

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

        # rreturn the response to the user request
        response.response_text = result['messages'][-1].get('content', '')
        return response


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
        agent = RosHomeAssistantAgent()

        # Use a MultiThreadedExecutor to allow concurrent callback execution
        executor = MultiThreadedExecutor()
        executor.add_node(agent)

        # Spin the node to process callbacks
        executor.spin()
    except (KeyboardInterrupt, Exception, ExternalShutdownException) as e:
        print(f'Shutting down agent node due to: {e}')


if __name__ == '__main__':
    main()
