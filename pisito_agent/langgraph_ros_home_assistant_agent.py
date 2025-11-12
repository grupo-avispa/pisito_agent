"""
ROS2 publisher/subscriber for LangGraph-based conversational AI.
"""

# Time import for measuring LLM call duration
import time

# Custom graph manager class
from pisito_agent.langgraph_home_assistant import LangGraphManager
# Custom Ollama agent and message utilities
from pisito_agent.ollama_utils import Messages
# Base class import
from pisito_agent.langgraph_ros_base import LangGraphRosBase

# ROS2 imports for subscriber and publisher implementation
import rclpy
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String

# ============= ROS2 SERVER =============


class RosHomeAssistantAgent(LangGraphRosBase):
    """
    ROS2 publisher/subscriber for LangGraph-based conversational AI.
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
        self.build_graph(self.graph_manager)

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


    # ============= SUBSCRIPTION CALLBACKS =============

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
