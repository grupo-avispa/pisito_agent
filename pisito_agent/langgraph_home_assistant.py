"""
LangGraph manager for agent flow control.
"""

import time
import logging
from langgraph.graph import START, StateGraph, END
from langsmith import traceable
from pisito_agent.ollama_utils import Ollama, Messages
from pisito_agent.langgraph_base import LangGraphBase

class LangGraphManager(LangGraphBase):
    """
    Manager class for LangGraph-based home assistant.

    This class encapsulates all functionality for the LangGraph workflow,
    including LLM response generation, step management, and interaction finalization.
    Attributes:
        graph (StateGraph): The compiled LangGraph workflow.
        logger: Optional ROS2 logger for logging messages.
        ollama_agent (Ollama): Instance of the Ollama agent for LLM interactions.
        steps (int): Counter for the number of steps taken in the conversation.
        messages_count (int): Counter for the number of messages exchanged.
        max_steps (int): Maximum allowed steps before finishing interaction.
    Methods:
        query_response(state: Messages) -> Messages:
            Generates LLM response based on conversation state.
        manage_steps(state: Messages) -> str:
            Determines the next step in the conversation flow.
        finish_ollama_interaction(state: Messages) -> Messages:
            Finalizes the Ollama interaction and returns the final response.
        make_graph() -> None:
            Initializes and compiles the LangGraph workflow.
    """

    def __init__(self,
                 logger=None,
                 ollama_agent: Ollama = None,
                 max_steps: int = 5):
        """
        Initialize the LangGraph Manager.

        Creates the LLM instance with default configuration.

        Parameters:
            logger: Optional ROS2 logger to use for logging (default: None).

        Returns:
            None
        """
        super().__init__(logger=logger,
                           ollama_agent=ollama_agent, 
                           max_steps=max_steps)
        if self.ollama_agent is None:
            raise ValueError("Ollama agent instance must be provided to LangGraphManager.")

    @traceable
    async def query_response(self, state: Messages) -> Messages:
        """
        Generate LLM response based on conversation state.
        Receives the current conversation message list from ollama agent
        and updates state with LLM response.

        Parameters:
            state (Messages): Current conversation state with messages.

        Returns:
            Messages: Updated state with agent response.
        """
        initial_time = time.time()
        
        # Invoke Ollama agent
        try:
            state = await self.ollama_agent.invoke(state=state)
        except ValueError as e:
            self._log(f"Error during Ollama agent invocation: {e}")
            raise e
        
        step_time = time.time() - initial_time
        self._log(f"LLM response generated in {step_time:.3f} seconds.")
        
        return state

    @traceable
    def manage_steps(self, state: Messages) -> str:
        """
        Determine the next step in the conversation flow.

        Checks if the last message contains a tool call to decide whether
        to continue querying or finish the interaction.

        Parameters:
            state (Messages): Current conversation state with messages.
        Returns:
            str: Next node to transition to ('query_response' or 'finish_ollama_interaction').
        """
        self.steps += 1
        uc = 'finish'
        self._log(f"Managing steps, current step: {self.steps}")
        try:
            # Check if the last message contains a tool call
            if state['messages'] and state['messages'][-1]['role'] == 'tool':
                # for msg in state['messages'][self.messages_count:]:
                #     self._log("--"*30)
                #     self._log(f"\nMessage Role:\n{msg['role']}\nContent:\n{msg['content']}\n" + 
                #               f"Reasoning:\n{msg['reasoning_content']}\nTool Calls:\n{msg['tool_calls']}")
                if self.steps < self.max_steps:
                    uc = 'agent'
                else:
                    self._log("Maximum steps reached, finishing interaction.")
            else:
                self._log("No tool call detected, finishing interaction.")
                self._log("Final response from assistant:\n" +
                        f"{state['messages'][-1]['content']}")
            # Update messages count
            self.messages_count = len(state['messages'])
            self._log(f"Total messages in conversation: {self.messages_count}")
        except Exception as e:
            self._log(f"Error in manage_steps: {e}")
            uc = 'finish'
        return uc
    
    @traceable
    async def finish_ollama_interaction(self, state: Messages) -> Messages:
        """
        Finalize the Ollama interaction and return the final response.

        Parameters:
            state (Messages): Current conversation state with messages.
        Returns:
            Messages: Final state after finishing interaction.
        """

        self._log("Finalizing Ollama interaction.")
        if self.steps >= self.max_steps:
            self._log("Maximum steps reached during finalization.")
        else:
            self._log("Agent reached final state before maximum steps.")
        
        self.ollama_agent.reset_memory()
        return state

    async def make_graph(self):
        """
        Initialize and compile the LangGraph workflow.

        This method creates a LangGraph StateGraph with nodes for query processing and
        conversation finalization. It defines the flow of the conversation based on
        LLM outputs and compiles the graph for execution.

        Returns:
            None: The compiled graph is stored in self.graph.
        """

        # Create the StateGraph workflow
        workflow = StateGraph(Messages)

        # Add graph nodes:
        # - query_response: Main LLM reasoning node
        workflow.add_node('query_response', self.query_response)
        # - finish_ollama_interaction: Final node to end interaction
        workflow.add_node('finish_ollama_interaction', self.finish_ollama_interaction)

        # Define graph edges and flow:
        # After start, proceed to query response
        workflow.add_edge(START, 'query_response')
        # After a agent step, check end conditions and proceed accordingly
        workflow.add_conditional_edges(
            'query_response',
            self.manage_steps, 
            {'agent': 'query_response', 'finish': 'finish_ollama_interaction'},
        )

        # Compile the graph workflow
        self.graph = workflow.compile()
