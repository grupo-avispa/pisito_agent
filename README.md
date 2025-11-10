# pisito_agent

ROS2 package providing two different implementations of intelligent conversational agents for robots, using language models (LLMs) and tools via the Model Context Protocol (MCP).

## Overview

This package offers two distinct approaches for implementing conversational agents in ROS2:

1. **LangGraph + Ollama Custom**: Uses LangGraph to manage the agent's workflow and Ollama as the backend to run LLM models locally.

2. **SmoLAgents + Custom Model/Agent**: Implements an agent based on Hugging Face's SmoLAgents library with custom quantized models.

Both implementations use MCP (Model Context Protocol) to provide external tools to the agent, enabling integration with external services and extended functionalities.

## Architecture

### Common Components

- **ROS2 Node**: Subscriber/Publisher for system communication
- **MCP Client**: Client to connect with MCP servers and retrieve tools
- **Prompt System**: Jinja2 templates for customizable system prompts
- **Configurable Parameters**: YAML file with all configuration

### ROS2 Topics

- **Input** (default: `user_query`): Receives user queries as `std_msgs/String` messages
- **Output** (default: `llm_response`): Publishes generated responses as `std_msgs/String` messages

### File Structure

```
pisito_agent/
├── launch/
│   ├── langgraph.launch.py          # Launcher for LangGraph
│   └── smolagents.launch.py         # Launcher for SmoLAgents
├── params/
│   ├── default_params.yaml          # Default parameters
│   ├── langgraph_mcp.json          # MCP config for LangGraph
│   └── smolagents_mcp.json         # MCP config for SmoLAgents
├── templates/
│   ├── system_prompt.jinja         # System prompt template
│   └── qwen3.jinja                 # Chat template for Qwen3
├── pisito_agent/
│   ├── langgraph_ros_agent.py      # LangGraph ROS2 node
│   ├── smolagent_ros_agent.py      # SmoLAgents ROS2 node
│   ├── langgraph_functions.py      # LangGraph graph functions
│   ├── ollama_utils.py             # Ollama utilities
│   ├── smolagent_custom_agent.py   # Custom agent
│   ├── smolagent_custom_model.py   # Custom model
│   └── fake_mcp_server.py          # Test MCP server
├── .env                            # Environment variables (LangSmith)
├── package.xml
└── setup.py
```

## Prerequisites

### Option A: Native Installation

1. **Ollama installed and running** (optional, for LangGraph flow):
   ```bash
   # Install Ollama (if not installed)
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Download the model
   ollama pull qwen3:0.6b
   
   # Verify Ollama is running
   ollama list
   ```

2. **Configured MCP server**:
   - Ensure the MCP server specified in `<agent_flow>_mcp.json` is accessible
   - Server must implement the MCP protocol correctly

3. **Python virtual environment** with dependencies:
   Ensure your python vitrual enviroment is sourced in the current terminal and dependences have been installed.
   ```bash
    python -m venv agent-venv
    source agent-venv/bin/activate
    pip install -r requirements.txt
    ```

4. **Environment variables** (optional, for LangSmith tracing):
   Edit the `.env` file in the package:
   ```bash
   LANGSMITH_API_KEY=your_api_key
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=project_name
   ```
    
6. **Building**:
    ```bash
    cd ~/colcon_ws
    source /opt/ros/humble/setup.bash
    colcon build --packages-select pisito_agent --symlink-install
    source install/setup.bash
    ```

### Option B: Docker Installation
Alternatively, you can run the agent using Docker, which includes all dependencies pre-configured.

1. **Build Docker image**:
   ```bash
   cd ~/colcon_ws/src/interaction/pisito_agent/Docker
   docker compose build
   ```
2. **Run with VsCode Dev Containers**:
   - Open the `pisito_agent` folder in VSCode
   - Reopen in container when prompted
   - Use the integrated terminal to run the agent

## Option 1: LangGraph + Ollama Custom

### Key Features

- **LangGraph Workflow**: State and conversation flow management via graphs
- **Ollama Backend**: Local execution of optimized LLM models
- **Raw Mode**: Direct control over message format without model templates
- **Chat Templates**: Support for custom templates (e.g., qwen3.jinja)
- **Optional Thinking**: Capability to enable intermediate reasoning
- **Async Operations**: Asynchronous processing for better performance

### Highlights

- Graph-based workflow with nodes and transitions
- Fine-grained control over agent lifecycle
- Native integration with LangSmith for tracing and debugging
- Greater flexibility in managing complex states
- Ideal for advanced conversational flows

### Configuration Parameters

File: `params/default_params.yaml` (section `langgraph_agent_node`)

```yaml
langgraph_agent_node:
  ros__parameters:
    # Communication topics
    query_topic: "user_query"          # Input topic
    response_topic: "llm_response"     # Output topic
    
    # Configuration files
    mcp_servers: "langgraph_mcp.json"           # MCP servers configuration
    system_prompt_file: "system_prompt.jinja"   # System prompt template
    model_chat_template_file: "qwen3.jinja"     # Model chat template
    
    # LLM Model
    llm_model: "qwen3:0.6b"            # Ollama model to use
    
    # Tool call extraction
    tool_call_pattern: "<tool_call>(.*?)</tool_call>"
    
    # Ollama generation parameters
    raw_mode: true                     # Use raw mode (without model template)
    debug_mode: false                  # Enable debug logs
    temperature: 0.0                   # Randomness (0.0 = deterministic)
    repeat_penalty: 1.1                # Repetition penalty
    top_k: 10                          # Top-K sampling
    top_p: 0.25                        # Top-P (nucleus) sampling
    num_ctx: 8192                      # Context size
    num_predict: 128                   # Maximum tokens to generate
    
    # LangGraph workflow parameters
    max_steps: 5                       # Maximum agent steps
    enable_thinking: false             # Enable intermediate thinking
```

### MCP Configuration Files

File `params/langgraph_mcp.json` defines the MCP servers to connect to. Can contain multiple servers.

### Getting Started

#### Running

```bash
# Activate virtual environment
source ~/venvs/pisito_agent/bin/activate

# Launch agent with default configuration
ros2 launch pisito_agent langgraph.launch.py
```

## Option 2: SmoLAgents + Custom Model/Agent

### Key Features

- **SmoLAgents Framework**: Lightweight Hugging Face library for agents
- **Quantized Models**: INT8-INT4 quantization support to reduce memory usage
- **Hugging Face Transformers**: Direct access to thousands of pre-trained models
- **Custom Agent**: Custom implementation of the ReAct cycle
- **Custom Model**: Custom wrapper for transformer models

### Highlights

- Lighter, agent-specific framework
- Local execution without external service dependencies (except MCP)
- Full control over model and tokenization
- Ideal for resource-constrained scenarios
- Easy integration with Hugging Face models

### Configuration Parameters

File: `params/default_params.yaml` (section `hf_agent`)

```yaml
hf_agent:
  ros__parameters:
    # Communication topics
    query_topic: "user_query"          # Input topic
    response_topic: "llm_response"     # Output topic
    
    # Configuration files
    mcp_servers: "smolagents_mcp.json"          # MCP servers configuration
    system_prompt_file: "system_prompt.jinja"   # System prompt template
    
    # Hugging Face LLM model
    llm_model: "Qwen/Qwen3-0.6B"       # Model ID on Hugging Face Hub
    
    # Quantization
    use_int8: true                     # Use INT8 quantization
    
    # Tool call extraction
    tool_call_pattern: "<tool_call>(.*?)</tool_call>"
    
    # Generation parameters
    max_new_tokens: 256                # Maximum tokens to generate
    top_k: 10                          # Top-K sampling
    temperature: 0.1                   # Randomness (higher = more creative)
    repetition_penalty: 1.1            # Repetition penalty
    do_sample: false                   # Use sampling or greedy decoding
    
    # Agent parameters
    return_full_result: false          # Return details of each step
    max_steps: 8                       # Maximum agent steps
    enable_thinking: false             # Enable intermediate thinking
```

### MCP Configuration Files

File: `params/smolagents_mcp.json` defines the MCP servers to connect to. Can contain multiple servers.

#### Running

```bash
# Activate virtual environment
source ~/venvs/pisito_agent/bin/activate

# Launch agent with default configuration
ros2 launch pisito_agent smolagents.launch.py
```

## Customization

### System Prompts

Edit `templates/system_prompt.jinja` to change agent behavior, for example:

```jinja
You are a helpful robot assistant.
You can use the tools to improve your behavior.
Always think step by step and use tools when necessary.
```

### Chat Templates

For LangGraph, add model templates to `/templates` folder to adjust message format according to the model used.

### MCP Configuration

Add additional MCP servers in the JSON files:

```json
{
    "server1": {
        "url": "http://localhost:3001/mcp"
    },
    "server2": {
        "url": "http://localhost:3002/mcp"
    }
}
```

For testing purposes, you can use the provided `fake_mcp_server.py` to simulate an MCP server.

## Development

### Adding New Tools

Tools are automatically retrieved from the MCP server. To add new ones:

1. Implement the tool on the MCP server
2. Restart the MCP server
3. Relaunch the ROS2 agent

### Modifying Agent Flow

**LangGraph**: Edit `pisito_agent/langgraph_functions.py` to change the graph

**SmoLAgents**: Not supported modifications over the ReAct cycle.

## License

Apache License 2.0

## Maintainer

Oscar Pons Fernandez (opfernandez@uma.es)

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama](https://ollama.com/)
- [SmoLAgents](https://github.com/huggingface/smolagents)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
