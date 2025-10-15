# Pisito Agent

[![ROS2](https://img.shields.io/badge/ROS2-Jazzy-blue)](https://docs.ros.org/en/jazzy/index.html)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A ROS2 package implementing a conversational agent using Hugging Face LLMs with MCP (Model Context Protocol) tool calling capabilities. This agent can be integrated into robotic systems to provide natural language interaction with external tools and APIs.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [MCP Integration](#mcp-integration)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Hugging Face LLM Integration**: Support for any Hugging Face model compatible with the transformers library
- **MCP Tool Calling**: Dynamic tool integration via Model Context Protocol servers
- **8-bit Quantization**: Optional INT8 quantization for efficient inference on consumer GPUs
- **Configurable Parameters**: All model and generation parameters exposed via ROS2 parameters
- **ROS2 Pub/Sub**: Simple topic-based communication for queries and responses
- **Multi-threaded Execution**: Concurrent callback processing for responsive behavior
- **Custom System Prompts**: Configurable system prompts via template files

## Prerequisites

- **ROS2**: Jazzy or later
- **Python**: 3.10 or higher
- **CUDA-capable GPU**: Recommended for inference (CPU fallback available)
- **Virtual Environment**: Python venv with required dependencies

### Python Dependencies

```bash
transformers
torch
langchain-mcp-adapters
bitsandbytes  # For 8-bit quantization
```

## Installation

### 1. Clone the repository

```bash
cd ~/colcon_ws/src
git clone https://github.com/grupo-avispa/pisito_agent.git
```

### 2. Set up Python virtual environment

```bash
cd ~/colcon_ws
python3 -m venv llm_venv
source llm_venv/bin/activate
pip install -r src/pisito_agent/requirements.txt
```

### 3. Build the ROS2 workspace

```bash
cd ~/colcon_ws
colcon build --packages-select pisito_agent
source install/setup.bash
```

## Usage

### Basic Launch

Launch the agent with default parameters:

```bash
source ~/colcon_ws/llm_venv/bin/activate
export VIRTUAL_ENV=~/colcon_ws/llm_venv
ros2 launch pisito_agent default.launch.py
```

### Sending Queries

Publish a user query to the agent:

```bash
ros2 topic pub /user_query std_msgs/msg/String "data: 'What is the weather in Málaga?'" --once
```

### Receiving Responses

Subscribe to agent responses:

```bash
ros2 topic echo /llm_response
```

## Configuration

### ROS2 Parameters

All parameters are configurable via `params/default_params.yaml`.

### MCP Servers Configuration

Define MCP servers in `params/mcp.json`.

### System Prompt

Customize the agent's behavior in `templates/system_prompt.jinja`.

## Architecture

### Node Structure

```
┌─────────────────────────────────────┐
│        Pisito Agent Node            │
│                                     │
│  ┌──────────────────────────────┐  │
│  │   ROS2 Subscriber            │  │
│  │   (user_query topic)         │  │
│  └──────────┬───────────────────┘  │
│             │                       │
│             ▼                       │
│  ┌──────────────────────────────┐  │
│  │   Tokenizer & LLM            │  │
│  │   (Hugging Face)             │  │
│  └──────────┬───────────────────┘  │
│             │                       │
│             ▼                       │
│  ┌──────────────────────────────┐  │
│  │   Response Processor         │  │
│  │   (Tool Call Extraction)     │  │
│  └──────────┬───────────────────┘  │
│             │                       │
│             ▼                       │
│  ┌──────────────────────────────┐  │
│  │   MCP Client                 │  │
│  │   (Tool Invocation)          │  │
│  └──────────┬───────────────────┘  │
│             │                       │
│             ▼                       │
│  ┌──────────────────────────────┐  │
│  │   ROS2 Publisher             │  │
│  │   (llm_response topic)       │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- **Grupo Avispa** - *Universidad de Málaga*

## Acknowledgments

- Hugging Face for transformers library
- ROS2 community
- Model Context Protocol (MCP) developers

## References

- [ROS2 Documentation](https://docs.ros.org/en/jazzy/index.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [BitsAndBytes Quantization](https://github.com/TimDettmers/bitsandbytes)

---

**Note**: This package is part of the AVISPA robotics framework for service robots.
