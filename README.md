# agent-chat

A lightweight Python framework for running structured multi-agent conversations using LLMs. Agents are defined in YAML, take turns according to configurable policies, and a post-processor can summarize the result.

The included example simulates a **scrum planning meeting**: a `planner` and a `critic` collaborate on a problem defined in `prompt.txt`, then the planner produces a final summary of agreed user stories.

## How it works

1. Agents are loaded from YAML files in the `agents/` directory.
2. A `Conversation` is created with those agents and a shared system prompt.
3. A user prompt is injected (from `prompt.txt`).
4. `conv.run()` drives the conversation using:
   - a **turn selector** (e.g. `round_robin`) to decide who speaks next
   - a **stop condition** (e.g. `max_turns`) to end the loop
   - optional **post-processors** (e.g. `summarize`) that run after the loop

## Project structure

```
agent_chat/
  agents.py       # Agent dataclass + YAML loader
  conversation.py # Conversation loop, message history, LLM calls
  policies.py     # Turn selectors, stop conditions, post-processors
agents/
  planner.yaml    # Strategic planner agent definition
  critic.yaml     # Constructive critic agent definition
main.py           # Entry point
prompt.txt        # The problem/task given to the agents
```

## Setup

**Requirements:** Python 3.13+, [uv](https://docs.astral.sh/uv/)

```bash
uv sync
```

Create a `.env` file with your API credentials:

```env
# Anthropic
ANTHROPIC_API_KEY=your_key_here

# Azure OpenAI — classic GPT deployments (AzureOpenAI client)
# Use the bare base URL — do NOT include /openai/v1/
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_API_VERSION=2024-02-01   # optional, defaults to 2024-02-01

# Azure AI Foundry — serverless models e.g. Mistral, Llama (OpenAI-compatible client)
# Must include /openai/v1/ — the URL is used as-is
AZURE_AI_ENDPOINT=https://your-resource.openai.azure.com/openai/v1/
AZURE_AI_API_KEY=your_key_here
```

## Usage

Edit `prompt.txt` with the problem you want the agents to discuss, then run:

```bash
uv run main.py
```

Output is streamed to the terminal. Each agent's turn is printed as it arrives, followed by a summary at the end.

## Defining agents

Each agent is a YAML file in `agents/`:

```yaml
name: planner
role: |
  You are a strategic planner who breaks down complex problems into
  clear, actionable steps. Be concise and structured.
model: claude-sonnet-4-6
provider: anthropic # or azure_openai
```

| Field        | Description                                                     |
| ------------ | --------------------------------------------------------------- |
| `name`       | Unique identifier used in turn selectors and history            |
| `role`       | System prompt appended to the shared conversation system prompt |
| `model`      | Model name or Azure deployment name                             |
| `provider`   | `anthropic` or `azure_openai`                                   |
| `max_tokens` | Max tokens per response (default: 4096)                         |

## Policies

### Turn selectors

| Policy                | Description                              |
| --------------------- | ---------------------------------------- |
| `round_robin(*names)` | Cycles through the named agents in order |

### Stop conditions

| Policy                       | Description                                            |
| ---------------------------- | ------------------------------------------------------ |
| `max_turns(n)`               | Stops after `n` agent turns                            |
| `stop_on_keyword(*keywords)` | Stops when a keyword appears in the last agent message |

### Post-processors

| Policy             | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `summarize(agent)` | Sends the full transcript to an agent and asks for a summary |

## Customising `main.py`

```python
conv.run(
    turn_selector=round_robin("planner", "critic"),
    stop_condition=max_turns(6),          # run for more turns
    stream=True,
    post_processors=[summarize(agents["planner"], stream=True)],
)
```

Swap in any combination of turn selectors, stop conditions, and post-processors, or implement your own — each is just a callable with a simple signature defined in `policies.py`.

## Model selection

Models you can use during this session:

| Model Name                     | Deployment Type |
| ------------------------------ | --------------- |
| gpt-4o                         | azure_openai    |
| gpt-4.1-nano                   | azure_openai    |
| mistral-medium-2505            | azure_ai        |
| Llama-4-Scout-17B-16E-Instruct | azure_ai        |
| Mistral-Large-3                | azure_ai        |
