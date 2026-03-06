from dotenv import load_dotenv
load_dotenv()

import sys
from datetime import datetime
from agent_chat.agents import load_agents
from agent_chat.conversation import Conversation
from agent_chat.policies import max_turns, round_robin, summarize


class _Tee:
    """Write to both the real stdout and a log file."""
    def __init__(self, file):
        self._file = file
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


def main():
    log_path = f"logs/conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_path, "w") as log_file:
        sys.stdout = _Tee(log_file)
        _run()
        sys.stdout = sys.stdout._stdout  # restore


def _run():
    agents = load_agents("agents")

    print("--- CONFIG ---")
    for agent in agents.values():
        print(f"{agent.name}: model={agent.model} provider={agent.provider} max_tokens={agent.max_tokens}")
    print("--- END CONFIG ---\n")

    conv = Conversation(agents=agents)

    with open("prompts/system_prompt.txt") as f:
        conv.system_prompt = f.read().strip()

    with open("prompts/prompt.txt") as f:
        prompt = f.read()
    conv.user(prompt)

    conv.run(
        turn_selector=round_robin("planner", "critic"),
        stop_condition=max_turns(4),
        stream=True,
        post_processors=[summarize(agents["planner"], stream=True, prompt_file="prompts/summarize_prompt.txt")],
    )


if __name__ == "__main__":
    main()

