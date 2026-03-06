from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cache
import time
from typing import Callable, cast
from openai import AzureOpenAI, OpenAI
import anthropic
from anthropic.types import MessageParam

from agent_chat.agents import Agent


@dataclass
class Message:
    speaker: str    # agent name, or "user" for human input
    content: str


@cache
def _azure_openai_client() -> AzureOpenAI:
    """AzureOpenAI client for classic Azure OpenAI GPT deployments."""
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


@cache
def _azure_ai_client() -> OpenAI:
    """Plain OpenAI-compatible client for Azure AI Foundry serverless models (Mistral, Llama, etc.)."""
    return OpenAI(
        base_url=os.environ["AZURE_AI_ENDPOINT"],
        api_key=os.environ["AZURE_AI_API_KEY"],
    )


@cache
def _anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=6)


def _build_history(agent: Agent, history: list[Message]) -> list[dict]:
    """
    Map shared conversation history to the messages list for one agent.

    The agent sees its own past turns as "assistant" and everyone else's
    as "user". Agent names are embedded as [name]: prefixes so the model
    understands who said what — the standard speaker-tag pattern used by
    most multi-agent frameworks (AutoGen, CrewAI, etc.).
    """
    messages = []
    for msg in history:
        if msg.speaker == agent.name:
            messages.append({"role": "assistant", "content": msg.content})
        else:
            prefix = "" if msg.speaker == "user" else f"[{msg.speaker}]: "
            messages.append({"role": "user", "content": f"{prefix}{msg.content}"})
    return messages


def call_agent(
    agent: Agent,
    history: list[Message],
    system: str,
    on_token: Callable[[str], None] | None = None,
) -> str:
    messages = _build_history(agent, history)

    if agent.provider in ("azure_openai", "azure_ai"):
        client = _azure_openai_client() if agent.provider == "azure_openai" else _azure_ai_client()
        chunks: list[str] = []
        stream = client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": system}] + messages,  # type: ignore
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                if on_token:
                    on_token(token)
                chunks.append(token)
        agent_response = "".join(chunks)

    elif agent.provider == "anthropic":
        with _anthropic_client().messages.stream(
            model=agent.model,
            system=system,
            messages=cast(list[MessageParam], messages),
            max_tokens=agent.max_tokens,
        ) as stream:
            for token in stream.text_stream:
                if on_token:
                    on_token(token)
            agent_response = stream.get_final_text()

    else:
        raise ValueError(f"Unknown provider: {agent.provider}")

    time.sleep(3)
    return agent_response



StopCondition = Callable[[list[Message]], bool]
TurnSelector = Callable[[list[Message], dict[str, Agent]], str]
PostProcessor = Callable[[list[Message]], None]


@dataclass
class Conversation:
    agents: dict[str, Agent]
    history: list[Message] = field(default_factory=list)
    system_prompt: str = ""  # prepended to every agent's role

    def user(self, content: str) -> None:
        """Inject a human message into the shared history."""
        self.history.append(Message(speaker="user", content=content))

    def step(self, agent_name: str, on_token: Callable[[str], None] | None = None) -> Message:
        """Let one agent respond to the current history."""
        agent = self.agents[agent_name]
        system = "\n\n".join(filter(None, [self.system_prompt, agent.role]))
        content = call_agent(agent, self.history, system=system, on_token=on_token)
        msg = Message(speaker=agent_name, content=content)
        self.history.append(msg)
        return msg

    def run(
        self,
        turn_selector: TurnSelector,
        stop_condition: StopCondition,
        stream: bool = False,
        post_processors: list[PostProcessor] | None = None,
    ) -> list[Message]:
        """Run until stop_condition is met, using turn_selector to pick each speaker."""
        produced = []
        while not stop_condition(self.history):
            name = turn_selector(self.history, self.agents)
            if stream:
                print(f"\n[{name.upper()}]\n", flush=True)
                msg = self.step(name, on_token=lambda t: print(t, end="", flush=True))
                print()
            else:
                msg = self.step(name)
            produced.append(msg)
        for processor in (post_processors or []):
            processor(self.history)
        return produced
