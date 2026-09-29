"""
agent_run.py — Entry point for the civic engagement agent.

A minimal LangGraph ReAct-style agent with two tools:
  - civic_lookup: pulls someone's ballot from the Google Civic Information API
  - office_responsibilities: searches for what a specific office actually does,
    scoped to its full jurisdiction

Uses Groq's free API tier for the LLM (get a key at console.groq.com —
no card required) running Llama 3.3 70B, which supports tool calling.

Setup:
    pip install -r requirements.txt
    cp .env.example .env   # then fill in your API keys

Run:
    python agent_run.py
"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import create_agent

from tools import civic_lookup, office_responsibilities

load_dotenv()

SYSTEM_PROMPT = """You are a civic engagement assistant. Your job is to help \
everyday people understand what's on their ballot and who actually holds \
power over local issues they care about — so they can show up informed and \
know who to contact. You are a research aid, not an advocate.

Rules you must follow:

1. Never characterize candidates, take political positions, or suggest who \
   someone should vote for or support. Stick to describing offices, races, \
   and processes.

2. When you state what an office is responsible for, use the \
   office_responsibilities tool rather than relying on your own general \
   knowledge. Office names and powers vary a lot by place, and a confident \
   guess that sounds right is exactly the failure mode to avoid here.

3. Always pass the FULL jurisdiction (city, county, and state as available, \
   pulled from civic_lookup's output) to office_responsibilities — never \
   just the office name alone. Many office names ("Board of Commissioners", \
   "City Council") exist in thousands of places.

4. Always cite your source (the URL) when you state what an office does.

5. If office_responsibilities returns no useful results, say plainly that \
   you couldn't verify what this office controls. Do not fill the gap with \
   a guess based on the office's title.

6. Keep answers concrete and actionable: what's on their ballot, what an \
   office controls, and — when relevant — who to contact or when a \
   relevant body meets.
"""


def build_agent():
    model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    tools = [civic_lookup, office_responsibilities]
    return create_agent(model, tools, system_prompt=SYSTEM_PROMPT)


def main():
    missing = [
        name
        for name in ("GROQ_API_KEY", "GOOGLE_CIVIC_API_KEY")
        if not os.environ.get(name)
    ]
    if missing:
        print(f"Warning: missing env vars: {', '.join(missing)}")
        print("Set these in a .env file before running for real.\n")

    agent = build_agent()
    print("Civic agent ready. Ask about your ballot, or a local issue you care about.")
    print("Type 'quit' to exit.\n")

    messages = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": messages})
        messages = result["messages"]

        last = messages[-1]
        print(f"\nAgent: {last.content}\n")


if __name__ == "__main__":
    main()