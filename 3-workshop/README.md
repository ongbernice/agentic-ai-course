# Singapore Kopitiam Multi-Agent Chat

## Overview

This project simulates a conversation at a Singapore kopitiam. A LangGraph
orchestrator coordinates four distinct regulars—Uncle Ah Seng, Mei Qi, Bala,
and Dr. Tan—and invites the human to contribute as the discussion unfolds.
When the human types `exit`, a summarizer closes the conversation.

The orchestrator uses a locally saved human persona to help decide who should
speak next. The persona is updated from conversation history and records only
low-risk interests and conversational preferences. The full history and
persona are stored as plain text in `human_memory.txt` beside the application.
That file is excluded from Git. Delete it to reset the saved persona and
history; do not share it if it contains personal information.

## Architecture

- **Orchestrator:** selects the next speaker using the current conversation
  and saved human persona, while tracking the number of agent turns.
- **Participants:** four distinct personas with different voices and
  specializations.
- **Tools:** Singapore time, weather, and news. Each persona is given a
  different list of available actions in its prompt.
- **Human persona memory:** persists conversation messages and updates a
  concise persona between turns.
- **Summarizer:** produces a closing summary when the human exits.

The graph is defined in `main.py`; state is defined in `state.py`. Participant
behavior is in `agents/participant.py`, and persona/history persistence is in
`memory.py`.

## Requirements

- Python 3.12 or later
- [`uv`](https://docs.astral.sh/uv/)
- An OpenAI API key with access to `gpt-5-mini` and `gpt-5-nano`
- Internet access for the OpenAI API and live weather/news tools

## Setup and run

From this directory:

```sh
uv sync
cp .env.example .env
```

Edit `.env` and replace the example value with your key:

```dotenv
OPENAI_API_KEY=your_real_api_key
DEBUG=false
```

Start the chat:

```sh
uv run python main.py
```

Type a message to join the conversation. Type `exit` to see the final summary.
Set `DEBUG=true` in `.env` to show orchestration decisions and participant
tool actions.
