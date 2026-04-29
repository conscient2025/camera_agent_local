# Local Camera Recommendation Agent

A minimal TAO-style camera recommendation agent built for learning AI agents.

## What it does

- uses a **local camera knowledge base** only
- follows a **Thought → Action → Observation** loop
- lets the LLM:
  - understand the user's abstract needs
  - ask follow-up questions when needed
  - search and compare cameras from the local database

## Key rules

- **Budget means body-only price** by default
- the agent can only recommend cameras that already exist in the local JSON database
- first-pass filtering is based **only on budget**, with a small tolerance above budget

## Files

- `camera_agent_local_kb_v6_3.py` — main program
- `local_camera_kb_beginner_v2.json` — local camera database
- `.env` — API configuration

## Setup

Install dependencies:

```bash
pip install openai python-dotenv
```

Create a `.env` file:

```env
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=your_base_url
MODEL_NAME=your_model
```

## Run

```bash
python camera_agent_local_kb_v6_3.py
```

Sample Input:

```bash
我预算15000元，主要拍旅行、风景、人像，注重视频体验，请帮我推荐一台相机。
我预算6000元，主要拍旅行和风景，基本不拍视频，想要容易上手一点，请帮我推荐一台相机。
```

## Main tools

- `understand_user_need()`
- `ask_user_clarification()`
- `search_cameras_by_need()`
- `get_camera_details()`
- `compare_cameras()`

## Notes

This project is designed for **agent-learning and experimentation**, not for production use.
