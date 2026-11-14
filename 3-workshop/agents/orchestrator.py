from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from memory import load_human_persona
from utils import debug


def orchestrator(state):
    """
    Select next speaker based on conversation context.
    Manages volley control and updates state accordingly.

    Updates state with:
    - next_speaker: Selected agent ID or "human"
    - volley_msg_left: Decremented counter

    Returns: Updated state
    """

    debug(state)
    volley_left = state.get("volley_msg_left", 0)
    debug(f"Volley messages left: {volley_left}", "ORCHESTRATOR")

    if volley_left <= 0:
        debug("No volleys left, returning to human", "ORCHESTRATOR")
        return {
            "next_speaker": "human",
            "volley_msg_left": 0
        }

    messages = state.get("messages", [])
    human_persona = load_human_persona().strip()
    if not human_persona:
        human_persona = "No established human persona yet."
    debug(f"human_persona: {human_persona}")

    conversation_text = ""
    for msg in messages:
        # Messages are now always dicts
        conversation_text += f"{msg.get('content', '')}\n"

    system_prompt = f"""You are managing a lively conversation at a Singapore kopitiam.

    Available speakers:
    - ah_seng: Uncle Ah Seng, 68yo kopi uncle, speaks Singlish, knows about drinks and weather
    - mei_qi: Young 21yo content creator, social media savvy, knows latest news and trends
    - bala: Ex-statistician turned football tipster, dry humor, analytical
    - dr_tan: Retired 72yo philosophy professor, thoughtful and deep thinker
    - human: The person chatting with the group

    Human persona built from previous conversations:
    {human_persona}

    Select who should speak next based on the conversation and the human persona.

    Consider:
    - Who hasn't spoken recently
    - Who has relevant expertise for the current topic
    - Most importantly, who would add interesting perspective
    - Natural kopitiam banter flow

    Respond with ONLY one speaker ID: ah_seng, mei_qi, bala, dr_tan, or human.
    """

    user_prompt = f"""Recent conversation:
{conversation_text}

Who should speak next to keep this kopitiam conversation lively and respectful of the human's preferences?"""

    debug("Analyzing conversation context...", "ORCHESTRATOR")

    # Call LLM
    try:
        llm = ChatOpenAI(model="gpt-5-nano", temperature=1)
        # llm = ChatOpenAI(model="gpt-5-nano", temperature=1, base_url=os.getenv("https://127.0.0.1:1337/v1"))

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        # Extract speaker from response
        if isinstance(response.content, list):
            selected_speaker = " ".join(str(item) for item in response.content).strip().lower()
        else:
            selected_speaker = str(response.content).strip().lower()
        debug(f"LLM selected: {selected_speaker}", "ORCHESTRATOR")

        # Validate speaker
        valid_speakers = ["ah_seng", "mei_qi", "bala", "dr_tan", "human"]
        if selected_speaker not in valid_speakers:
            # Fallback to round-robin if invalid
            import random
            selected_speaker = random.choice(valid_speakers[:-1])
            debug(f"Invalid speaker, fallback to: {selected_speaker}", "ORCHESTRATOR")

    except Exception as e:
        # Fallback selection if LLM fails
        import random
        valid_speakers = ["ah_seng", "mei_qi", "bala", "dr_tan"]
        selected_speaker = random.choice(valid_speakers)
        debug(f"LLM error, random selection: {selected_speaker}", "ORCHESTRATOR")

    debug(f"Final selection: {selected_speaker} (volley {volley_left} -> {volley_left - 1})", "ORCHESTRATOR")

    # Return only the updates (LangGraph will merge with existing state)
    return {
        "next_speaker": selected_speaker,
        "volley_msg_left": volley_left - 1
    }
