from typing import Literal
from state import State
from agents import orchestrator, participant, summarizer
from memory import PersonaUpdateError, append_conversation_messages, update_human_memory


def human_node(state: State) -> dict:
    """
    Human input node - gets user input and sets volley count.
    """
    if state.get("next_speaker") == "human":
        print("\nThe group would like your input.")

    user_input = input("\nYou: ").strip()

    human_message = {
        "role": "user",
        "content": f"You: {user_input}"
    }
    volley_left = state.get("volley_msg_left", 0)

    try:
        update_human_memory([human_message])
    except PersonaUpdateError as error:
        print(f"\nPersona update unavailable: {error}")

    return {
        "messages": [human_message],
        "volley_msg_left": volley_left + 2 if volley_left <= 0 else volley_left
    }


def check_exit_condition(state: State) -> Literal["summarizer", "orchestrator"]:
    """
    Check if user typed 'exit' to end conversation.
    """
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        content = last_message.get("content", "")

        if "exit" in content.lower():
            return "summarizer"

    return "orchestrator"


def orchestrator_routing(state: State) -> Literal["participant", "human"]:
    """
    Route according to the speaker selected by the orchestrator.
    """
    if state.get("next_speaker") in {"ah_seng", "mei_qi", "bala", "dr_tan"}:
        return "participant"

    return "human"


def participant_node(state: State) -> dict:
    """
    Participant node - calls the appropriate participant and handles output.
    """
    next_speaker = state.get("next_speaker", "ah_seng")  # Default fallback

    # Call participant with the selected speaker
    result = participant(next_speaker, state)

    # Print and return messages
    if result and "messages" in result:
        for msg in result["messages"]:
            print(msg.get("content", ""))
        append_conversation_messages(result["messages"])
        return {"messages": result["messages"]}

    return {}


def summarizer_node(state: State) -> dict:
    """
    Summarizer node - generates and displays conversation summary.
    """
    print("\n=== CONVERSATION ENDING ===\n")

    # Generate and print summary
    summary = summarizer(state)
    print(summary)
    print("\nThank you! Come back to kopitiam anytime lah!")

    return {}  # Empty update to end
