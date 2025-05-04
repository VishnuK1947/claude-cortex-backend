from .appointment_agent import call_appointment_agent
from .health_data_agent import call_health_data_agent
from .medication_agent import call_medication_agent
from langgraph.graph import add_messages
from langgraph.graph import entrypoint
from langgraph.graph import AIMessage


@entrypoint()
def caresync_workflow(messages):
    messages = add_messages([], messages)
    call_active_agent = call_appointment_agent  # Start with appointment agent

    while True:
        agent_messages = call_active_agent(messages).result()
        messages = add_messages(messages, agent_messages)

        # Check for transfer tool calls
        ai_msg = next(
            (m for m in reversed(agent_messages) if isinstance(m, AIMessage)), None
        )

        if not ai_msg or not ai_msg.tool_calls:
            break

        tool_call = ai_msg.tool_calls[-1]

        # Route to the appropriate agent based on tool call
        if tool_call["name"] == "transfer_to_health_data_agent":
            call_active_agent = call_health_data_agent
        elif tool_call["name"] == "transfer_to_medication_agent":
            call_active_agent = call_medication_agent
        else:
            break

    return messages
