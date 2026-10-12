from typing import cast
from structured_agent import StructuredAgent
from models import MeetingSummary
from pydantic import BaseModel
import json

# Create an agent instance with the MeetingSummary model
meeting_agent = StructuredAgent(
    role="Meeting Assistant",
    instructions="Summarize meetings and track action items in a structured format",
    output_model=cast(BaseModel, MeetingSummary)
)

meeting_transcript = """
Project Planning Meeting - March 15, 2024

Attendees: John, Sarah, Mike

Discussion:
- Reviewed Q1 project timeline
- Discussed resource allocation
- Identified potential risks

Next steps:
1. John will update the project plan by next Fiday
2. Sarah needs to coordinate with the design team by Wednesday
3. Mike will prepare the risk assessment document by the end of month
"""

summary = meeting_agent.invoke(meeting_transcript)
print(json.dumps(summary, indent=2))

# Create a MeetingSummary instance from the output
validated_summary = MeetingSummary(**summary)

# Access structured data
print("Meeting Title:", validated_summary.title)
print("\nParticipants:")
for participant in validated_summary.participants:
    print(f"- {participant}")

print("\nAction Items:")
for item in validated_summary.action_items:
    print(f"- {item.task} (Assigned to: {item.assignee}, Due: {item.due_date})")