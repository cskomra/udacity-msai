from typing import List
from pydantic import BaseModel, Field

# Create Pydantic model for action items
class ActionItem(BaseModel):
    task: str = Field(..., description="Task description")
    assignee: str = Field(..., description="Person responsible for the task")
    due_date: str = Field(..., description="Due date for the task")

# Create Pydantic model for meeting summaries
class MeetingSummary(BaseModel):
    title: str = Field(..., description="Title of the meeting")
    date: str = Field(..., description="Date of the meeting")
    participants: List[str] = Field(..., description="List of participants in the meeting")
    key_points: List[str] = Field(..., description="List of key points discussed in the meeting")
    action_items: List[ActionItem] = Field(..., description="List of action items from the meeting")
