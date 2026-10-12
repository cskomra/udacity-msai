from typing import Optional, List, Any
from pydantic import BaseModel, Field

from messages import UserMessage, SystemMessage
from tooling import tool
from llm import LLM
from parsers import PydanticOutputParser, JsonOutputParser

from dotenv import load_dotenv

# Create Agent the supports structured outputs for meeting summaries and action items
class StructuredAgent:
    """An AI Agent that provides structured outputs"""

    def __init__(
        self, 
        role: str = "Meeting Assistant",
        instructions: str = "Help summarize meetings and track action items",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        tools: Optional[List[Any]] = None,
        output_model: Optional[BaseModel] = None
    ):
        """Initialize the agent with its configuration
        
        Args:
            role (str): The role of the agent.
            instructions (str): Instructions for the agent.
            model (str): The language model to use.
            temperature (float): The temperature setting for the model.
            tools (List[Any], optional): List of tools available to the agent.
            output_model (BaseModel, optional): The Pydantic model for structured output.
        """

        # Initialize agent configuration
        self.role = role
        self.instructions = instructions
        self.model = model
        self.temperature = temperature
        self.tools = tools or []
        self.output_model = output_model

        # Load environment variables 
        load_dotenv()

        # Initialize the LLM with tools if provided
        self.llm = LLM(
            model=model,
            temperature=temperature,
            tools=tools
        )

    def invoke(self, user_message: str) -> dict:
        """Process a user message and return a structured response
        
        Args:
            user_message (str): The message from the user.

        Returns:
            The agent's structured response after processing tools if needed as a dictionary.
        """
        messages: List[Any] = [
            SystemMessage(
                content=(
                    f"You're an AI Agent and your role is: {self.role}. "
                    f"Your instructions are: {self.instructions}"
                )
            )
        ]
        # Add user message to conversation
        messages.append(UserMessage(content=user_message))

        # Get AI response with structured format if output_model exists
        ai_message = self.llm.invoke(messages)
        messages.append(ai_message)

        # Return parsed, structured output if output_model is provided
        if self.output_model:
            ai_message = self.llm.invoke(input=messages, response_format=self.output_model)
            parser = JsonOutputParser()
            return parser.parse(ai_message)
        else:
            # Handle unstructured response if no model specified
            ai_message = self.llm.invoke(input=messages)
            return {"response": ai_message.content}