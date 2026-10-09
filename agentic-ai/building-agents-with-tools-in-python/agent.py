from typing import List, Any, cast
from dotenv import load_dotenv
import json

from messages import UserMessage, SystemMessage, ToolMessage
from tooling import tool
from llm import LLM

class Agent:
    """An AI Agent that can use tools to help answer questions"""
    
    def __init__(
        self,
        role: str = "Personal Assistant",
        instructions: str = "Help users with any questions",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        tools: List[Any] = None
    ):
        """
        Initialize the agent with its configuration and tools

        Args:
            role: The agen's role/persona
            instructions: Basic instructions for the agent
            model: The LLM model to use
            temperature: Creativity parameter (0.0 means more 'deterministic')
            tools: List of tools the agent can use
        """
        self.model = model
        self.role = role
        self.instructions = instructions
        self.tools = tools
        self.temperature = temperature

        # Load environment variables (e.g. API keys)
        load_dotenv()

        # Initialize the LLM with tools if provided
        self.llm = LLM(
            model=model,
            temperature=temperature,
            tools=tools,
        )

    def invoke(self, user_message: str) -> str:
        """
        Process a user message and return a response
        
        Args:
            user_message: The user's input message

        Returns:
            The agent's response after processing tools if needed
        """

        messages = [
            SystemMessage(
                content=(
                    f"You're an AI Agent and your role is '{self.role}'. "
                    f"Your instructions: {self.instructions}"
                )
            ),
            UserMessage(content=user_message),
        ]
        # Add user message to conversation
        # messages.append(UserMessage(content=user_message))

        # Get AI response and add to conversation
        ai_message = self.llm.invoke(messages)
        messages.append(ai_message)

        # Check if tools were required
        while ai_message.tool_calls:
            # Process each tool call
            for call in ai_message.tool_calls:
                # Access tool call data correctly
                function_name = call.function.name
                function_args = json.loads(call.function.arguments)
                tool_call_id = call.id
                # Find the matching tool
                tool = next((t for t in self.tools if t.name == function_name), None)
                if tool:
                    result = tool(**function_args)
                    messages.append(
                        ToolMessage(
                            content=json.dumps(result),
                            tool_call_id=tool_call_id,
                            name=function_name,
                        )
                    )

            # Get final AI response after tool usage and add to conversation
            ai_message = self.llm.invoke(messages)
            messages.append(ai_message)

        for m in messages:
            print(m)  # Print each message in the conversation
        return (cast(str, ai_message.content))  # Return the final AI response content
