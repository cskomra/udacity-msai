from agent import Agent
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    agent = Agent(role="Coding Assistant")
    response = agent.invoke("What is Python? Be concise.")
    print(response)