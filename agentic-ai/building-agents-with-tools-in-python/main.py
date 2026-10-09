from tooling import tool
from typing import List
from agent import Agent
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    # agent = Agent(role="Coding Assistant")
    # response = agent.invoke("What is Python? Be concise.")
    # print(response)

    ### TOOLS ###

    @tool
    def calculate(expression: str) -> float:
        """Evaluate a mathematical expression."""
        return eval(expression)

    @tool
    def get_games(num_games:int=1, top:bool=True) -> List[dict]:
        """
        Returns the top or bottom N games with highest or lowest scores.
        args:
            num_games (int): Number of games to return (default is 1)
            top (bool): If True, return top games, otherwise return bottom (default is True)
        """
        data = [
            {"Game": "The Legend of Zelda: Breath of the Wild", "Platform": "Switch", "Score": 98},
            {"Game": "Super Mario Odyssey", "Platform": "Switch","Score": 97},
            {"Game": "Metroid Prime", "Platform": "GameCube", "Score": 97},
            {"Game": "Super Smash Bros. Brawl", "Platform": "Wii", "Score": 93},
            {"Game": "Mario Kart 8 Deluxe", "Platform": "Switch", "Score": 92},
            {"Game": "Fire Emblem: Awakening", "Platform": "3DS", "Score": 92},
            {"Game": "Donkey Kong Country Returns", "Platform": "Wii", "Score": 87},
            {"Game": "Luigi's Mansion 3", "Platform": "Switch", "Score": 86},
            {"Game": "Pikmin 3", "Platform": "Wii U", "Score": 85},
            {"Game": "Animal Crossing: New Leaf", "Platform": "3DS", "Score": 88}
        ]
        # Sort the games list by Score
        # If top is True, descending order
        sorted_games = sorted(data, key=lambda x: x["Score"], reverse=top)

        # Return the N games
        return sorted_games[:num_games]

    ##########

    coder_agent = Agent(role="Coding Assistant")

    math_agent = Agent(
        role="Math Assistant",
        tools = [calculate]
    )

    data_analyst_agent = Agent(
        role="Game Stats Assistant",
        instructions="You can bring insights about a game dataset based on users questions",
        tools = [get_games]
    )

    
    response = data_analyst_agent.invoke("What's the worst game in the dataset?")
    print(response)
