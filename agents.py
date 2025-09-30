from crewai import Agent
from tools import yt_tool # type: ignore
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"

from crewai.llm import LLM
llm = LLM(api_key=os.environ["OPENAI_API_KEY"], model=os.environ["OPENAI_MODEL_NAME"])

blog_researcher = Agent(
    role='Blog Researcher from YouTube videos',
    goal='Get the relevant video content for the topic {topic} from a YouTube channel.',
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI, Data Science, Machine Learning, and Generative AI, "
        "and providing suggestions."
    ),
    tools=[yt_tool],  
    allow_delegation=True
)

blog_writer = Agent(
    role='Blog Writer',
    goal='Narrate compelling tech stories about the video {topic} from a YouTube channel.',
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft "
        "engaging narratives that captivate and educate, bringing new "
        "discoveries to light in an accessible manner."
    ),
    tools=[yt_tool],
    llm=llm,
    allow_delegation=False
)
