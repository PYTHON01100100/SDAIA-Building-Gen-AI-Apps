from src.agent.observable_agent import ObservableAgent
from src.tools.registry import registry


def create_researcher(model: str = None, max_steps: int = 15):
    """
    The Researcher: finds, retrieves, and extracts information.

    This function implements the Factory Pattern, returning a configured ObservableAgent
    specialized for research tasks.
    """
    system_prompt = (
        "You are a world-class researcher. Your job is to find accurate, "
        "relevant information using the tools available to you. "
        "Search the web, read webpages, and gather comprehensive data. "
        "Always cite your sources. Be thorough but concise in your findings."
    )
    research_tools = registry.get_tools_by_category("research")
    return ObservableAgent(
        model=model,
        max_steps=max_steps,
        agent_name="Researcher",
        system_prompt=system_prompt,
        tools=research_tools,
    )


def create_analyst(model: str = None, max_steps: int = 20):
    """
    The Analyst: evaluates, cross-references, and identifies patterns.
    """
    system_prompt = (
        "You are an expert analyst. Your job is to evaluate information, "
        "identify patterns, cross-reference data, and provide structured analysis. "
        "Look for contradictions, biases, and gaps. "
        "Present your analysis in a clear, structured format with key findings."
    )
    return ObservableAgent(
        model=model,
        max_steps=max_steps,
        agent_name="Analyst",
        system_prompt=system_prompt,
        tools=[],
    )


def create_writer(model: str = None, max_steps: int = 4):
    """
    The Writer: synthesizes analysis into polished, readable output.
    """
    system_prompt = (
        "You are a professional writer. Your job is to take research and analysis "
        "and synthesize it into a polished, well-structured, and readable report. "
        "Use clear headings, concise language, and ensure the output is engaging. "
        "Include key findings, conclusions, and recommendations."
    )
    return ObservableAgent(
        model=model,
        max_steps=max_steps,
        agent_name="Writer",
        system_prompt=system_prompt,
        tools=[],
    )
