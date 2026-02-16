import asyncio
import sys

from dotenv import load_dotenv

# Import tools so they register themselves with the registry
import src.tools.search_tool  # noqa: F401

from src.agent.specialists import create_researcher, create_analyst, create_writer

# Load environment variables
load_dotenv()


async def main():
    """
    Main entry point for the AI Agent system.
    Runs a linear chain: Researcher -> Analyst -> Writer.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m src.main \"Your research query\"")
        sys.exit(1)

    query = sys.argv[1]
    print(f"\n🔍 Starting research on: {query}\n")

    # 1. Research phase
    researcher = create_researcher()
    print("=" * 50)
    print("PHASE 1: RESEARCH")
    print("=" * 50)
    research_result = await researcher.run(query)
    research_text = research_result["answer"]
    print(f"\n📋 Research complete ({len(research_text)} chars)\n")

    # 2. Analysis phase
    analyst = create_analyst()
    print("=" * 50)
    print("PHASE 2: ANALYSIS")
    print("=" * 50)
    analysis_prompt = (
        f"Analyze the following research findings about '{query}':\n\n"
        f"{research_text}\n\n"
        "Identify key patterns, contradictions, and insights."
    )
    analysis_result = await analyst.run(analysis_prompt)
    analysis_text = analysis_result["answer"]
    print(f"\n📊 Analysis complete ({len(analysis_text)} chars)\n")

    # 3. Writing phase
    writer = create_writer()
    print("=" * 50)
    print("PHASE 3: WRITING")
    print("=" * 50)
    writing_prompt = (
        f"Write a comprehensive report about '{query}' based on:\n\n"
        f"Research:\n{research_text}\n\n"
        f"Analysis:\n{analysis_text}\n\n"
        "Create a polished, well-structured report."
    )
    writing_result = await writer.run(writing_prompt)
    final_report = writing_result["answer"]

    # Print final report
    print("\n" + "=" * 50)
    print("FINAL REPORT")
    print("=" * 50)
    print(final_report)

    # Print cost breakdown
    print("\n" + "=" * 50)
    print("COST BREAKDOWN")
    print("=" * 50)
    researcher.cost_tracker.print_cost_breakdown()
    analyst.cost_tracker.print_cost_breakdown()
    writer.cost_tracker.print_cost_breakdown()


if __name__ == "__main__":
    asyncio.run(main())
