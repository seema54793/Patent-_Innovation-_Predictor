import os
from datetime import datetime
import requests

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from opensearch_client import get_opensearch_client

# -------------------- Ollama Utilities --------------------

def check_ollama_availability():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model.get("name") for model in models if model.get("name")]
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return []

def test_model(model_name):
    try:
        llm = OllamaLLM(model=model_name, temperature=0.2)
        prompt = ChatPromptTemplate.from_template("Say hello!")
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({})
        return bool(result)
    except Exception as e:
        print(f"Error testing model {model_name}: {e}")
        return False

# -------------------- Custom Tools --------------------

class SearchPatentsTool(BaseTool):
    name: str = "search_patents"
    description: str = "Search for patents matching a query"

    def _run(self, query: str, top_k: int = 20) -> str:
        client = get_opensearch_client("localhost", 9200)
        index_name = "patents"

        search_query = {
            "size": top_k,
            "query": {"bool": {"must": [{"match": {"abstract": query}}]}},
            "_source": ["title", "abstract", "publication_date", "patent_id"],
        }

        try:
            response = client.search(index=index_name, body=search_query)
            results = response["hits"]["hits"]

            formatted = [
                f"{i+1}. Title: {hit['_source'].get('title')[:80]}\n   Date: {hit['_source'].get('publication_date')}\n   Patent ID: {hit['_source'].get('patent_id')}\n   Abstract: {hit['_source'].get('abstract')[:200]}...\n"
                for i, hit in enumerate(results)
            ]
            return "\n".join(formatted)
        except Exception as e:
            return f"Error searching patents: {str(e)}"

class SearchPatentsByDateRangeTool(BaseTool):
    name: str = "search_patents_by_date_range"
    description: str = "Search for patents in a specific date range"

    def _run(self, query: str, start_date: str, end_date: str, top_k: int = 30) -> str:
        client = get_opensearch_client("localhost", 9200)
        index_name = "patents"

        search_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [{"match": {"abstract": query}}],
                    "filter": [
                        {"range": {"publication_date": {"gte": start_date, "lte": end_date}}}
                    ],
                }
            },
            "_source": ["title", "abstract", "publication_date", "patent_id"],
        }

        try:
            response = client.search(index=index_name, body=search_query)
            results = response["hits"]["hits"]

            formatted = [
                f"{i+1}. Title: {hit['_source'].get('title')[:80]}\n   Date: {hit['_source'].get('publication_date')}\n   Patent ID: {hit['_source'].get('patent_id')}\n   Abstract: {hit['_source'].get('abstract')[:200]}...\n"
                for i, hit in enumerate(results)
            ]
            return "\n".join(formatted)
        except Exception as e:
            return f"Error searching patents: {str(e)}"

class AnalyzePatentTrendsTool(BaseTool):
    name: str = "analyze_patent_trends"
    description: str = "Analyze trends in patent data"

    def _run(self, patents_data: str) -> str:
        return f"Analyzing {len(patents_data.splitlines())} lines of patent data:\n{patents_data[:500]}..."

# -------------------- Crew & Agents Setup --------------------

def create_patent_analysis_crew(model_name, research_area):
    available_models = check_ollama_availability()
    if not available_models:
        raise RuntimeError("Ollama service is not available. Make sure Ollama is running.")

    if not test_model(model_name):
        raise RuntimeError(f"Model {model_name} is not responding to test prompts.")

    print("Model found and tested successfully.")

    if not model_name.startswith("ollama/"):
        model_name = f"ollama/{model_name}"

    llm = OllamaLLM(model=model_name, temperature=0.2)

    tools = [SearchPatentsTool(), SearchPatentsByDateRangeTool(), AnalyzePatentTrendsTool()]

    research_director = Agent(
        role="Research Director",
        goal="Coordinate research and define the analysis scope",
        backstory=f"Lead researcher specializing in {research_area}.",
        verbose=True,
        allow_delegation=True,
        llm=llm,
        tools=tools,
    )

    patent_retriever = Agent(
        role="Patent Retriever",
        goal=f"Retrieve top {research_area} patents",
        backstory="Expert in patent search and classification.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
    )

    data_analyst = Agent(
        role="Patent Data Analyst",
        goal="Identify trends and tech patterns",
        backstory=f"Analyzes {research_area} patent data to forecast innovation.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
    )

    innovation_forecaster = Agent(
        role="Innovation Forecaster",
        goal="Predict future technologies and R&D needs",
        backstory="Futurist skilled in analyzing patents to detect breakthroughs.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
    )

    task1 = Task(
        description=f"""
        Define a research plan for {research_area} patent analysis:
        - Key sub-technology areas
        - Analysis period (last 3–5 years)
        - Applications or challenges of interest
        """,
        expected_output="Research plan with focus areas and timeline.",
        agent=research_director,
    )

    task2 = Task(
        description=f"""
        Retrieve {research_area} patents using the plan. Group them by sub-technology.
        - Identify top assignees
        - Summarize key innovations
        """,
        expected_output="List of patents with companies and summaries.",
        agent=patent_retriever,
        dependencies=[task1],
    )

    task3 = Task(
        description=f"""
        Analyze trends in {research_area} patents:
        - Growth vs decline areas
        - Innovation timeline
        - Emerging technologies
        """,
        expected_output="Patent trend analysis with key insights.",
        agent=data_analyst,
        dependencies=[task2],
    )

    task4 = Task(
        description=f"""
        Predict future directions in {research_area}:
        - Technologies to watch in next 2–3 years
        - Suggested R&D investment areas
        """,
        expected_output="Forecast report with innovation roadmap.",
        agent=innovation_forecaster,
        dependencies=[task3],
    )

    crew = Crew(
        agents=[research_director, patent_retriever, data_analyst, innovation_forecaster],
        tasks=[task1, task2, task3, task4],
        verbose=True,
        process=Process.sequential,
        cache=False,
    )

    return crew

# -------------------- Run Entry --------------------

def run_patent_analysis(research_area, model_name):
    try:
        crew = create_patent_analysis_crew(model_name=model_name, research_area=research_area)
        result = crew.kickoff(inputs={"research_area": research_area})

        if hasattr(result, "output"):
            return result.output
        elif hasattr(result, "result"):
            return result.result
        else:
            return str(result)
    except Exception as e:
        return (
            f"Analysis failed: {str(e)}\n\nTroubleshooting tips:\n"
            + "1. Make sure Ollama is running: 'ollama serve'\n"
            + "2. Pull a compatible model: 'ollama pull mistral'\n"
            + "3. Check model name from 'ollama list'\n"
            + "4. Try a simpler model like 'phi' or 'tinyllama'"
        )

if __name__ == "__main__":
    research_area = input("Enter the research area to analyze (default: Lithium Battery): ")
    if not research_area:
        research_area = "Lithium Battery"

    model_name = input("Enter the Ollama model to use (default: llama2): ")
    if not model_name:
        model_name = "llama2"

    result = run_patent_analysis(research_area, model_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"patent_analysis_{timestamp}.txt"

    with open(filename, "w") as f:
        f.write(str(result))

    print(f"Analysis completed and saved to {filename}")
