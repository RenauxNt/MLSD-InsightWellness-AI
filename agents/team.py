from agno.agent import Agent
from agno.team import Team
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools

from model_tool import predict_obesity_risk
from rag_agent import search_knowledge_base


YOUR_PROJECT_ID = "mlsd-487610"

ml_agent = Agent(
    name="Obesity Predictor",
    model=Gemini(
        id="gemini-2.5-flash",
        vertexai=True,
        project_id=YOUR_PROJECT_ID,
        location="europe-west1",
    ),
    tools=[predict_obesity_risk],
    instructions="""
You are an obesity prediction model.

RULES:
- Return ONLY the predicted class label
- DO NOT explain anything
- DO NOT interpret results
- DO NOT provide health advice
""",
)

rag_agent = Agent(
    name="Health Knowledge Expert",
    model=Gemini(
        id="gemini-2.5-flash",
        vertexai=True,
        project_id=YOUR_PROJECT_ID,
        location="europe-west1",
    ),
    tools=[search_knowledge_base],
    instructions="""
You are a scientific health researcher.

STRICT RULES:
- You MUST use the knowledge base tool for every answer
- NEVER answer from memory
- NEVER guess or hallucinate
- ALWAYS cite retrieved sources explicitly

Focus on:
- obesity risk factors
- dataset explanation
- lifestyle interpretation based on data
""",
)

web_agent = Agent(
    name="Web Researcher",
    model=Gemini(
        id="gemini-2.5-flash",
        vertexai=True,
        project_id=YOUR_PROJECT_ID,
        location="europe-west1",
    ),
    tools=[DuckDuckGoTools()],
    instructions="""
You are a web research assistant.

RULES:
- Use web search ONLY if:
  - information is not in dataset
  - question is about recent events
  - external comparisons are needed

Do NOT repeat information already provided by ML or RAG agents.
Keep responses short and factual.
""",
)

team = Team(
    name="InsightWellness Team",
    mode="coordinate",
    model=Gemini(
        id="gemini-2.5-flash",
        vertexai=True,
        project_id=YOUR_PROJECT_ID,
        location="europe-west1",
    ),
    members=[ml_agent, rag_agent, web_agent],
    instructions="""
You are a strict orchestrator.

MANDATORY TOOL USAGE:

1. If user provides lifestyle data:
   → MUST call Obesity Predictor

2. AFTER prediction:
   → MUST call Health Knowledge Expert

3. Health Knowledge Expert:
   → MUST explain using dataset OR BMI markdown OR general obesity knowledge

4. ALWAYS call Web Researcher IF:
   - explanation is incomplete
   - dataset has no direct answer
   - or prediction confidence is high but explanation is weak

5. NEVER skip a tool if it could add information

OUTPUT FORMAT:
- Prediction
- Explanation (RAG)
- External context (Web if used)
""",
    markdown=True,
)
