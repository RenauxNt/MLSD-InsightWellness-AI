from agno.agent import Agent
from agno.team import Team
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools

from model_tool import explain_obesity_risk, predict_obesity_risk
from rag_agent import search_knowledge_base


PROJECT_ID = "mlsd-487610"

ml_agent = Agent(
    name="Obesity Predictor",
    model=Gemini(
        id="gemini-2.5-flash",
        vertexai=True,
        project_id=PROJECT_ID,
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

explain_agent = Agent(
    name="Explainability Expert",
    model=Gemini(
        id="gemini-2.5-flash",
        vertexai=True,
        project_id=PROJECT_ID,
        location="europe-west1",
    ),
    tools=[explain_obesity_risk],
    instructions="""
You are a SHAP explainability expert.

RULES:
- ALWAYS call explain_obesity_risk for every request
- Return the tool output verbatim
- DO NOT invent feature importances
- DO NOT add health advice or interpretation beyond what the tool returns
""",
)

rag_agent = Agent(
    name="Health Knowledge Expert",
    model=Gemini(
        id="gemini-2.5-flash",
        vertexai=True,
        project_id=PROJECT_ID,
        location="europe-west1",
    ),
    tools=[search_knowledge_base],
    instructions="""
You are a scientific health researcher.

RULES:
- You MUST query the knowledge base first for every answer
- When the knowledge base covers the question, cite the retrieved sources explicitly
- When it does not, you MAY rely on general medical knowledge, but MUST flag it
  as "general information" and not as a dataset/model fact
- NEVER invent specific numbers, dataset facts, or model behavior — those must
  come from the knowledge base
- ALWAYS append this disclaimer at the end of your response:
  "This is general information, not personalized medical advice. Consult a
  healthcare professional for individual guidance."

Focus on:
- obesity risk factors
- dataset explanation
- lifestyle interpretation based on data
- general health recommendations tailored to the predicted class
""",
)

web_agent = Agent(
    name="Web Researcher",
    model=Gemini(
        id="gemini-2.5-flash",
        vertexai=True,
        project_id=PROJECT_ID,
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
        project_id=PROJECT_ID,
        location="europe-west1",
    ),
    members=[ml_agent, explain_agent, rag_agent, web_agent],
    instructions="""
You are a strict orchestrator.

MANDATORY TOOL USAGE:

1. If user provides lifestyle data:
   → MUST call Obesity Predictor

2. If the user asks WHY, asks for an EXPLANATION, asks which factors drive
   the prediction, or asks about FEATURE IMPORTANCE:
   → MUST call Explainability Expert

3. AFTER prediction:
   → MUST call Health Knowledge Expert

4. Health Knowledge Expert:
   → MUST explain using dataset OR BMI markdown OR general obesity knowledge

5. ALWAYS call Web Researcher IF:
   - explanation is incomplete
   - dataset has no direct answer
   - or prediction confidence is high but explanation is weak

6. NEVER skip a tool if it could add information

OUTPUT FORMAT:
- Prediction
- SHAP drivers (only if Explainability Expert was called)
- Explanation (RAG)
- External context (Web if used)
""",
    markdown=True,
)
