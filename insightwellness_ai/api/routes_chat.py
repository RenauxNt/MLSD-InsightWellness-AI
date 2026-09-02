"""Chat endpoint backed by the multi-agent team."""

from flask import Blueprint, jsonify, request

chat_bp = Blueprint("chat", __name__)

_agent_team = None


def get_agent_team():
    # lazy: importing the team pulls in agno/chromadb, keep startup fast
    global _agent_team
    if _agent_team is None:
        from insightwellness_ai.agents.team import team

        _agent_team = team
    return _agent_team


@chat_bp.route("/chat", methods=["POST"])
def chat():
    """
    Ask the multi-agent team a question about a prediction, the dataset, or healthy habits.
    The team coordinates a predictor agent (calls /predict), a SHAP explainability agent
    (calls /explain), a RAG agent grounded in the project knowledge base, and a web
    research agent.
    ---
    tags:
      - Chat
    parameters:
      - in: body
        name: chat_request
        required: true
        schema:
          type: object
          properties:
            question:
              type: string
              example: Why was I classified as Overweight_Level_I?
    responses:
      200:
        description: Answer from the agent team.
      400:
        description: Missing question.
      500:
        description: Agent team failure.
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        team = get_agent_team()
        response = team.run(question, stream=False)
        answer = response.content if hasattr(response, "content") else str(response)
        return jsonify({"answer": answer}), 200
    except Exception as e:
        return jsonify({"error": f"Chat failed: {str(e)}"}), 500
