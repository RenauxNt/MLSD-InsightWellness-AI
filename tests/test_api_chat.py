from insightwellness_ai.api import app as app_module


def test_chat_rejects_missing_question():
    client = app_module.app.test_client()
    response = client.post("/chat", json={})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_chat_rejects_empty_body():
    client = app_module.app.test_client()
    response = client.post("/chat", data="not json", content_type="text/plain")
    assert response.status_code == 400
