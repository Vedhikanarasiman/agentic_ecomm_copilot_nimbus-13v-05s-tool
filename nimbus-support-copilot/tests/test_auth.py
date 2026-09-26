from httpx import ASGITransport, AsyncClient

from app.core.security import create_access_token, get_current_customer_id
from app.main import app

_TEST_BODY = {
    "session_id": "11111111-1111-1111-1111-111111111111",
    "message": "hello",
    "turn_index": 0,
}


async def test_chat_without_token_is_rejected():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json=_TEST_BODY)
    assert response.status_code == 401


async def test_chat_with_garbage_token_is_rejected():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat", json=_TEST_BODY, headers={"Authorization": "Bearer not-a-real-token"}
        )
    assert response.status_code == 401


def test_validly_issued_token_is_accepted_by_the_auth_dependency():
    # Full end-to-end /chat with a valid token needs real (or overridden)
    # db/LLM/embedding dependencies — that path is already covered by the
    # router graph tests plus the manual testing already done against the
    # real stack. This isolates just the auth check itself: does a token
    # this app issued get accepted by the same app's verification logic.
    token = create_access_token("CUST-0001")
    assert get_current_customer_id(token) == "CUST-0001"
