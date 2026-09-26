import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.models import Order

_ORDER_ID_PATTERN = re.compile(r"\bORD-\d{5}\b", re.IGNORECASE)


def extract_order_id(text: str) -> str | None:
    """Regex first, LLM never. An order ID has one fixed format — there is
    no reasoning task here, so using the LLM to extract it would only add
    latency and a new failure mode (hallucinated IDs) for zero benefit."""
    match = _ORDER_ID_PATTERN.search(text)
    return match.group(0).upper() if match else None


async def lookup_order(order_id: str, customer_id: str, db: AsyncSession) -> Order | None:
    # Filtering by customer_id here, not just order_id, is the actual
    # authorization check. A valid JWT only proves *someone* is logged in —
    # it says nothing about whether this order is theirs. Returning None
    # uniformly for "doesn't exist" and "exists but isn't yours" avoids
    # leaking which order IDs are valid to an authenticated-but-unrelated user.
    result = await db.execute(
        select(Order).where(Order.order_id == order_id, Order.customer_id == customer_id)
    )
    return result.scalar_one_or_none()


def order_to_context(order: Order) -> dict:
    """What the LLM is allowed to see and restate — not the raw ORM object,
    so we control exactly what customer data reaches the model/logs."""
    return {
        "order_id": order.order_id,
        "product_name": order.product_name,
        "status": order.status,
        "order_date": order.order_date.isoformat(),
        "ship_date": order.ship_date.isoformat() if order.ship_date else None,
        "delivery_date": order.delivery_date.isoformat() if order.delivery_date else None,
        "tracking_number": order.tracking_number,
        "amount": float(order.amount),
    }
