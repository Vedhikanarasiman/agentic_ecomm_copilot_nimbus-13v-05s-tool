"""
Generates ~200 synthetic orders for Nimbus Electronics.
Run: python data/seed_orders.py > data/seed_orders.sql

Distribution choices are deliberate, not random noise:
- status skews toward 'delivered' (most realistic for a support system —
  most queries are about past orders, not fresh ones)
- amounts are category-appropriate (a laptop isn't $15)
- a small slice of 'returned'/'cancelled' orders exist on purpose, since
  those are exactly the ones that will trigger policy questions
"""

import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

CATEGORIES = {
    "laptops":        (450, 2200),
    "smartphones":    (300, 1400),
    "audio":          (25, 400),
    "wearables":      (60, 600),
    "accessories":    (10, 120),
    "gaming":         (200, 900),
    "cameras":        (150, 1800),
    "tv-and-monitors":(180, 1600),
}

PRODUCT_NAMES = {
    "laptops": ["Nimbus Air 14", "Nimbus Pro 16", "Nimbus Book Go"],
    "smartphones": ["Nimbus Phone S", "Nimbus Phone S Pro", "Nimbus Phone Lite"],
    "audio": ["Nimbus Buds 2", "Nimbus Headphones ANC", "Nimbus Speaker Mini"],
    "wearables": ["Nimbus Watch SE", "Nimbus Watch Pro", "Nimbus Band"],
    "accessories": ["Nimbus Fast Charger 65W", "Nimbus USB-C Cable 2m", "Nimbus Laptop Sleeve"],
    "gaming": ["Nimbus Controller X", "Nimbus Gaming Headset", "Nimbus Mechanical Keyboard"],
    "cameras": ["Nimbus Cam One", "Nimbus Action Cam", "Nimbus Cam One Pro"],
    "tv-and-monitors": ["Nimbus Monitor 27\" QHD", "Nimbus TV 55\" 4K", "Nimbus Monitor 34\" Ultrawide"],
}

STATUS_WEIGHTS = {
    "delivered": 0.55,
    "shipped": 0.15,
    "processing": 0.10,
    "returned": 0.12,
    "cancelled": 0.08,
}

PAYMENT_METHODS = ["credit_card", "debit_card", "upi", "net_banking", "wallet"]


def weighted_status():
    return random.choices(
        list(STATUS_WEIGHTS.keys()), weights=list(STATUS_WEIGHTS.values())
    )[0]


def escape(s: str) -> str:
    return s.replace("'", "''")


def generate_order(i: int):
    order_id = f"ORD-{i:05d}"
    customer_id = f"CUST-{random.randint(1, 140):04d}"  # some customers have multiple orders
    name = fake.name()
    email = fake.email()
    category = random.choice(list(CATEGORIES.keys()))
    product = random.choice(PRODUCT_NAMES[category])
    low, high = CATEGORIES[category]
    amount = round(random.uniform(low, high), 2)
    payment = random.choice(PAYMENT_METHODS)
    status = weighted_status()

    order_date = fake.date_time_between(start_date="-180d", end_date="-1d")
    ship_date = None
    if status in ("shipped", "delivered", "returned"):
        ship_date = order_date + timedelta(days=random.randint(1, 3))
    delivery_date = None
    if status in ("delivered", "returned") and ship_date:
        delivery_date = ship_date + timedelta(days=random.randint(2, 6))

    tracking = f"NMB{random.randint(10**9, 10**10 - 1)}" if ship_date else None
    address = fake.address().replace("\n", ", ")

    return {
        "order_id": order_id,
        "customer_id": customer_id,
        "customer_name": name,
        "email": email,
        "product_name": product,
        "category": category,
        "amount": amount,
        "payment_method": payment,
        "status": status,
        "order_date": order_date,
        "ship_date": ship_date,
        "delivery_date": delivery_date,
        "tracking_number": tracking,
        "shipping_address": address,
    }


def to_sql(o: dict) -> str:
    def val(v):
        if v is None:
            return "NULL"
        if isinstance(v, datetime):
            return f"'{v.isoformat()}'"
        if isinstance(v, (int, float)):
            return str(v)
        return f"'{escape(str(v))}'"

    cols = [
        "order_id", "customer_id", "customer_name", "email", "product_name",
        "category", "amount", "payment_method", "status", "order_date",
        "ship_date", "delivery_date", "tracking_number", "shipping_address",
    ]
    values = ", ".join(val(o[c]) for c in cols)
    return f"INSERT INTO orders ({', '.join(cols)}) VALUES ({values});"


if __name__ == "__main__":
    print("-- Synthetic order data for Nimbus Electronics (generated, seed=42)")
    for i in range(1, 201):
        print(to_sql(generate_order(i)))
