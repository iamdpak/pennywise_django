from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Sequence

from django.db import transaction
from django.utils import timezone

from apps.receipts.models import (
    Category,
    Merchant,
    Purchase,
    PurchaseEmbedding,
    Receipt,
    ReceiptItem,
)
from apps.receipts.services.embedding import embed_texts


logger = logging.getLogger(__name__)


@dataclass
class ParsedItem:
    line_text: str
    quantity: float | None
    unit_price: float | None
    amount: float | None
    brand: str | None = ""
    variety: str | None = ""
    form: str | None = ""
    unit_type: str | None = ""
    pack_size: float | None = None
    pack_size_unit: str | None = ""
    confidence: float | None = None  # optional from parser


@dataclass
class ParsedReceipt:
    uuid: str
    total: float
    currency: str
    purchased_at: Optional[datetime]
    merchant_name: str
    merchant_abn: str = ""
    merchant_address: str = ""
    merchant_normalized_name: str = ""
    category: str | None = None
    image_uri: str = ""
    raw_json: dict | None = None
    items: Sequence[ParsedItem] = ()


def normalize_text(item: ParsedItem) -> str:
    # Build a deterministic normalized string for embedding.
    # If you use an LLM to rewrite/clean, do it before passing here; this keeps assembly consistent.
    parts: List[str] = []
    for value in (
        item.line_text,
        item.brand,
        item.variety,
        item.form,
        item.unit_type,
        _format_pack_size(item.pack_size, item.pack_size_unit),
    ):
        if value:
            parts.append(str(value).strip().lower())
    normalized = " ".join(parts)
    return " ".join(normalized.split())  # collapse whitespace


def _format_pack_size(size: float | None, unit: str | None) -> str:
    if size is None:
        return ""
    unit = (unit or "").strip().lower()
    try:
        size_str = f"{Decimal(str(size)).normalize()}"
    except InvalidOperation:
        size_str = str(size)
    return f"{size_str}{unit}" if unit else size_str


def _compute_price_per_unit(item: ParsedItem) -> Optional[Decimal]:
    """
    Compute normalized unit price when quantity is known; falls back to unit_price.
    """
    if item.quantity:
        try:
            amount = Decimal(str(item.amount)) if item.amount is not None else Decimal(str(item.unit_price or 0))
            qty = Decimal(str(item.quantity))
            if qty > 0:
                return (amount / qty).quantize(Decimal("0.0001"))
        except (InvalidOperation, ZeroDivisionError):
            return None
    if item.unit_price is not None:
        try:
            return Decimal(str(item.unit_price)).quantize(Decimal("0.0001"))
        except InvalidOperation:
            return None
    return None


@transaction.atomic
def ingest_parsed_receipt(payload: ParsedReceipt, embedding_model: str | None = None) -> Receipt:
    """
    Persist parsed receipt, items, purchases, and embeddings.

    embedding_model is informational; purchase embeddings always use the configured default.
    """
    merchant, _ = Merchant.objects.get_or_create(
        name=payload.merchant_name,
        defaults={
            "abn": payload.merchant_abn,
            "address": payload.merchant_address,
            "normalized_name": payload.merchant_normalized_name or payload.merchant_name.lower(),
        },
    )

    category_obj = None
    if payload.category:
        category_obj, _ = Category.objects.get_or_create(name=payload.category)

    receipt = Receipt.objects.create(
        uuid=payload.uuid,
        total=payload.total,
        currency=payload.currency,
        purchased_at=payload.purchased_at,
        merchant=merchant,
        category=category_obj,
        image_uri=payload.image_uri,
        raw_json=payload.raw_json or {},
    )

    normalized_texts: List[str] = []
    purchases: List[Purchase] = []

    for item in payload.items:
        ri = ReceiptItem.objects.create(
            receipt=receipt,
            line_text=item.line_text,
            quantity=item.quantity,
            unit_price=item.unit_price,
            amount=item.amount,
            brand=item.brand or "",
            variety=item.variety or "",
            form=item.form or "",
            unit_type=item.unit_type or "",
            pack_size=item.pack_size,
            pack_size_unit=item.pack_size_unit or "",
        )

        norm_text = normalize_text(item)
        normalized_texts.append(norm_text)

        purchase = Purchase.objects.create(
            receipt_item=ri,
            merchant=merchant,
            purchased_at=payload.purchased_at or timezone.now(),
            currency=payload.currency,
            price_per_unit=_compute_price_per_unit(item),
            confidence=item.confidence,
            normalized_text=norm_text,
        )
        purchases.append(purchase)

    # Embed and store vectors
    if normalized_texts:
        vectors = embed_texts(normalized_texts)
        for purchase, vector in zip(purchases, vectors):
            if not vector:
                continue
            PurchaseEmbedding.objects.create(
                purchase=purchase,
                model_name=embedding_model or "",
                vector=vector,
            )

    return receipt
