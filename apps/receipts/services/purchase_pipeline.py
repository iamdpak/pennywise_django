from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Sequence

from django.db import transaction
from django.utils import timezone

from apps.receipts.models import Category, Merchant, Purchase, PurchaseEmbedding, Receipt, ReceiptItem
from apps.receipts.services.embedding import embed_texts


logger = logging.getLogger(__name__)


@dataclass
class ParsedItem:
    line_text: str
    quantity: float | None
    unit_price: float | None
    amount: float | None
    unit_type: str | None = ""
    pack_size: float | None = None
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
        item.unit_type,
        _format_pack_size(item.pack_size),
    ):
        if value:
            parts.append(str(value).strip().lower())
    normalized = " ".join(parts)
    return " ".join(normalized.split())  # collapse whitespace


def _format_pack_size(size: float | None) -> str:
    if size is None:
        return ""
    try:
        size_str = f"{Decimal(str(size)).normalize()}"
    except InvalidOperation:
        size_str = str(size)
    return size_str


def _compute_price_per_unit(item: ParsedItem) -> Optional[Decimal]:
    """
    Compute normalized price per base unit.
    - Weight/volume: normalize g->kg, ml->L.
    - Each/unknown: return price per quantity as-is.
    """
    UNIT_MULTIPLIER = {
        "kg": Decimal("1"),
        "g": Decimal("0.001"),
        "l": Decimal("1"),
        "ml": Decimal("0.001"),
        "each": Decimal("1"),
    }

    # Pick unit context from explicit unit_type only.
    unit = (item.unit_type or "").strip().lower()
    multiplier = UNIT_MULTIPLIER.get(unit, None)

    # Prefer explicit quantity; fallback to pack_size as quantity hint.
    qty_val = item.quantity
    if qty_val is None and item.pack_size is not None:
        qty_val = item.pack_size

    amount = None
    if item.amount is not None:
        try:
            amount = Decimal(str(item.amount))
        except InvalidOperation:
            amount = None
    if amount is None and item.unit_price is not None and qty_val is not None:
        try:
            amount = Decimal(str(item.unit_price)) * Decimal(str(qty_val))
        except InvalidOperation:
            amount = None

    if amount is None:
        # No amount computed; fall back directly to unit_price if provided (even without quantity)
        if item.unit_price is not None:
            try:
                return Decimal(str(item.unit_price)).quantize(Decimal("0.0001"))
            except InvalidOperation:
                return None
        return None

    if qty_val is None:
        # No quantity: treat the amount as the unit price (best effort)
        try:
            return Decimal(str(amount)).quantize(Decimal("0.0001"))
        except InvalidOperation:
            return None

    try:
        qty = Decimal(str(qty_val))
        if qty <= 0:
            return None
        base_qty = qty * (multiplier if multiplier is not None else Decimal("1"))
        if base_qty <= 0:
            return None
        return (amount / base_qty).quantize(Decimal("0.0001"))
    except (InvalidOperation, ZeroDivisionError):
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
            unit_type=item.unit_type or "",
            pack_size=item.pack_size,
        )

        norm_text = normalize_text(item)
        normalized_texts.append(norm_text)

        purchase = Purchase.objects.create(
            receipt_item=ri,
            currency=payload.currency,
            price_per_unit=_compute_price_per_unit(item),
            confidence=item.confidence,
            normalized_text=norm_text,
            unit_type=item.unit_type or "",
            pack_size=item.pack_size,
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


@transaction.atomic
def build_purchases_for_receipt(
    receipt: Receipt,
    item_overrides: Optional[List[tuple[ReceiptItem, ParsedItem]]] = None,
    embedding_model: str | None = None,
) -> None:
    """
    Create or update Purchase/embeddings for an existing receipt and its items.
    Optional item_overrides: list of (ReceiptItem, ParsedItem) to supply unit_type/pack_size.
    """
    if item_overrides is not None:
        items_with_parsed = item_overrides
    else:
        items_with_parsed = []
        for item in receipt.items.all():
            parsed = ParsedItem(
                line_text=item.line_text,
                quantity=item.quantity,
                unit_price=float(item.unit_price) if item.unit_price is not None else None,
                amount=float(item.amount) if item.amount is not None else None,
                unit_type=item.unit_type,
                pack_size=float(item.pack_size) if item.pack_size is not None else None,
                confidence=None,
            )
            items_with_parsed.append((item, parsed))

    if not items_with_parsed:
        if not items:
            return

    normalized_texts: List[str] = []
    purchases: List[Purchase] = []

    for item, parsed in items_with_parsed:
        norm_text = normalize_text(parsed)
        normalized_texts.append(norm_text)

        purchase, _ = Purchase.objects.get_or_create(
            receipt_item=item,
            defaults={
                "currency": receipt.currency,
                "price_per_unit": _compute_price_per_unit(parsed),
                "confidence": None,
                "normalized_text": norm_text,
                "unit_type": parsed.unit_type or "",
                "pack_size": parsed.pack_size,
            },
        )
        updated_price = _compute_price_per_unit(parsed)
        if (
            purchase.normalized_text != norm_text
            or purchase.price_per_unit != updated_price
            or purchase.unit_type != (parsed.unit_type or "")
            or purchase.pack_size != parsed.pack_size
        ):
            purchase.normalized_text = norm_text
            purchase.price_per_unit = updated_price
            purchase.currency = receipt.currency
            purchase.unit_type = parsed.unit_type or ""
            purchase.pack_size = parsed.pack_size
            purchase.save(
                update_fields=["normalized_text", "price_per_unit", "currency", "unit_type", "pack_size", "updated_at"]
            )
        purchases.append(purchase)

    # Refresh embeddings
    PurchaseEmbedding.objects.filter(purchase__in=purchases).delete()
    vectors = embed_texts(normalized_texts)
    for purchase, vector in zip(purchases, vectors):
        if not vector:
            continue
        PurchaseEmbedding.objects.create(
            purchase=purchase,
            model_name=embedding_model or "",
            vector=vector,
        )
