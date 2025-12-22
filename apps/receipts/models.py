from django.db import models
from django.contrib.postgres.fields import ArrayField

class Merchant(models.Model):
    name = models.CharField(max_length=255)
    abn = models.CharField(max_length=32, blank=True, default="")
    address = models.TextField(blank=True, default="")
    normalized_name = models.CharField(max_length=255, blank=True, default="")
    def __str__(self): return self.name


class Category(models.Model):
    name = models.CharField(max_length=120)
    parent = models.ForeignKey("self", null=True, blank=True, on_delete=models.SET_NULL)
    def __str__(self): return self.name


class Receipt(models.Model):
    uuid = models.CharField(max_length=64, unique=True)
    total = models.DecimalField(max_digits=10, decimal_places=2)
    currency = models.CharField(max_length=8, default="AUD")
    purchased_at = models.DateTimeField(null=True, blank=True)
    merchant = models.ForeignKey(Merchant, on_delete=models.PROTECT)
    category = models.ForeignKey(Category, null=True, blank=True, on_delete=models.SET_NULL)
    image_uri = models.TextField()
    raw_json = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class ReceiptItem(models.Model):
    receipt = models.ForeignKey(Receipt, related_name="items", on_delete=models.CASCADE)
    line_text = models.TextField()
    quantity = models.FloatField(null=True, blank=True)
    unit_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    amount = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)

    # Parsed attributes to assist association; optional and can be backfilled later.
    brand = models.CharField(max_length=255, blank=True, default="")
    variety = models.CharField(max_length=255, blank=True, default="")
    form = models.CharField(max_length=255, blank=True, default="")
    unit_type = models.CharField(max_length=32, blank=True, default="")
    pack_size = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True)
    pack_size_unit = models.CharField(max_length=32, blank=True, default="")


class Job(models.Model):
    PENDING, RUNNING, SUCCEEDED, FAILED = "PENDING","RUNNING","SUCCEEDED","FAILED"
    STATUSES = [(s, s) for s in (PENDING, RUNNING, SUCCEEDED, FAILED)]
    idempotency_key = models.CharField(max_length=128, unique=True)
    receipt = models.ForeignKey(Receipt, null=True, blank=True, on_delete=models.SET_NULL)
    status = models.CharField(max_length=16, choices=STATUSES, default=PENDING)
    error = models.TextField(blank=True, default="")
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class Purchase(models.Model):
    """Link a parsed receipt line to a canonical product with normalized price for querying."""

    receipt_item = models.OneToOneField(ReceiptItem, related_name="purchase", on_delete=models.CASCADE)
    currency = models.CharField(max_length=8, default="AUD")
    price_per_unit = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    cluster = models.ForeignKey("PurchaseCluster", null=True, blank=True, on_delete=models.SET_NULL)
    normalized_text = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["cluster"]),
        ]

    def __str__(self):
        return f"Purchase {self.id}"


class PurchaseCluster(models.Model):
    """
    Stable grouping of similar purchases without requiring a curated product catalog.
    """

    label = models.CharField(max_length=255, blank=True, default="")
    notes = models.TextField(blank=True, default="")
    centroid = ArrayField(models.FloatField(), null=True, blank=True)  # optional representative vector
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.label or f"Cluster {self.id}"


class PurchaseEmbedding(models.Model):
    """
    Embedding for a specific purchase to support semantic search over purchases.
    """

    purchase = models.ForeignKey(Purchase, related_name="embeddings", on_delete=models.CASCADE)
    model_name = models.CharField(max_length=255)
    vector = ArrayField(models.FloatField(), null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["model_name"])]
