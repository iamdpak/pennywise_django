import logging

from celery import shared_task
from django.utils import timezone
from django.db import transaction
from .models import Job, Receipt, Merchant, Category, ReceiptItem
from .services.llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)

@shared_task
def process_receipt_job(job_id: int, image_uri: str):
    job = Job.objects.get(id=job_id)
    job.status = Job.RUNNING; job.started_at = timezone.now()
    job.save(update_fields=["status","started_at"])
    try:
        adapter = LLMAdapter(); 
        result = adapter.parse_receipt(image_uri)
        logger.info(
            "Parsed receipt job=%s uuid=%s total=%s currency=%s items=%s",
            job.id,
            result.uuid,
            result.total,
            result.currency,
            len(result.items),
        )
        logger.debug("Parsed receipt job=%s raw_json=%s", job.id, result.raw_json)
        with transaction.atomic():
            merchant, _ = Merchant.objects.get_or_create(
                name=result.merchant.get("name","Unknown"),
                defaults={"abn":result.merchant.get("abn",""),"address":result.merchant.get("address",""),
                          "normalized_name":result.merchant.get("name","").lower()},
            )
            category = None
            if result.category: 
                category, _ = Category.objects.get_or_create(name=result.category)
            receipt = Receipt.objects.create(
                uuid=result.uuid,total=result.total,currency=result.currency,purchased_at=result.purchased_at,
                merchant=merchant,category=category,image_uri=image_uri,raw_json=result.raw_json,
            )
            for item in result.items:
                ReceiptItem.objects.create(
                    receipt=receipt,line_text=item.get("line_text",""),quantity=item.get("quantity"),
                    unit_price=item.get("unit_price"),amount=item.get("amount"),
                )
        job.receipt = receipt; job.status = Job.SUCCEEDED; job.finished_at = timezone.now()
        job.save(update_fields=["receipt","status","finished_at"])
    except Exception as e:
        job.status = Job.FAILED; job.error = str(e); job.finished_at = timezone.now()
        job.save(update_fields=["status","error","finished_at"]); raise
