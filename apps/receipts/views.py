import uuid
from datetime import datetime
from pathlib import Path

import boto3
from botocore.client import Config
import numpy as np
from django.conf import settings
from django.db import transaction
from django.utils.crypto import get_random_string
from django.utils.dateparse import parse_date, parse_datetime
from django.urls import reverse
from rest_framework import parsers, status, viewsets, mixins
from rest_framework.viewsets import ReadOnlyModelViewSet
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Job, Receipt, Category, ReceiptItem, PurchaseEmbedding
from .serializers import JobSerializer, ReceiptSerializer, ConfirmReceiptSerializer
from .tasks import process_receipt_job
from .services.embedding import embed_texts

  
class ReceiptViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.DestroyModelMixin, viewsets.GenericViewSet):
    queryset = Receipt.objects.all().order_by("-created_at")
    serializer_class = ReceiptSerializer

class JobViewSet(ReadOnlyModelViewSet):
    queryset = Job.objects.all().order_by("-created_at")
    serializer_class = JobSerializer

#---------------------------------------------------------------

class HealthView(APIView):
    def get(self, request): 
        return Response({"status":"ok"})


def _enqueue_job(image_uri: str, idem: str):
        with transaction.atomic():
            job, created = Job.objects.get_or_create(idempotency_key=idem)
            if not created:
                return job, status.HTTP_200_OK
            process_receipt_job.delay(job.id, image_uri)
            #job.refresh_from_db()  # ensure we return the latest status set inside the task
            return job, status.HTTP_202_ACCEPTED


class IngestReceiptView(APIView):
    def post(self, request):
        image_uri = request.data.get("image_uri")
        idem = request.headers.get("Idempotency-Key") or get_random_string(24)
        if not image_uri:
            return Response({"detail":"image_uri required"}, status=status.HTTP_400_BAD_REQUEST)
        job, code = _enqueue_job(image_uri, idem)
        payload = {
            "job_id": job.id,
            "status": job.status,
            "poll_url": request.build_absolute_uri(reverse("job-detail", args=[job.id])),
        }
        return Response(payload, status=code)


class UploadAndIngestView(APIView):
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]

    def post(self, request):
        image_file = request.FILES.get("image")
        idem = request.headers.get("Idempotency-Key") or get_random_string(24)
        if not image_file:
            return Response({"detail": "image file required"}, status=status.HTTP_400_BAD_REQUEST)

        s3 = boto3.client(
            "s3",
            endpoint_url=settings.AWS_S3_ENDPOINT_URL,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_S3_REGION_NAME,
            config=Config(signature_version=settings.AWS_S3_SIGNATURE_VERSION),
        )

        ext = Path(image_file.name).suffix or ".jpg"
        prefix = datetime.utcnow().strftime("%Y/%m/%d")
        key = f"{prefix}/{uuid.uuid4()}{ext}"
        extra = {"ContentType": image_file.content_type or "application/octet-stream"}

        try:
            s3.upload_fileobj(image_file.file, settings.AWS_STORAGE_BUCKET_NAME, key, ExtraArgs=extra)
            image_uri = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.AWS_STORAGE_BUCKET_NAME, "Key": key},
                ExpiresIn=3600,
            )
        except Exception as exc:
            return Response({"detail": f"upload failed: {exc}"}, status=status.HTTP_502_BAD_GATEWAY)

        job, code = _enqueue_job(image_uri, idem)
        payload = {
            "job_id": job.id,
            "status": job.status,
            "poll_url": request.build_absolute_uri(reverse("job-detail", args=[job.id])),
        }
        return Response(payload, status=code)


class PendingJobsView(APIView):
    """
    Lightweight endpoint to poll pending/running jobs without fetching receipt data.
    """

    def get(self, request):
        qs = Job.objects.filter(status__in=[Job.PENDING, Job.RUNNING]).order_by("-created_at")
        data = [{"id": j.id, "status": j.status, "receipt": j.receipt_id} for j in qs]
        return Response({"jobs": data}, status=status.HTTP_200_OK)


class ConfirmReceiptView(APIView):
    """
    Allow client to edit extracted receipt metadata and persist the changes.
    """

    def patch(self, request, receipt_id: int):
        try:
            receipt = Receipt.objects.get(id=receipt_id)
        except Receipt.DoesNotExist:
            return Response({"detail": "receipt not found"}, status=status.HTTP_404_NOT_FOUND)

        serializer = ConfirmReceiptSerializer(data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        with transaction.atomic():
            if "total" in data:
                receipt.total = data["total"]
            if "currency" in data:
                receipt.currency = data["currency"]
            if "purchased_at" in data:
                receipt.purchased_at = data["purchased_at"]

            if "merchant" in data:
                m = data["merchant"]
                receipt.merchant.name = m.get("name", receipt.merchant.name)
                receipt.merchant.abn = m.get("abn", receipt.merchant.abn)
                receipt.merchant.address = m.get("address", receipt.merchant.address)
                receipt.merchant.save()

            if "category" in data:
                cat_name = data["category"]
                if cat_name:
                    category, _ = Category.objects.get_or_create(name=cat_name)
                    receipt.category = category
                else:
                    receipt.category = None

            if "items" in data:
                receipt.items.all().delete()
                for item in data["items"]:
                    ReceiptItem.objects.create(
                        receipt=receipt,
                        line_text=item.get("line_text", ""),
                        quantity=item.get("quantity"),
                        unit_price=item.get("unit_price"),
                        amount=item.get("amount"),
                    )

            receipt.save()

        return Response(ReceiptSerializer(receipt).data, status=status.HTTP_200_OK)


class PurchaseSearchView(APIView):
    """
    Semantic search over purchases using per-purchase embeddings.
    Query params:
      - q: query text (required)
      - start: ISO date/datetime inclusive lower bound
      - end: ISO date/datetime inclusive upper bound
      - vendor: merchant name (optional; case-insensitive)
      - k: number of results (optional; default 50)
    """

    def get(self, request):
        query = (request.query_params.get("q") or "").strip()
        if not query:
            return Response({"detail": "q required"}, status=status.HTTP_400_BAD_REQUEST)

        start = request.query_params.get("start")
        end = request.query_params.get("end")
        vendor = (request.query_params.get("vendor") or "").strip()
        try:
            k = int(request.query_params.get("k", "50"))
        except ValueError:
            k = 50
        k = max(1, min(k, 200))

        start_dt = _parse_iso(start)
        end_dt = _parse_iso(end)

        # Embed query
        vector_list = embed_texts([query])
        if not vector_list or not vector_list[0]:
            return Response({"results": []}, status=status.HTTP_200_OK)
        query_vec = np.array(vector_list[0], dtype=np.float32)
        q_norm = np.linalg.norm(query_vec) or 1.0

        qs = PurchaseEmbedding.objects.select_related(
            "purchase__merchant",
            "purchase__receipt_item",
            "purchase__receipt_item__receipt",
            "purchase__cluster",
        )
        if start_dt:
            qs = qs.filter(purchase__purchased_at__gte=start_dt)
        if end_dt:
            qs = qs.filter(purchase__purchased_at__lte=end_dt)
        if vendor:
            qs = qs.filter(purchase__merchant__name__iexact=vendor)

        results = []
        for emb in qs.iterator():
            if not emb.vector:
                continue
            vec = np.array(emb.vector, dtype=np.float32)
            denom = (np.linalg.norm(vec) or 1.0) * q_norm
            score = float(np.dot(query_vec, vec) / denom)
            results.append((score, emb))

        results.sort(key=lambda x: x[0], reverse=True)
        payload = []
        for score, emb in results[:k]:
            purchase = emb.purchase
            payload.append({
                "purchase_id": purchase.id,
                "receipt_id": purchase.receipt_item.receipt_id,
                "receipt_item_id": purchase.receipt_item_id,
                "merchant": purchase.merchant.name,
                "purchased_at": purchase.purchased_at,
                "price_per_unit": purchase.price_per_unit,
                "currency": purchase.currency,
                "cluster_id": purchase.cluster_id,
                "confidence": purchase.confidence,
                "line_text": purchase.receipt_item.line_text,
                "normalized_text": purchase.normalized_text,
                "score": score,
            })

        return Response({"results": payload}, status=status.HTTP_200_OK)


def _parse_iso(value: str | None):
    if not value:
        return None
    dt = parse_datetime(value)
    if dt:
        return dt
    d = parse_date(value)
    if d:
        return datetime.combine(d, datetime.min.time())
    return None
