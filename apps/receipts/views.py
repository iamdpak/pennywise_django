import uuid
from datetime import datetime
from pathlib import Path

import boto3
from botocore.client import Config
from django.conf import settings
from django.db import transaction
from django.utils.crypto import get_random_string
from django.urls import reverse
from rest_framework import parsers, status, viewsets, mixins
from rest_framework.viewsets import ReadOnlyModelViewSet
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Job, Receipt, Category, ReceiptItem
from .serializers import JobSerializer, ReceiptSerializer, ConfirmReceiptSerializer
from .tasks import process_receipt_job

  
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
