from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    HealthView,
    IngestReceiptView,
    UploadAndIngestView,
    ConfirmReceiptView,
    ReceiptViewSet,
    JobViewSet,
    PendingJobsView,
    PurchaseSearchView,
    PurchaseClusterJobView,
)

router = DefaultRouter()
router.register(r"receipts", ReceiptViewSet, basename="receipt")
router.register(r"jobs", JobViewSet, basename="job")

urlpatterns = [
    path("healthz/", HealthView.as_view()),
    path("receipt/ingest", IngestReceiptView.as_view()),
    path("receipt/upload", UploadAndIngestView.as_view()),
    path("receipt/<int:receipt_id>/confirm", ConfirmReceiptView.as_view()),
    path("jobs/pending", PendingJobsView.as_view(), name="job-pending"),
    path("purchases/search", PurchaseSearchView.as_view(), name="purchase-search"),
    path("purchases/cluster", PurchaseClusterJobView.as_view(), name="purchase-cluster-job"),
    path("", include(router.urls)),
]
