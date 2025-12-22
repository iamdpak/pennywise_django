from rest_framework import serializers
from .models import Receipt, Merchant, Category, ReceiptItem, Job
class MerchantSerializer(serializers.ModelSerializer):
    class Meta: model = Merchant; fields = ("id","name","abn","address","normalized_name")
class CategorySerializer(serializers.ModelSerializer):
    class Meta: model = Category; fields = ("id","name","parent")
class ReceiptItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReceiptItem
        fields = ("id","line_text","quantity","unit_price","amount","unit_type","pack_size")
class ReceiptSerializer(serializers.ModelSerializer):
    merchant = MerchantSerializer(); category = CategorySerializer(allow_null=True)
    items = ReceiptItemSerializer(many=True, required=False)
    class Meta:
        model = Receipt
        fields = ("id","uuid","total","currency","purchased_at","merchant","category","image_uri","raw_json","items","created_at","updated_at")


class ConfirmReceiptItemSerializer(serializers.Serializer):
    line_text = serializers.CharField(required=False, allow_blank=True, default="")
    quantity = serializers.FloatField(required=False, allow_null=True)
    unit_price = serializers.FloatField(required=False, allow_null=True)
    amount = serializers.FloatField(required=False, allow_null=True)
    unit_type = serializers.CharField(required=False, allow_blank=True, default="")
    pack_size = serializers.FloatField(required=False, allow_null=True)


class ConfirmReceiptSerializer(serializers.Serializer):
    total = serializers.FloatField(required=False)
    currency = serializers.CharField(required=False, max_length=8)
    purchased_at = serializers.DateTimeField(required=False, allow_null=True)
    merchant = MerchantSerializer(required=False)
    category = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    items = ConfirmReceiptItemSerializer(many=True, required=False)
class JobSerializer(serializers.ModelSerializer):
    class Meta: model = Job; fields = ("id","idempotency_key","receipt","status","error","started_at","finished_at","created_at")
