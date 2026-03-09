from rest_framework import serializers
from .models import MLTask

class MLTaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLTask
        fields = '__all__'
        read_only_fields = ['created_by', 'status', 'result', 'created_at', 'updated_at']
