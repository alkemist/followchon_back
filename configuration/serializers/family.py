from rest_framework import serializers

from configuration.models import Family
from configuration.serializers.serializers import FamilyParentSerializer


class FamilySerializer(serializers.ModelSerializer):
    parent_id = serializers.PrimaryKeyRelatedField(queryset=Family.objects.all())
    parent = FamilyParentSerializer()

    class Meta:
        model = Family
        fields = [
            'id', 'index', 'name', 'color', 'parent_id', 'parent',
            'is_tracked', 'is_trigger', 'is_abstract', 'is_unique', 'is_zoned'
        ]
