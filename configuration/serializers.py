from rest_framework import serializers

from configuration.models import Family, Zone


class FamilyParentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Family
        fields = ['id', 'index', 'name', 'is_tracked', 'is_trigger', 'is_abstract', 'is_unique', 'is_zoned']


class FamilySerializer(serializers.ModelSerializer):
    parent_id = serializers.PrimaryKeyRelatedField(queryset=Family.objects.all())
    parent = FamilyParentSerializer()

    class Meta:
        model = Family
        fields = ['id', 'index', 'name', 'is_tracked', 'is_trigger', 'is_abstract', 'parent_id', 'parent']


class ZoneSerializer(serializers.ModelSerializer):
    class Meta:
        model = Zone
        fields = ['id', 'slug', 'name', 'is_trigger', 'is_ignored']
