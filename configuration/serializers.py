from rest_framework import serializers

from configuration.models import Family, Zone


class FamilyParentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Family
        fields = ['id', 'index', 'name', 'tracked', 'trigger']


class FamilySerializer(serializers.ModelSerializer):
    parent_id = serializers.PrimaryKeyRelatedField(queryset=Family.objects.all())
    parent = FamilyParentSerializer()

    class Meta:
        model = Family
        fields = ['id', 'index', 'name', 'tracked', 'trigger', 'parent_id', 'parent']


class ZoneSerializer(serializers.ModelSerializer):
    class Meta:
        model = Zone
        fields = ['id', 'slug', 'name', 'trigger']
