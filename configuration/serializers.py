from rest_framework import serializers

from configuration.models import Family, Zone


class FamilySerializer(serializers.ModelSerializer):
    class Meta:
        model = Family
        fields = ['id', 'index', 'name']


class ZoneSerializer(serializers.ModelSerializer):
    class Meta:
        model = Zone
        fields = ['id', 'slug', 'name']
