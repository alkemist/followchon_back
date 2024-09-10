from rest_framework import serializers

from configuration.models import Family, Zone, Parameter


class FamilyParentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Family
        fields = [
            'id', 'index', 'name', 'color', 'parent_id',
            'is_tracked', 'is_trigger', 'is_abstract', 'is_unique', 'is_zoned'
        ]


class ZoneSerializer(serializers.ModelSerializer):
    class Meta:
        model = Zone
        fields = ['id', 'slug', 'name', 'is_trigger', 'is_ignored', 'is_indoor', 'is_enabled', 'center_x', 'center_y',
                  'width',
                  'height']


class ParameterSerializer(serializers.ModelSerializer):
    class Meta:
        model = Parameter
        fields = ['id', 'slug', 'name', 'value']
