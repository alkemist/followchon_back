from django.contrib import admin
from django.contrib.admin import AdminSite

from configuration.models import Zone, Family, Parameter, Log


class CustomAdminSite(AdminSite):
    admin.site.site_title = 'Pichon'
    admin.site.site_header = 'Pichon'
    admin.site.index_title = 'Administration'


@admin.register(Family)
class FamilyAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'index', 'parent', 'color', 'is_tracked', 'is_trigger', 'is_abstract', 'is_unique',
                    'is_zoned', 'is_listed']
    list_display_links = ['name']
    search_fields = ['index', 'name']
    ordering = ['index']
    list_editable = ['is_tracked', 'is_trigger', 'is_abstract', 'is_unique', 'is_zoned', 'is_listed']


@admin.register(Zone)
class ZoneAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'slug', 'is_trigger', 'is_ignored', 'is_indoor', 'is_enabled']
    list_display_links = ['name']
    search_fields = ['slug', 'name']
    ordering = ['id']
    list_editable = ['is_trigger', 'is_ignored', 'is_indoor', 'is_enabled']


@admin.register(Parameter)
class ParameterAdmin(admin.ModelAdmin):
    fields = ['slug', 'name', 'value']
    list_display = ['name', 'slug', 'value']
    search_fields = ['name', 'slug', 'value']
    ordering = ['slug']
    list_display_links = ['name', 'slug']
    list_editable = ['value']


@admin.register(Log)
class LogAdmin(admin.ModelAdmin):
    readonly_fields = ['date', 'source']
    list_display = ['date', 'level_color', 'event', 'info', 'source']
    search_fields = ['date', 'level', 'event', 'info', 'source']
    list_filter = ['date', 'level', 'event', 'source']
    ordering = ['-id']
    list_display_links = ['date']
    list_editable = []
    list_per_page = 100


admin_site = CustomAdminSite()
