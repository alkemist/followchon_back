from django.contrib import admin
from django.contrib.admin import AdminSite

from configuration.models import Zone, Family


class CustomAdminSite(AdminSite):
    admin.site.site_title = 'Followchon'
    admin.site.site_header = 'Followchon'
    admin.site.index_title = 'Administration'


@admin.register(Family)
class FamilyAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'index', 'parent', 'tracked', 'trigger']
    list_display_links = ['name']
    search_fields = ['index', 'name']
    ordering = ['index']
    list_editable = ['tracked', 'trigger']


@admin.register(Zone)
class ZoneAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'slug', 'trigger']
    list_display_links = ['name']
    search_fields = ['slug', 'name']
    ordering = ['id']
    list_editable = ['trigger']


admin_site = CustomAdminSite()
