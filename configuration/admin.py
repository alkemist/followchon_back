from django.contrib import admin
from django.contrib.admin import AdminSite

from configuration.models import Zone, Family


class CustomAdminSite(AdminSite):
    admin.site.site_title = 'Followchon'
    admin.site.site_header = 'Followchon'
    admin.site.index_title = 'Administration'


@admin.register(Family)
class FamilyAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'index']
    list_display_links = ['name']
    search_fields = ['index', 'name']
    ordering = ['index']


@admin.register(Zone)
class ZoneAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'slug']
    list_display_links = ['name']
    search_fields = ['slug', 'name']
    ordering = ['slug']


admin_site = CustomAdminSite()
