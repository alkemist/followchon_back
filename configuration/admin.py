from configuration.models import Zone, Family
from django.contrib import admin
from django.contrib.admin import AdminSite


class CustomAdminSite(AdminSite):
    admin.site.site_title = 'Followchon'
    admin.site.site_header = 'Followchon'
    admin.site.index_title = 'Administration'


class ZoneAdmin(admin.ModelAdmin):
    list_display = ['name', 'index']
    search_fields = ['index', 'name']
    ordering = ['index']


class FamilyAdmin(admin.ModelAdmin):
    list_display = ['name', 'index']
    search_fields = ['index', 'name']
    ordering = ['index']


admin_site = CustomAdminSite()
admin.site.register(Zone, ZoneAdmin)
admin.site.register(Family, FamilyAdmin)
