from django.contrib import admin
from django.contrib import messages
from django.utils.translation import ngettext
from django_admin_relation_links import AdminChangeLinksMixin

from .models import Capture
from .models import Detection


class DetectionInline(admin.TabularInline):
    model = Detection
    extra = 0


@admin.register(Capture)
class CaptureAdmin(admin.ModelAdmin):
    fieldsets = [
        ('Identification',
         {'fields': ['date', 'photo_file', 'image_tag', 'status'], 'classes': []}),
    ]
    list_display = ['id', 'date', 'status', 'image_tag', 'front_url']
    list_display_links = ['date']
    readonly_fields = ['image_tag', 'status']
    list_editable = []
    search_fields = ['id', 'date', 'photo_file']
    ordering = ['-date']
    list_filter = ['date', 'status']
    actions = ['mark_as_draft', 'mark_as_verified', 'mark_as_archived', 'mark_as_deleted']
    inlines = [
        DetectionInline,
    ]
    list_per_page = 10

    @admin.action(description="Mark selected captures as draft")
    def mark_as_draft(self, request, queryset):
        for item in queryset.iterator():
            item.mark_as(Capture.Statuses.DRAFT, True)

        updated = queryset.update(status=Capture.Statuses.DRAFT)

        self.message_user(
            request,
            ngettext(
                "%d captures was successfully marked as draft.",
                "%d captures were successfully marked as draft.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    @admin.action(description="Mark selected captures as verified")
    def mark_as_verified(self, request, queryset):
        for item in queryset.iterator():
            item.mark_as(Capture.Statuses.VERIFIED, True)

        updated = queryset.update(status=Capture.Statuses.VERIFIED)

        self.message_user(
            request,
            ngettext(
                "%d captures was successfully marked as verified.",
                "%d captures were successfully marked as verified.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    @admin.action(description="Mark selected captures as archived")
    def mark_as_archived(self, request, queryset):
        for item in queryset.iterator():
            item.mark_as(Capture.Statuses.ARCHIVED, True)

        updated = queryset.update(status=Capture.Statuses.ARCHIVED)

        self.message_user(
            request,
            ngettext(
                "%d captures was successfully marked as archived.",
                "%d captures were successfully marked as archived.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    @admin.action(description="Mark selected captures as deleted")
    def mark_as_deleted(self, request, queryset):
        for item in queryset.iterator():
            item.mark_as(Capture.Statuses.DELETED, True)

        updated = queryset.update(status=Capture.Statuses.DELETED)

        self.message_user(
            request,
            ngettext(
                "%d captures was successfully marked as deleted.",
                "%d captures were successfully marked as deleted.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    def delete_queryset(self, request, queryset):
        for item in queryset.iterator():
            item.remove_files()

        queryset.delete()


@admin.register(Detection)
class DetectionAdmin(AdminChangeLinksMixin, admin.ModelAdmin):
    fieldsets = [
        ('Identification',
         {'fields': ['capture', 'family', 'score', 'trigger'], 'classes': []}),
        ('Position', {'fields': ['zone', 'center_x', 'center_y', 'width', 'height'], 'classes': ['inline']}),
    ]
    list_display = ['id', 'capture_id', 'capture_link', 'image_tag', 'family', 'zone', 'score', 'trigger']
    readonly_fields = ['capture']
    list_editable = []
    search_fields = ['capture__id', 'family__id', 'zone__id']
    ordering = ['-capture_id', 'family__index']
    list_filter = ['family', 'zone', 'trigger', 'capture__status', ('score', admin.EmptyFieldListFilter)]
    list_display_links = ['id']
    change_links = ['capture']
    list_per_page = 10
