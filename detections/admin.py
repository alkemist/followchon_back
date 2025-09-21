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
         {'fields': ['date', 'photo_file', 'image_tag', 'status', 'errors', 'version_detect', 'version_classify',
                     'changed',
                     'front_url',
                     ], 'classes': []}),
    ]
    list_display = ['id', 'date', 'status', 'changed', 'errors', 'version_detect', 'version_classify', 'image_tag',
                    'front_url']
    list_display_links = ['date']
    readonly_fields = ['image_tag', 'status', 'errors', 'source', 'front_url']
    list_editable = []
    search_fields = ['id', 'date', 'photo_file']
    ordering = ['-date']
    list_filter = ['date', 'status', 'changed', 'version_detect', 'version_classify', 'train_all', 'train_chons',
                   'source', 'errors']
    actions = ['resize', 'mark_as_draft', 'mark_as_verified', 'mark_as_archived', 'mark_as_waiting', 'mark_as_deleted',
               'migrate']
    inlines = [
        DetectionInline,
    ]
    list_per_page = 10

    @admin.action(description="Resize image")
    def resize(self, request, queryset):
        for item in queryset.iterator():
            item.resize_auto()

        updated = len(queryset)

        self.message_user(
            request,
            ngettext(
                "%d capture resized.",
                "%d captures resized.",
                updated,
            )
            % updated,
            messages.SUCCESS if updated > 0 else messages.ERROR,
        )

    @admin.action(description="Migrate")
    def migrate(self, request, queryset):
        for item in queryset.iterator():
            item.calc_errors()

        updated = len(queryset)

        self.message_user(
            request,
            ngettext(
                "%d checked.",
                "%d checked.",
                updated,
            )
            % updated,
            messages.SUCCESS if updated > 0 else messages.ERROR,
        )

    @admin.action(description="Mark selected captures as archived")
    def mark_as_archived(self, request, queryset):
        for item in queryset.iterator():
            if item.status == Capture.Statuses.VERIFIED:
                item.mark_as(Capture.Statuses.ARCHIVED, True)

        updated = queryset.filter(status=Capture.Statuses.VERIFIED) \
            .update(status=Capture.Statuses.ARCHIVED)

        self.message_user(
            request,
            ngettext(
                "%d capture was successfully marked as archived.",
                "%d captures were successfully marked as archived.",
                updated,
            )
            % updated,
            messages.SUCCESS if updated > 0 else messages.ERROR,
        )

    @admin.action(description="Mark selected captures as waiting")
    def mark_as_waiting(self, request, queryset):
        for item in queryset.iterator():
            if item.status == Capture.Statuses.DRAFT:
                item.mark_as(Capture.Statuses.WAITING, True)

        updated = queryset.filter(status=Capture.Statuses.DRAFT) \
            .update(status=Capture.Statuses.WAITING)

        self.message_user(
            request,
            ngettext(
                "%d capture was successfully marked as waiting.",
                "%d captures were successfully marked as waiting.",
                updated,
            )
            % updated,
            messages.SUCCESS if updated > 0 else messages.ERROR,
        )

    def delete_queryset(self, request, queryset):
        for item in queryset.iterator():
            if item.status == Capture.Statuses.DELETED:
                item.remove_files()

        deleted = queryset.filter(status=Capture.Statuses.DELETED) \
            .delete()

        self.message_user(
            request,
            ngettext(
                "%d capture was successfully deleted.",
                "%d captures were successfully deleted.",
                deleted[0],
            )
            % deleted[0],
            messages.SUCCESS if deleted[0] > 0 else messages.ERROR,
        )


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
    list_filter = ['capture__date', 'capture__status', 'family', 'zone', 'trigger', 'capture__source',
                   ('score', admin.EmptyFieldListFilter)]
    list_display_links = ['id']
    change_links = ['capture']
    list_per_page = 10
