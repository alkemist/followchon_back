from django.contrib import admin
from django.contrib import messages
from django.utils.translation import ngettext

from .models import Capture
from .models import Detection


class DetectionInline(admin.TabularInline):
    model = Detection
    extra = 3


class CaptureAdmin(admin.ModelAdmin):
    fieldsets = [
        ('Identification',
         {'fields': ['date', 'photo_file', 'image_tag', 'verified'], 'classes': []}),
    ]
    list_display = ['date', 'verified', 'image_tag']
    search_fields = ['date']
    ordering = ['-date']
    list_filter = ['date', 'verified']
    readonly_fields = ['image_tag']
    actions = ['mark_as_verified', 'mark_as_draft']
    inlines = [
        DetectionInline,
    ]

    @admin.action(description="Mark selected detections as verified")
    def mark_as_verified(self, request, queryset):
        updated = queryset.update(verified=True)

        for item in queryset.iterator():
            item.move_directory(True)

        self.message_user(
            request,
            ngettext(
                "%d detections was successfully marked as draft.",
                "%d detections were successfully marked as draft.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )

    @admin.action(description="Mark selected detections as draft")
    def mark_as_draft(self, request, queryset):
        updated = queryset.update(verified=False)

        for item in queryset.iterator():
            item.move_directory(False)

        self.message_user(
            request,
            ngettext(
                "%d detections was successfully marked as draft.",
                "%d detections were successfully marked as draft.",
                updated,
            )
            % updated,
            messages.SUCCESS,
        )


class DetectionAdmin(admin.ModelAdmin):
    fieldsets = [
        ('Identification',
         {'fields': ['capture', 'family', 'score'], 'classes': []}),
        ('Position', {'fields': ['zone', 'center_x', 'center_y', 'width', 'height'], 'classes': ['inline']}),
    ]
    list_display = ['id', 'capture', 'family', 'zone', 'score',
                    'center_x', 'center_y', 'width', 'height']
    search_fields = ['capture', 'family', 'zone']
    ordering = ['-id']
    list_filter = ['capture', 'family', 'zone']


admin.site.register(Capture, CaptureAdmin)
admin.site.register(Detection, DetectionAdmin)
