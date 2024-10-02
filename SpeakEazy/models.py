
from django.db import models

class VideoFile(models.Model):
    video_file = models.FileField(upload_to='videos/')
