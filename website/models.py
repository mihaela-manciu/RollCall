
from django.db import models

class AttendanceRecord(models.Model):
    student_name = models.CharField(max_length=100)
    class_name = models.CharField(max_length=10)
    timestamp = models.DateTimeField()

    def __str__(self):
        return f"{self.student_name} - {self.class_name} - {self.timestamp}"

