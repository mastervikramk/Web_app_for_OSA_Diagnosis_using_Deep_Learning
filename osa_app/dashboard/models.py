from django.db import models
from django.contrib.auth.models import User


class Patient(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100, default="Unknown")
    age = models.IntegerField()
    sex = models.CharField(max_length=10, choices=[('male', 'Male'), ('female', 'Female')])
    weight = models.FloatField()
    height = models.FloatField()

    def __str__(self):
        return self.first_name
    

class CSVFile(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    csv_file = models.FileField(upload_to='csv_files/')
    sampling_frequency = models.FloatField()
    diagnosis_output = models.CharField(max_length=100, blank=True, null=True)
  