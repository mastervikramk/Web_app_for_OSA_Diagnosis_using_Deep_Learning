# forms.py

from django import forms
from .models import Patient
from .models import CSVFile

class PatientForm(forms.ModelForm):
    class Meta:
        model = Patient
        exclude = ['user']  # Exclude the 'user' field from the form


class CSVFileForm(forms.ModelForm):
    class Meta:
        model = CSVFile
        fields = ['csv_file', 'sampling_frequency']


class CSVUploadForm(forms.ModelForm):
    class Meta:
        model = CSVFile
        fields = ['csv_file', 'sampling_frequency']