from django.shortcuts import render,redirect,get_object_or_404
from .forms import PatientForm
from .models import Patient,CSVFile
from .forms import PatientForm, CSVFileForm
from django.contrib.auth.decorators import login_required  # Import the login_required decorator

# Create your views here.
def dashboard(request):
    return render(request,"dashboard/dashboard.html")


from django.urls import reverse

@login_required
def new_patient(request):
    if request.method == 'POST':
        patient_form = PatientForm(request.POST)
        csv_file_form = CSVFileForm(request.POST, request.FILES)
        if patient_form.is_valid() and csv_file_form.is_valid():
            # Set the user field to the currently logged-in user
            patient = patient_form.save(commit=False)
            patient.user = request.user  # Set the user field
            patient.save()
            
            # Save the CSVFile object with the patient reference
            csv_file = csv_file_form.save(commit=False)
            csv_file.patient = patient
            csv_file.save()
            
            # Redirect to the new patient profile with patient ID as an argument
            return redirect('new_patient_profile', patient_id=patient.id)
    else:
        patient_form = PatientForm()
        csv_file_form = CSVFileForm()
    return render(request, "dashboard/new_patient.html", {'patient_form': patient_form, 'csv_file_form': csv_file_form})



def old_patient(request):
    search_query = request.GET.get('search_query')
    if search_query:
        patients = Patient.objects.filter(
            user=request.user,
            first_name__icontains=search_query
        )
    else:
        patients = Patient.objects.filter(user=request.user)
    return render(request, 'dashboard/old_patient.html', {'patients': patients})


def old_patient_profile(request, patient_id):
    patient = Patient.objects.get(id=patient_id)
    return render(request, 'dashboard/old_patient_profile.html', {'patient': patient})

def new_patient_profile(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id)
    return render(request, 'dashboard/new_patient_profile.html', {'patient': patient})