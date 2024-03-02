from django.shortcuts import render,redirect
from .forms import PatientForm
from .models import Patient,CSVFile
from .forms import PatientForm, CSVFileForm
from django.contrib.auth.decorators import login_required  # Import the login_required decorator

# Create your views here.
def dashboard(request):
    return render(request,"dashboard/dashboard.html")


@login_required  # Protect the view so that only logged-in users can access it
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
            
            return redirect('patient_profile')  # Redirect to a success page
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



def patient_profile(request):
    return render(request,"dashboard/patient_profile.html")