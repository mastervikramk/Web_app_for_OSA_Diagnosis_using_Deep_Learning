from django.shortcuts import render,redirect,get_object_or_404
from .forms import PatientForm
from .models import Patient,CSVFile
from .forms import PatientForm, CSVFileForm
from django.contrib.auth.decorators import login_required  # Import the login_required decorator

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from keras.models import load_model
from .models import CSVFile

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

from .forms import CSVUploadForm


def old_patient_profile(request, patient_id):
    # print("Inside old_patient_profile view function")  # Debug print statement
    patient = get_object_or_404(Patient, id=patient_id)
    form = CSVUploadForm()
    # print("Form initialized successfully")  # Debug print statement
    return render(request, 'dashboard/old_patient_profile.html', {'patient': patient, 'form': form})


# def new_patient_profile(request, patient_id):
#     patient = get_object_or_404(Patient, id=patient_id)
#     return render(request, 'dashboard/new_patient_profile.html', {'patient': patient})



# # Load the machine learning model
# MODEL_FILE_PATH = 'my_model.h5'
# model = load_model(MODEL_FILE_PATH)
import os
from PIL import Image

MODEL_DIRECTORY = 'models'
def load_model_from_file(model_name):
    model_path = os.path.join(MODEL_DIRECTORY, f"{model_name}.h5")
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        raise FileNotFoundError(f"Model file '{model_name}.h5' not found.")


def butter_bandpass(lowcut, highcut, fs, order=1):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data,FS,LOWCUT,HIGHCUT):
    b, a = butter_bandpass(LOWCUT, HIGHCUT, FS)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def reshape_float_input(input_data, num_rows, num_columns):
    # Convert input data to numpy array
    input_array = np.array(input_data)
    # Convert num_rows and num_columns to integers
    num_rows = int(num_rows)
    num_columns = int(num_columns)
    # Calculate the expected size of the input array
    expected_size = num_rows * num_columns
    
    # Check if the input data can be reshaped to the desired shape
    if input_array.size != expected_size:
        raise ValueError(f"Input data size {input_array.size} does not match the expected size {expected_size}")
    
    # Reshape the input array, handling float values
    reshaped_array = input_array.reshape(num_rows, num_columns).astype(float)
    
    return reshaped_array

def new_patient_profile(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id)
    try:
        csv_file = CSVFile.objects.filter(patient=patient).first()
    except CSVFile.DoesNotExist:
        csv_file = None
    # Load the selected model if available
    model = None
    model_name = request.POST.get('model', None)
    if model_name:
        try:
            model = load_model_from_file(model_name)
            # Model loaded successfully
        except FileNotFoundError as e:
            print(e)
            # Handle the case where the model file is not found

    if request.method == 'POST':
        if model:
            # Retrieve the CSV file path and sampling frequency
            csv_file_path = csv_file.csv_file.path
            sampling_frequency = csv_file.sampling_frequency

            # Load CSV data
            df = pd.read_csv(csv_file_path)
            
            FS = sampling_frequency
            LOWCUT = 1  # Low cut-off frequency in Hz
            HIGHCUT = 45  # High cut-off frequency in Hz

            signal = df.values.flatten()
            filtered_data = signal
            num_columns = 60 * FS

            num_ones = 0
            num_zeros = 0

            # Iterate through each row of the filtered data
            for i in range(420):
                start_index = int(i * num_columns)
                end_index = int(start_index + num_columns)
                
                row = filtered_data[start_index:end_index]
                
                reshaped_row = reshape_float_input(row, 1, num_columns)
                
                predictions = model.predict(reshaped_row)
                
                predicted_class = 1 if predictions > 0.5 else 0

                if predicted_class == 1:
                    num_ones += 1
                else:
                    num_zeros += 1

            ratio = num_ones / (num_ones + num_zeros)
            if ratio >= 0.5:
                diagnosis_output = 'Severe'
            elif ratio >= 0.25 and ratio < 0.5:
                diagnosis_output = 'Moderate'
            elif ratio >= 0.125 and ratio < 0.25:
                diagnosis_output = 'Mild'
            else:
                diagnosis_output = 'Normal'
            
            # Save diagnosis output to CSVFile model
            csv_file.diagnosis_output = diagnosis_output
            csv_file.save()
            
            return render(request, 'dashboard/new_patient_profile.html', {'patient': patient, 'diagnosis_output': diagnosis_output, 'apneac_events': num_ones, 'total_events': 420, 'model': model_name})

    return render(request, 'dashboard/new_patient_profile.html', {'patient': patient, 'model': model_name})

from django.shortcuts import redirect, get_object_or_404
from django.contrib import messages
from .forms import CSVUploadForm
from .models import Patient, CSVFile

def upload_csv(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id)
    
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = form.cleaned_data['csv_file']
            sampling_frequency = form.cleaned_data['sampling_frequency']
            
            try:
                # Save the CSV file to the CSVFile model
                new_csv_file = CSVFile(patient=patient, csv_file=csv_file, sampling_frequency=sampling_frequency)
                new_csv_file.save()
                
                # Redirect to the new patient profile page
                return redirect('new_patient_profile', patient_id=patient_id)
            except Exception as e:
                messages.error(request, f"Error saving CSV file: {e}")
        else:
            messages.error(request, "Invalid form data. Please check the provided information.")
    
    # If the request method is not POST or if form is not valid, render the form again
    else:
        form = CSVUploadForm()
    
    return render(request, 'dashboard/new_patient_profile.html', {'form': form, 'patient': patient})
