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
import os
from PIL import Image
from .forms import CSVUploadForm
from django.urls import reverse
from django.contrib import messages
from tensorflow.lite.python import interpreter as interpreter_wrapper

# Create your views here.
def dashboard(request):
    return render(request,"dashboard/dashboard.html")

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
    # print("Inside old_patient_profile view function")  # Debug print statement
    patient = get_object_or_404(Patient, id=patient_id)
    form = CSVUploadForm()
    # print("Form initialized successfully")  # Debug print statement
    return render(request, 'dashboard/old_patient_profile.html', {'patient': patient, 'form': form})


MODEL_DIRECTORY = 'models'
def load_model_from_file(model_name):
    model_path = os.path.join(MODEL_DIRECTORY, f"{model_name}.tflite")
    if os.path.exists(model_path):
        return interpreter_wrapper.Interpreter(model_path=model_path)
    else:
        raise FileNotFoundError(f"Model file '{model_name}.tflite' not found.")



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

from scipy import signal

def upsample_ecg(ecg_signal, input_fs, output_fs):
    # Calculate the interpolation factor
    interpolation_factor = int(output_fs / input_fs)
    
    # Upsample the ECG signal
    upsampled_ecg = signal.resample_poly(ecg_signal, interpolation_factor, 1)
    
    return upsampled_ecg

def downsample_ecg(ecg_signal, input_fs, output_fs):
    # Calculate the decimation factor
    decimation_factor = int(input_fs / output_fs)
    
    # Downsample the ECG signal
    downsampled_ecg = signal.decimate(ecg_signal, decimation_factor)
    
    return downsampled_ecg



def reshape_float_input(input_data, num_rows, num_columns):
    # Convert input data to numpy array
    input_array = np.array(input_data)
    # Convert num_rows and num_columns to integers
    num_rows = int(num_rows)
    num_columns = int(num_columns)
    
    # Reshape the input array into a 2D array with shape (num_rows, num_columns)
    reshaped_array = input_array.reshape(num_rows, num_columns).astype(float)
    
    return reshaped_array



def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi >= 18.5 and bmi < 25:
        return "Normal"
    elif bmi >= 25 and bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def new_patient_profile(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id)
    try:
        csv_file = CSVFile.objects.filter(patient=patient).last()
    except CSVFile.DoesNotExist:
        csv_file = None
    
    # Load the selected model if available
    interpreter = None
    model_name = request.POST.get('model', None)
    if model_name:
        try:
            interpreter = load_model_from_file(model_name)
            # Allocate tensors.
            interpreter.allocate_tensors()
        except FileNotFoundError as e:
            print(e)
            # Handle the case where the model file is not found
    
     # Calculate BMI
    height_in_meters = patient.height / 100  # Convert height to meters
    weight = patient.weight
    if height_in_meters > 0:  # Avoid division by zero
        bmi = round(weight / (height_in_meters ** 2), 2)
    else:
        bmi = None

    # Categorize BMI
    bmi_category = categorize_bmi(bmi)

    if request.method == 'POST':
        if interpreter:
            # Retrieve the CSV file path and sampling frequency
            csv_file_path = csv_file.csv_file.path
            sampling_frequency = csv_file.sampling_frequency

            # Load CSV data
            df = pd.read_csv(csv_file_path)
            
            if sampling_frequency < 100:
                # Downsample the ECG signal
                new_ecg = upsample_ecg(df, sampling_frequency, 100)
            
            elif sampling_frequency > 100:
                new_ecg = downsample_ecg(df, sampling_frequency, 100)

            else:
                new_ecg = df

            FS = 100
            LOWCUT = 1  # Low cut-off frequency in Hz
            HIGHCUT = 45  # High cut-off frequency in Hz
            
            ecg_df = pd.DataFrame(new_ecg)
            ecg_signal = ecg_df.values.flatten()
            num_columns = 60 * FS
            total_datapoints = len(ecg_signal)
            num_rows = total_datapoints // num_columns
            num_ones = 0
            num_zeros = 0
            
            # Iterate through each row of the filtered data
            for i in range(num_rows):
                start_index = int(i * num_columns)
                end_index = int(start_index + num_columns)
                
                row = ecg_signal[start_index:end_index]
                new_row = apply_bandpass_filter(data=row, FS=FS, LOWCUT=LOWCUT, HIGHCUT=HIGHCUT)
                reshaped_row = reshape_float_input(new_row, 1, num_columns)
                
                # Convert input data to TensorFlow Lite compatible type (e.g., np.float32)
                input_data = reshaped_row.astype(np.float32)  # Ensure input data is of type np.float32
                # Reshape input data to match the expected shape of the input tensor
                input_data = input_data.reshape((1, 6000, 1))

                # Get input details
                input_details = interpreter.get_input_details()
                # print("Expected Input Shape:", input_details[0]['shape'])

                # Print shape of your input data
                # print("Shape of Input Data:", input_data.shape)
                output_details = interpreter.get_output_details()
                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], input_data)

                # Invoke the interpreter
                interpreter.invoke()
                # Get output tensor
                output_data = interpreter.get_tensor(output_details[0]['index'])
                # print("Output Shape:", output_data.shape)
                # print("Output Data:", output_data)  # Print output data for debugging
                # Perform inference
                predictions = output_data  # Assuming output_data contains predictions
                # print("Predictions:", predictions)  # Print predictions for debugging
                
                predicted_class = 1 if predictions > 0.5 else 0

                if predicted_class == 1:
                    num_ones += 1
                else:
                    num_zeros += 1


            ratio = num_ones / (num_ones + num_zeros)
            if ratio >= 0.5:
                diagnosis_output = 'SEVERE'
            elif ratio >= 0.25 and ratio < 0.5:
                diagnosis_output = 'MODERATE'
            elif ratio >= 0.125 and ratio < 0.25:
                diagnosis_output = 'MILD'
            else:
                diagnosis_output = 'NORMAL'
            
            # Save diagnosis output to CSVFile model
            csv_file.diagnosis_output = diagnosis_output
            csv_file.save()
            
            return render(request, 'dashboard/new_patient_profile.html', {'patient': patient, 'diagnosis_output': diagnosis_output, 'apneac_events': num_ones, 'total_events': num_rows, 'model': model_name})
    return render(request, 'dashboard/new_patient_profile.html', {'patient': patient,'bmi': bmi, 'bmi_category': bmi_category, 'model': model_name})



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

