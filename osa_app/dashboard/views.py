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


def old_patient_profile(request, patient_id):
    patient = Patient.objects.get(id=patient_id)
    return render(request, 'dashboard/old_patient_profile.html', {'patient': patient})

def new_patient_profile(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id)
    return render(request, 'dashboard/new_patient_profile.html', {'patient': patient})



# Load the machine learning model
MODEL_FILE_PATH = 'my_model.h5'
model = load_model(MODEL_FILE_PATH)

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
    """
    Reshape the input data to match the desired shape.
    
    Parameters:
    - input_data: The input array-like object to be reshaped.
    - num_rows: The number of rows in the reshaped array.
    - num_columns: The number of columns in the reshaped array.
    
    Returns:
    - reshaped_array: The reshaped array.
    """
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
    csv_file = CSVFile.objects.get(patient=patient)

    if request.method == 'POST':
        # Retrieve the CSV file path and sampling frequency
        csv_file_path = csv_file.csv_file.path
        sampling_frequency = csv_file.sampling_frequency

        # Load CSV data
        df = pd.read_csv(csv_file_path)
        
        FS=sampling_frequency
        LOWCUT = 1  # Low cut-off frequency in Hz
        HIGHCUT = 45  # High cut-off frequency in Hz

        # Apply band-pass filter to the signal
        # filtered_data = apply_bandpass_filter(df['signal'].values,FS,LOWCUT,HIGHCUT)

        # Reshape data to match model input shape if needed
        # For example, if the model expects input shape (batch_size, sequence_length, num_features),
        # and your data is 1D, you might need to reshape it to (1, sequence_length, 1)
        # filtered_data = np.expand_dims(filtered_data, axis=0)  # Add batch dimension
        # filtered_data = np.expand_dims(filtered_data, axis=2)  # Add feature dimension

        signal=df.values.flatten()
        filtered_data=signal
        num_columns=60*FS
        # precision = 3
        # filtered_data = np.array([np.round(value, precision) for value in filtered_data])
        num_rows = len(filtered_data) // num_columns  # Determine the number of rows

        # Initialize variables to count the number of one and zero classes
        num_ones = 0
        num_zeros = 0

        # Iterate through each row of the filtered data
        for i in range(10):
            # Calculate start and end indices for the current row
            start_index =int ( i * num_columns)
            end_index = int(start_index + num_columns)

            
            # Get the current row
            row = filtered_data[start_index:end_index]
            
            # Apply band-pass filter to the row
            # filtered_row = apply_bandpass_filter(row, FS, LOWCUT, HIGHCUT)
            # Round the values of the row to the desired precision
            # rounded_row = np.round(row, 3)
            # Reshape the row to match the input shape of your model
            # Reshape the row to match the input shape of your model, handling float values
            reshaped_row = reshape_float_input(row, 1, num_columns)
            
            # Make predictions on the reshaped row
            predictions = model.predict(reshaped_row)
            
            # If your model outputs probabilities, you can get the predicted class by taking the argmax
            predicted_class = np.argmax(predictions)

            # Increment the count of one and zero classes based on the predicted class
            if predicted_class == 1:
                num_ones += 1
            else:
                num_zeros += 1

        ratio=num_ones/(num_ones+num_zeros)
        if ratio>=0.5:
            diagnosis_output='Severe'
        elif ratio>=0.25 and ratio<0.5:
            diagnosis_output='Moderate'
        elif ratio>=0.125 and ratio<0.25:
            diagnosis_output='Mild'
        else:
            diagnosis_output='Normal'


        return render(request, 'dashboard/new_patient_profile.html', {'patient': patient, 'diagnosis_output': diagnosis_output})

    return render(request, 'dashboard/new_patient_profile.html', {'patient': patient})
