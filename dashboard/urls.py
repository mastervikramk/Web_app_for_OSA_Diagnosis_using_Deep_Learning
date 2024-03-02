from django.urls import path
from . import views

urlpatterns=[
    path('dashboard',views.dashboard,name='dashboard'),
    path('new_patient',views.new_patient,name='new_patient'),
    path('old_patient', views.old_patient, name='old_patient'),
    path('patient_profile',views.patient_profile,name='patient_profile'),
 
]