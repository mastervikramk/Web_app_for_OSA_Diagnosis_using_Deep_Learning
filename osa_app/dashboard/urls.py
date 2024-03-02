from django.urls import path
from . import views

urlpatterns=[
    path('dashboard',views.dashboard,name='dashboard'),
    path('new_patient',views.new_patient,name='new_patient'),
    path('old_patient', views.old_patient, name='old_patient'),
    path('patient/<int:patient_id>/old/', views.old_patient_profile, name='old_patient_profile'),
    path('patient/<int:patient_id>/new/', views.new_patient_profile, name='new_patient_profile'),

 
]