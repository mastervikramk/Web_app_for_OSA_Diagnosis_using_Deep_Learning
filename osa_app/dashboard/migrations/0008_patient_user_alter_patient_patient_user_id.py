# Generated by Django 5.0.1 on 2024-03-01 04:51

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("dashboard", "0007_remove_patient_csv_file_and_more"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(
            model_name="patient",
            name="user",
            field=models.ForeignKey(
                default=None,
                on_delete=django.db.models.deletion.CASCADE,
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AlterField(
            model_name="patient",
            name="patient_user_id",
            field=models.CharField(blank=True, default=None, max_length=255),
        ),
    ]
