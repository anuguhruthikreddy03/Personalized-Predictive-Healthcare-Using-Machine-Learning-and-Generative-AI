from django.shortcuts import render, HttpResponse
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
from django.conf import settings

import seaborn as sns
from django.core.files.storage import FileSystemStorage
from django.db import IntegrityError
from django.utils import timezone

def UserRegisterActions(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        loginid = request.POST.get('loginid')
        password = request.POST.get('password')
        mobile = request.POST.get('mobile')
        email = request.POST.get('email')
        locality = request.POST.get('locality')
        status = request.POST.get('status', 'waiting')  # default to 'waiting'

        try:
            # Create user manually
            user = UserRegistrationModel.objects.create(
                name=name,
                loginid=loginid,
                password=password,
                mobile=mobile,
                email=email,
                locality=locality,
                status=status,
                date_joined=timezone.now()
            )
            user.save()
            messages.success(request, '✅ You have been successfully registered.')

        except IntegrityError as e:
            if 'email' in str(e).lower():
                messages.error(request, '❌ Email already exists.')
            elif 'mobile' in str(e).lower():
                messages.error(request, '❌ Mobile number already exists.')
            elif 'loginid' in str(e).lower():
                messages.error(request, '❌ Login ID already exists.')
            else:
                messages.error(request, f'❌ Registration failed: {str(e)}')

    return render(request, 'UserRegistrations.html')


from django.contrib import messages
from django.shortcuts import render, redirect
from .models import UserRegistrationModel

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get("loginid")
        password = request.POST.get("pswd")
        print("Login ID:", loginid)
        print("Password:", password)

        try:
            user = UserRegistrationModel.objects.get(loginid=loginid, password=password)
            status = user.status.lower()

            if status == "activated":
                # Set session variables
                request.session['id'] = user.id
                request.session['loginid'] = user.loginid
                request.session['password'] = user.password
                request.session['email'] = user.email
                return render(request, 'users/UserHome.html')

            elif status == "waiting":
                messages.warning(request, "⚠️ Your account is waiting for admin approval.")
            elif status == "blocked":
                messages.error(request, "🚫 Your account has been blocked by the admin.")
            else:
                messages.info(request, f"Account status: {status}")

        except UserRegistrationModel.DoesNotExist:
            messages.error(request, "❌ Invalid login credentials.")

    return render(request, 'UserLogin.html')



from .models import PredictionHistory
from django.utils.timesince import timesince

def UserHome(request):
    user_id = request.session.get('id')
    user = UserRegistrationModel.objects.get(id=user_id)

    # Count of predictions made by user
    prediction_count = PredictionHistory.objects.filter(user=user).count()

    # Recent predictions (latest 3)
    recent_predictions = PredictionHistory.objects.filter(user=user).order_by('-created_at')[:3]

    # Dummy model accuracy (or load from model training)
    model_accuracy = 90.0

    # Dummy health alerts (you can connect to alerts model later)
    health_alerts = 2

    prediction_logs = []
    for p in recent_predictions:
        prediction_logs.append({
            'disease': p.predicted_disease,
            'confidence': round(p.confidence, 1),
            'time': timesince(p.created_at) + " ago",
        })

    return render(request, "users/UserHome.html", {
        'prediction_count': prediction_count,
        'model_accuracy': model_accuracy,
        'health_alerts': health_alerts,
        'prediction_logs': prediction_logs,
    })



def view_data(request):
    from django.conf import settings
    import pandas as pd
    import os

    file_path = os.path.join(settings.MEDIA_ROOT, 'final_dataset_30000.csv')
    d = pd.read_csv(file_path)

    # Move 'Disease' column to the end if it exists
    if 'Disease' in d.columns:
        cols = [col for col in d.columns if col != 'Disease'] + ['Disease']
        d = d[cols]

    # Show only first 100 records
    d = d.head(100)

    context = {'dataset': d}
    return render(request, 'users/dataset.html', context)



# Django View for Model Training


from django.shortcuts import render
import numpy as np
import joblib
import google.generativeai as genai
from .models import PredictionHistory 
from django.conf import settings

# Configure Gemini API
if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
else:
    model = None

# Load ML components once at the top (recommended)
best_model = joblib.load("media/best_model.pkl")
scaler = joblib.load("media/scaler.pkl")
label_encoders = joblib.load("media/label_encoders.pkl")
disease_encoder = joblib.load("media/disease_encoder.pkl")

def prediction(request):
    predicted_disease = None
    precautions = None

    if request.method == "POST":
        try:
            # Collect inputs
            age = int(request.POST.get("age"))
            gender = request.POST.get("gender")
            symptoms = [request.POST.get(f'symptom_{i+1}') for i in range(7)]

            encoded = [age]
            encoded.append(label_encoders["Gender"].transform([gender])[0])

            for i, symptom in enumerate(symptoms):
                key = f"Symptom{i+1}"
                if symptom in label_encoders[key].classes_:
                    val = label_encoders[key].transform([symptom])[0]
                else:
                    val = 0
                encoded.append(val)

            input_data = scaler.transform(np.array(encoded).reshape(1, -1))
            prediction_index = best_model.predict(input_data)[0]
            predicted_disease = disease_encoder.inverse_transform([prediction_index])[0]

            # Save prediction history
            user_id = request.session.get('id')
            user = UserRegistrationModel.objects.get(id=user_id)

            # Confidence score if model supports it (optional)
            confidence = best_model.predict_proba(input_data).max() * 100

            PredictionHistory.objects.create(
                user=user,
                predicted_disease=predicted_disease,
                confidence=confidence
            )

            # Generate health advice with Gemini
            if model:
                prompt = (
                    f"What are the general precautions for {predicted_disease}? "
                    f"Give 4 to 5 short bullet points. Add a note at the end: "
                    f"'Disclaimer: I am not a doctor. Please consult a medical professional.'"
                )
                response = model.generate_content(prompt)
                precautions = response.text.strip()
            else:
                precautions = "Gemini API key not configured. Please add it to your .env file."

        except Exception as e:
            predicted_disease = "Error in prediction"
            precautions = str(e)

        return render(request, "users/prediction.html", {
            "predicted_disease": predicted_disease,
            "precautions": precautions
        })

    return render(request, "users/prediction.html")



from django.shortcuts import render
from .train_models import train_models  # Make sure this path is correct

def training(request):
    accuracies, best_model_name = train_models()
    return render(request, 'users/modelresults.html', {
        'accuracies': accuracies,
        'best_model': best_model_name
    })


