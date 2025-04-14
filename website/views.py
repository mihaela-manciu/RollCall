
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import RegisterForm
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
from datetime import datetime
from .models import AttendanceRecord
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
import cv2
import numpy as np
from keras.models import Model,load_model
import pandas as pd

vgg16_model = load_model('D:/Poli/Licenta/RollCall/RollCall/TestCNN/test_VGG16_0.0001_50.model.keras')
embeddings = pd.read_csv('D:/Poli/Licenta/RollCall/RollCall/TestPrototypical/class_embeddings_50_5_2_1.csv', index_col=0).values
class_names = pd.read_csv('D:/Poli/Licenta/RollCall/RollCall/TestPrototypical/class_embeddings_50_5_2_1.csv', index_col=0).index.to_list()

def home(request):

    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, "You Have Been Logged In!")
            return redirect('home')
        else:
            messages.success(request, "There Was An Error Logging In, Please Try Again...")
            return redirect('home')
    else:
        return render(request, 'home.html')

def logout_user(request):
    logout(request)
    messages.success(request, "You Have Been Logged Out...")
    return redirect('home')


def register_user(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            # Authenticate and login
            username = form.cleaned_data['username']
            password = form.cleaned_data['password1']
            user = authenticate(username=username, password=password)
            login(request, user)
            messages.success(request, "You Have Successfully Registered! Welcome!")
            return redirect('home')
    else:
        form = RegisterForm()
        return render(request, 'register.html', {'form': form})

    return render(request, 'register.html', {'form': form})




@login_required
def class_view(request, class_name):
    return render(request, 'class.html', {'class_name': class_name})


@login_required
def record_attendance_view(request):
    if request.method == 'POST':
        image_data = request.POST.get('image')
        format, imgstr = image_data.split(';base64,')
        nparr = np.frombuffer(base64.b64decode(imgstr), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        faces = detect_faces(img)
        recognized_faces = recognize_faces(img, faces, vgg16_model, embeddings, class_names)

        attendance_records = []
        for (x, y, w, h, class_name) in recognized_faces:
            attendance_records.append(class_name)
            AttendanceRecord.objects.create(student_name=class_name, class_name='A', timestamp=datetime.now())

        return JsonResponse({'recognized_faces': recognized_faces, 'attendance_records': attendance_records})

    return render(request, 'record_attendance.html')


def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def recognize_faces(image, faces, model, embeddings, class_names):
    recognized_faces = []
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (224, 224))
        face = np.expand_dims(face, axis=0) / 255.0
        face_features = model.predict(face)
        distances = np.linalg.norm(embeddings - face_features, axis=1)
        min_distance = np.min(distances)
        if min_distance < 0.5:
            class_name = class_names[np.argmin(distances)]
        else:
            class_name = 'Unknown'
        recognized_faces.append((x, y, w, h, class_name))
    return recognized_faces