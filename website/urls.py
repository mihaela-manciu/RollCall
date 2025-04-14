from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('logout/', views.logout_user, name='logout_user'),
    path('register/', views.register_user, name='register'),

    path('class/<str:class_name>/', views.class_view, name='class_attendance'),
    path('record_attendance/', views.record_attendance_view, name='record_attendance'),

]