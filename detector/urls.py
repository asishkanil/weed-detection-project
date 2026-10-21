
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("detect/", views.detect, name="detect"),
     path("results/", views.results, name="results"),
    path("weed-management/", views.weed_management, name="weed_management"),
]

