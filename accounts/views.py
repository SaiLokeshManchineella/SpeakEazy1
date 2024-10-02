from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views.generic import CreateView
from django.contrib.auth import login

class SignUpView(CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy("login")  # Always redirect to login page
    template_name = "registration/signup.html"

    def form_valid(self, form):
        self.object = form.save()
        login(self.request, self.object)  # Login the user after signup
        return redirect(self.success_url)  # Redirect to login page