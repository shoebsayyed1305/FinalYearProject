from django.views.generic import TemplateView
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse 
from chat.forms import *
import cv2
from TESTING_CNN import get

class HomePage(TemplateView):
	template_name = 'index.html'


def DETECTION_PAGE(request):   
    if request.method == 'POST': 
        form = INPUT_IMAGE_FORM(request.POST, request.FILES) 
  
        if form.is_valid(): 
            form.save() 
            return redirect('display_result') 
    else: 
        form = INPUT_IMAGE_FORM() 
    return render(request, 'detection.html', {'form' : form}) 
  

def predicted_results(request):
		if request.method == 'GET':
			sym1 =request.GET['filename']
			print(sym1)
		return render(request, 'detect_tumor.html')

def display_result(request):   
        IMAGES = INPUT_IMAGES.objects.all()
        print(IMAGES[len(IMAGES)-1].Input_image )
        print('./' + str(IMAGES[len(IMAGES)-1].Input_image ))


        bt_result = get('./' + str(IMAGES[len(IMAGES)-1].Input_image ))

        return render(request, 'display_result.html', {'input_image':IMAGES[len(IMAGES)-1].Input_image ,'result':bt_result})

