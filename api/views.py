from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import serializers
from coselector.coselector_0527 import process_pick_with_api  # Ensure this is the correct import path


class CoSelectorView(APIView):
    def post(self, request):
        request_list = request.data
        print(request_list)
        response_data = process_pick_with_api(resource_list= request_list)
        return Response(response_data, status=status.HTTP_201_CREATED)
    