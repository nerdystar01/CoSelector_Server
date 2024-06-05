from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import serializers
from coselector.coselector_0527 import process_pick_with_api  # Ensure this is the correct import path
# import asyncio


class CoSelectorView(APIView):
    def post(self, request):
        # Extract model_name and resource_list from the request data
        model_name = request.data.get('model_name')
        resource_list = request.data.get('resource_list', [])

        # Call the function to process the API with the received data
        response_data = process_pick_with_api(resource_list=resource_list, model_name=model_name)
        print("나가기 전에 체크 한사바리")
        print(response_data)
        return Response(response_data, status=status.HTTP_201_CREATED)
