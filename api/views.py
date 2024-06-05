from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import serializers
from coselector.coselector_0527 import process_pick_with_api  # Ensure this is the correct import path
import asyncio


# class CoSelectorView(APIView):
#     async def post(self, request):
#         # Extract model_name and resource_list from the request data
#         model_name = request.data.get('model_name')
#         resource_list = request.data.get('resource_list', [])

#         # Call the function to process the API with the received data
#         response_data = await process_pick_with_api_async(resource_list=resource_list, model_name=model_name)
        
#         # Return the response
#         return Response(response_data, status=status.HTTP_201_CREATED)


# async def process_pick_with_api_async(resource_list, model_name):
#     # Your processing logic here
#     await asyncio.sleep(5)  # Simulating a long-running task
#     return {"message": "Processing complete"}

from rest_framework.decorators import api_view

@api_view(['POST'])
async def coselector_view(request):
    # Extract model_name and resource_list from the request data
    model_name = request.data.get('model_name')
    resource_list = request.data.get('resource_list', [])

    # Call the function to process the API with the received data
    response_data = await process_pick_with_api_async(resource_list=resource_list, model_name=model_name)
    
    # Return the response
    return Response(response_data, status=status.HTTP_201_CREATED)

async def process_pick_with_api_async(resource_list, model_name):
    # Your processing logic here
    await asyncio.sleep(5)  # Simulating a long-running task
    return {"message": "Processing complete"}

