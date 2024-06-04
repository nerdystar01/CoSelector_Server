from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import serializers
from coselector.coselector_0527 import process_pick_with_api  # Ensure this is the correct import path

class CoSelectorItemSerializer(serializers.Serializer):
    resource_id = serializers.IntegerField()
    image_np = serializers.CharField(max_length=100)
    similarity_score = serializers.FloatField(required=False, allow_null=True)

class CoSelectorSerializer(serializers.ListSerializer):
    child = CoSelectorItemSerializer()


class CoSelectorView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = CoSelectorSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data['data']
            response_data = process_pick_with_api(data)
            return Response(response_data, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    