from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import MLTask
from .serializers import MLTaskSerializer
from .polyp.detector import PolypDetector
from .llm.phi3 import Phi3Client


class MLTaskViewSet(viewsets.ModelViewSet):
  """
  Unified entrypoint for ML-powered background jobs.
  Currently records requests only; execution can be wired later.
  """

  serializer_class = MLTaskSerializer
  permission_classes = [permissions.AllowAny]

  def get_queryset(self):
    base_qs = MLTask.objects.select_related('created_by').order_by('-created_at')
    # For demo purposes, allow everyone to see all tasks
    return base_qs

  def perform_create(self, serializer):
    user = self.request.user if self.request.user.is_authenticated else None
    serializer.save(created_by=user)

  @action(detail=False, methods=['post'])
  def trigger(self, request):
    task_type = request.data.get('task_type')
    file_id = request.data.get('file_id')

    valid_types = {choice[0] for choice in MLTask.TASK_TYPES}

    if task_type not in valid_types:
      return Response(
        {'detail': 'Invalid task_type. Expected one of: {}'.format(', '.join(sorted(valid_types)))},
        status=status.HTTP_400_BAD_REQUEST,
      )

    if not file_id:
      return Response({'detail': 'file_id is required.'}, status=status.HTTP_400_BAD_REQUEST)

    user = request.user if request.user.is_authenticated else None
    task = MLTask.objects.create(
      file_id=file_id,
      task_type=task_type,
      status='queued',
      created_by=user,
    )

    # Future hook: enqueue job for Celery / external worker here
    if task_type == 'polyp_detect':
        try:
            # Synchronous execution for demo/MVP
            detector = PolypDetector.get_instance()
            
            # In this isolated version, file_id is treated as the file path
            import os
            image_path = task.file_id
            
            if os.path.exists(image_path):
                 detections = detector.predict(image_path)
                 task.result = {'detections': detections}
                 task.status = 'done'
                 task.save()
            else:
                 task.status = 'failed'
                 task.result = {'error': f'File not found: {image_path}'}
                 task.save()
                 
        except Exception as e:
            task.status = 'failed'
            task.result = {'error': str(e)}
            task.save()

    return Response({'task_id': task.id, 'status': task.status, 'result': task.result}, status=status.HTTP_202_ACCEPTED)

  @action(detail=True, methods=['post'])
  def explain(self, request, pk=None):
      """
      Generate a text explanation for the detection results using Phi-3.
      """
      task = self.get_object()
      
      if task.task_type != 'polyp_detect':
          return Response({'detail': 'Only polyp_detect tasks can be explained.'}, status=status.HTTP_400_BAD_REQUEST)
          
      if task.status != 'done':
          return Response({'detail': 'Task must be completed before explanation.'}, status=status.HTTP_400_BAD_REQUEST)
          
      detections = task.result.get('detections', [])
      
      import os
      file_name = os.path.basename(task.file_id) if task.file_id else "unknown_image"
      
      try:
          client = Phi3Client()
          explanation = client.generate_explanation(detections, file_name)
          
          # Update result with explanation
          task.result['explanation'] = explanation
          task.save()
          
          return Response({'explanation': explanation})
      except Exception as e:
          return Response({'detail': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
