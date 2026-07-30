import logging
from django_q.tasks import async_task
from core.models import ModuleMaterial
from viva_evaluator.services.indexing.module_indexer import index_module_material

logger = logging.getLogger(__name__)

def process_module_material_task(material_id: str):
    """
    Background task to extract and index module materials.
    """
    try:
        material = ModuleMaterial.objects.get(id=material_id)
        logger.info(f"Starting indexing for ModuleMaterial: {material_id}")
        index_module_material(material)
        logger.info(f"Finished indexing for ModuleMaterial: {material_id}")
    except ModuleMaterial.DoesNotExist:
        logger.error(f"ModuleMaterial {material_id} not found.")
    except Exception as e:
        logger.error(f"Error in process_module_material_task for {material_id}: {e}")

def sweep_expired_sessions_task():
    """
    Background task scheduled every 5 minutes to sweep and expire old sessions.
    """
    from core.models import EvaluationSession
    from django.utils import timezone
    
    now = timezone.now()
    expired = EvaluationSession.objects.filter(
        status__in=[EvaluationSession.Status.SCHEDULED, EvaluationSession.Status.IN_PROGRESS],
        scheduled_end__lt=now
    )
    count = expired.update(status=EvaluationSession.Status.EXPIRED)
    if count > 0:
        logger.info(f"Swept and expired {count} abandoned evaluation sessions.")
