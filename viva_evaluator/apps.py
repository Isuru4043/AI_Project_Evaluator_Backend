from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)


class VivaEvaluatorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'viva_evaluator'

    def ready(self):
        # Warm up SBERT and CrossEncoder ML models in a daemon thread at server boot
        # so HTTP requests experience zero cold-start delay.
        import threading

        def _warmup_models():
            try:
                from viva_evaluator.services.rag.embeddings import _get_model as get_sbert
                from viva_evaluator.services.rag.rerank import _get_model as get_reranker
                logger.info('[boot] Warmup: pre-loading SBERT embedding model...')
                get_sbert()
                logger.info('[boot] Warmup: pre-loading CrossEncoder reranker model...')
                get_reranker()
                logger.info('[boot] Warmup complete: RAG models pre-loaded in RAM.')
            except Exception as exc:
                logger.warning('[boot] Warmup failed: %s', exc)

        threading.Thread(target=_warmup_models, daemon=True).start()

        # Schedule the sweeper task
        def _schedule_tasks():
            from django_q.models import Schedule
            if not Schedule.objects.filter(func='viva_evaluator.tasks.sweep_expired_sessions_task').exists():
                Schedule.objects.create(
                    func='viva_evaluator.tasks.sweep_expired_sessions_task',
                    schedule_type=Schedule.MINUTES,
                    minutes=5,
                    repeats=-1
                )
        
        try:
            _schedule_tasks()
        except Exception as e:
            logger.warning('[boot] Task scheduling failed (DB might not be ready): %s', e)

