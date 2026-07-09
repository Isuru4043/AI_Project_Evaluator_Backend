
class TrailingSlashAPIMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not request.path.endswith('/') and request.path.startswith('/api/'):
            request.path_info = request.path_info + '/'
            request.path = request.path + '/'
        return self.get_response(request)
