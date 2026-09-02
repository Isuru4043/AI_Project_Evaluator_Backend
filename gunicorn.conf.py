"""Gunicorn settings for the API server.

Gunicorn reads this file automatically when it starts with the repository as
its working directory, so the systemd unit on the server needs no edit.
Command-line flags still win over anything set here.

WHY THE TIMEOUT IS RAISED
-------------------------
Seat binding (``POST /api/sessions/<id>/attribution/bind/``) calls the Modal
face-recognition engine and waits for the answer. Measured against the live
endpoint with a three-student roster: 24s on a cold container and 15s warm,
before the kiosk's own upload of five camera frames is counted. Gunicorn's
default worker timeout is 30 seconds, so the worker handling that request was
being killed mid-flight. nginx then answers the browser itself, and an nginx
error page carries none of Django's CORS headers, so the kiosk reported

    "No 'Access-Control-Allow-Origin' header is present on the requested
     resource"

which looks like a CORS misconfiguration but is really a killed worker. The
CORS settings were verified correct: preflight and small POSTs to the same URL
return the right headers.

Keep this comfortably above the slowest recognition call. It is an upper bound
for a stuck worker, not a target: a warm run with cached enrollment vectors
finishes in a few seconds.
"""

# Seconds a worker may spend on one request before it is killed and replaced.
timeout = 180

# Let an in-flight request finish when the server is restarted or reloaded,
# so a deploy during a live session does not abort a scan already running.
graceful_timeout = 60

# nginx keeps upstream connections open; matching this avoids the proxy
# reusing a connection gunicorn has just closed.
keepalive = 5

# Access log to stdout so `journalctl -u gunicorn` shows request durations,
# which is what identifies a slow endpoint next time. %(D)s is microseconds.
accesslog = "-"
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(D)sus'
