"""FastAPI back end for the IndexTTS 2.5 Easy GUI web interface.

The package is framework-light on purpose: services are plain Python modules
that raise :class:`tools.gui_server.errors.ServiceError`, and ``app.py`` maps
them onto HTTP routes. Nothing here imports torch at module import time.
"""
