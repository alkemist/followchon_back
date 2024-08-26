import os, sys
wsgi_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.dirname(wsgi_dir)
sys.path.append(project_dir)
sys.path.append('/home/jaden/Projects/followchon_back/admin')
os.environ['PYTHON_EGG_CACHE'] = '/home/jaden/Projects/followchon_back/.python-egg'
project_settings = os.path.join(project_dir,'settings')
os.environ['DJANGO_SETTINGS_MODULE'] ='admin.settings'
import django.core.handlers.wsgi
application =django.core.handlers.wsgi.WSGIHandler()