# Python

### Install python 3.10

https://gist.github.com/rutcreate/c0041e842f858ceb455b748809763ddb

### Install environment

`python3 -m venv .venv`
`.venv/bin/pip install -r requirements.txt`

Check package version : `pip freeze | findstr onnx`  
Install package version : `pip install --force-reinstall -v "onnx==1.14.0"`

# Django

Generate base site : `django-admin startproject mysite`  
Generate sub site : `python manage.py startapp detections`  
Start site : `python manage.py runserver`

Update schema : `.venv/bin/python manage.py makemigrations && .venv/bin/python manage.py migrate`

Create first admin user : `python manage.py createsuperuser`  
Export
data :
`venv/bin/python manage.py dumpdata --format=yaml --exclude=admin --exclude=auth --exclude=authtoken --exclude=sessions --exclude=contenttypes --exclude=detections --exclude=configuration.log --exclude=configuration.zone`  
Import data : `python manage.py loaddata data.yaml`

# Dev

Start
kill old backoffice : `lsof -t -i tcp:8000 | xargs kill -9`
backoffice : `.venv/bin/python manage.py runserver_plus --cert-file ~/cert.pem --key-file ~/key.pem 8000`  
Kill processus : `pkill -9 ffmpeg`  
Start vision : `.venv/bin/python manage.py vision`

# Server

Deployment statics files : `.venv/bin/python manage.py collectstatic`
Authorization port : `sudo ufw allow 8000`

# CUDA

Check CUDA : `nvidia-smi`
Install PyTorch : `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`

# HAILO

## From sources

### Uninstall

`sudo dpkg --purge hailort-pcie-driver`  
`sudo dpkg --purge hailo-all`  
`sudo dpkg --purge hailort`
`sudo dpkg --purge hailo-tappas-core`
`sudo dpkg --purge hailofw`

### Build

Hailort : `cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release && cmake --build build --config release`
Tappas : `./install.sh`
Driver : `dpkg -i hailort-pcie-driver_4.19.0_all.deb`

### Tools

Show usage : `hailortcli monitor`
Visualize model : https://netron.app/

### Notes

#### Outputs

Ok :

- 41 => 42 (8)
- 52 => 53 (16)
- 62 => 63 (32)

Pas géré : 39, 50, 60

# Jupyter

`nano /home/jaden/.jupyter/jupyter_notebook_config.py`
`venv/bin/python manage.py shell_plus --lab`

# Links

https://www.digitalocean.com/community/tutorials/how-to-serve-django-applications-with-uwsgi-and-nginx-on-ubuntu-16-04#setting-up-the-uwsgi-application-server