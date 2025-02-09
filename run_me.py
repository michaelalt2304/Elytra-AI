import venv
import os
#### UPDATE PATH HERE ####
try:
    PYTHON38 = "C:\\Python38\\python.exe" if os.environ["COMPUTERNAME"] == "INSPIRATION" else "python"
except Exception:
    PYTHON38 = "python"
####                  ####
VIRTUAL_FOLDER = "virtual"
VIRTUAL_PIP = f'{VIRTUAL_FOLDER}\\Scripts\\pip.exe'
VIRTUAL_PYTHON = f'{VIRTUAL_FOLDER}\\Scripts\\python.exe'

if not os.path.exists(VIRTUAL_FOLDER):
    print("Setting up...")
    os.mkdir(VIRTUAL_FOLDER)
    os.system(f"{PYTHON38} -m venv {VIRTUAL_FOLDER}")
    os.system(f"{VIRTUAL_PYTHON} -m pip install --upgrade pip")

dummy = f"{VIRTUAL_FOLDER}\\initialized.txt"
if not os.path.exists(dummy):
    status = os.system(f"{VIRTUAL_PIP} install -r requirements.txt")
    with open(dummy, 'a') as f:
        f.write("All initialized in terms of packages, etc, should run smoothly. If not contact mjstraus2304@gmail.com")

status = os.system(f"{VIRTUAL_PYTHON} main.py")
print(status)