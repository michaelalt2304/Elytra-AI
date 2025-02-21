import os
#### ADDITIONAL INSTRUCTIONS HERE ####
# Add an "elif" statement  for your OS if necessary
# Inside of the system-dependent block below, specify some unique environment variable
# to your computer (see all by printing out os.environ) and add the specification below the try block
######################################

#### HOW TO ADD A PACKAGE ####
# ADD package and most recent version to requirements.txt in folder
# DELETE virtual in this repository, if it's there
# Run this file with Python again
##############################
VIRTUAL_FOLDER = "virtual"
CLASSIFICATION_DATA= "classification"
current_system = os.environ["OS"]
print(current_system)
if current_system == "Windows_NT":
    if "COMPUTERNAME" in os.environ.keys() and os.environ["COMPUTERNAME"] == "INSPIRATION":
        # Michael's 3.8.20 Python Path
        PYTHON38 = os.path.join("C:/","Python38", "python.exe")
        # IF YOUR SYSTEM IS WINDOWS, ADD ELIF BELOW
    else:
        assert("Please assign PYTHON38 the proper runtime path, as specified in run_me.py")
    VIRTUAL_PIP = os.path.join(VIRTUAL_FOLDER, "Scripts", "pip.exe")
    VIRTUAL_PYTHON = os.path.join(VIRTUAL_FOLDER, "Scripts", "python.exe")
# IF YOUR SYSTEM IS NOT MAC OR WINDOWS, ADD ELIF HERE AND SEARCH FOR BINARIES
else:
    PYTHON38 = "python3"
    VIRTUAL_FOLDER = "virtual"
    VIRTUAL_PIP = "virtual/bin/pip3"
    VIRTUAL_PYTHON = "virtual/bin/python3"
    
if not os.path.exists(VIRTUAL_FOLDER):
    print("Setting up...")
    os.mkdir(VIRTUAL_FOLDER)
    os.system(f"{PYTHON38} -m venv {VIRTUAL_FOLDER}")
    os.system(f"{VIRTUAL_PYTHON} -m pip install --upgrade pip")

dummy = os.path.join(VIRTUAL_FOLDER, "initialized.txt")
if not os.path.exists(dummy):
    status = os.system(f"{VIRTUAL_PIP} install -r requirements.txt")
    with open(dummy, 'a') as f:
        f.write("All initialized in terms of packages, etc, should run smoothly. If not contact mjstraus2304@gmail.com")

status = os.system(f"{VIRTUAL_PYTHON} main.py")
