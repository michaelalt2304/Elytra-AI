import paramiko
from scp import SCPClient
from time import time
def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

server="100.79.182.84"
port=22
user="elytra"
password="Elytra@2025"

ssh = createSSHClient(server, port, user, password)

cl = SCPClient(ssh.get_transport())
t1 = time()
cl.get("/home/elytra/AI/test2.txt", ".")

cl.put("move_to_linux.txt", "/home/elytra/AI/")
t2 = time()
print(t2 - t1)