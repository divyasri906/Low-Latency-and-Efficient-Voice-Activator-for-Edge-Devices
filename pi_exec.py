import paramiko
import sys

def get_client():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('10.255.2.225', username='gspi5', password='12345678', timeout=10)
    return client

def run_cmd(cmd):
    client = get_client()
    print(f"=== RUNNING: {cmd} ===")
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    print("--- STDOUT ---")
    print(out)
    if err.strip():
        print("--- STDERR ---")
        print(err)
    client.close()
    return out, err

if __name__ == '__main__':
    if len(sys.argv) > 1:
        cmd = " ".join(sys.argv[1:])
    else:
        cmd = "pip3 list"
    run_cmd(cmd)
