import paramiko

def check():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('10.255.2.225', username='gspi5', password='12345678', timeout=10)
    
    code = """
import sys
mods = ['numpy', 'scipy', 'tflite_runtime', 'tensorflow', 'librosa', 'soundfile', 'psutil', 'torch']
for m in mods:
    try:
        mod = __import__(m)
        print(f"{m}: {getattr(mod, '__version__', 'installed')}")
    except Exception as e:
        print(f"{m}: NOT INSTALLED ({e})")
"""
    stdin, stdout, stderr = client.exec_command(f"python3 -c \"{code}\"")
    print("=== PI ENVIRONMENT CHECK ===")
    print(stdout.read().decode())
    err = stderr.read().decode()
    if err:
        print("ERRORS:", err)
    client.close()

if __name__ == '__main__':
    check()
