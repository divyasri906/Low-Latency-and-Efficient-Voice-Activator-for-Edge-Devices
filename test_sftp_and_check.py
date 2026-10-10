import paramiko

def test_sftp():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('10.255.2.225', username='gspi5', password='12345678', timeout=10)
    sftp = client.open_sftp()
    
    # Ensure directory exists
    try:
        sftp.mkdir('/home/gspi5/kws_benchmarks')
    except IOError:
        pass
    
    # Write a test script directly
    test_code = """
import sys
mods = ['numpy', 'scipy', 'tflite_runtime', 'tensorflow', 'librosa', 'soundfile', 'psutil', 'torch']
print(f"Python version: {sys.version}")
for m in mods:
    try:
        mod = __import__(m)
        print(f"{m:15s}: {getattr(mod, '__version__', 'installed')}")
    except Exception as e:
        print(f"{m:15s}: NOT INSTALLED ({e})")
"""
    with sftp.open('/home/gspi5/kws_benchmarks/check_env.py', 'w') as f:
        f.write(test_code)
    
    print("Uploaded check_env.py to /home/gspi5/kws_benchmarks/check_env.py")
    sftp.close()
    
    stdin, stdout, stderr = client.exec_command('python3 /home/gspi5/kws_benchmarks/check_env.py')
    print("=== SCRIPT OUTPUT ===")
    print(stdout.read().decode())
    err = stderr.read().decode()
    if err:
        print("ERRORS:", err)
    client.close()

if __name__ == '__main__':
    test_sftp()
