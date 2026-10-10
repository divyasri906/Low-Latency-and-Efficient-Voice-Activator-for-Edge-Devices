import sys
import paramiko

def run(cmd, timeout=120):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('10.255.2.225', username='gspi5', password='12345678', timeout=15)
    print(f">>> Running on Pi: {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    client.close()
    
    # Safe print to Windows console
    safe_out = out.encode('ascii', errors='replace').decode('ascii')
    safe_err = err.encode('ascii', errors='replace').decode('ascii')
    print("=== STDOUT ===")
    print(safe_out)
    if safe_err.strip():
        print("=== STDERR ===")
        print(safe_err)
    return out, err

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'uname -a'
    run(cmd)
