import paramiko

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("Connecting to 10.255.2.225...")
    client.connect('10.255.2.225', username='gspi5', password='12345678', timeout=10)
    print("Connected successfully!")
    
    cmd = "uname -a; lscpu | head -n 20; python3 --version; which pip3; free -h"
    stdin, stdout, stderr = client.exec_command(cmd)
    print("STDOUT:\n", stdout.read().decode())
    err = stderr.read().decode()
    if err:
        print("STDERR:\n", err)
    client.close()

if __name__ == '__main__':
    main()
