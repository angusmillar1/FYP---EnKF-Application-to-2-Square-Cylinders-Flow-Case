import subprocess

subprocess.run("echo 'The run has finished!' | mail -s 'Job Done' acm21@ic.ac.uk", shell=True)