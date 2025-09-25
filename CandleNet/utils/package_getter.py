import sys
import subprocess


def get_lib(package: str):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install package {package}. Error: {e}")
        sys.exit(1)