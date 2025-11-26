from setuptools import setup, find_packages
from pathlib import Path

req_file = Path(__file__).parent / "requirements.txt"
install_requires = req_file.read_text().splitlines() if req_file.exists() else []

setup(
    name="doAn",
    version="0.1",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=install_requires,
)