from setuptools import find_packages, setup
from typing import List
import os

HYPEN_E_DOT = "-e."

def get_requirements(file_path: str) -> List[str]:
    '''
    will return list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and req.strip() != HYPEN_E_DOT]
    return requirements

setup(
    name='mlproj',
    version='0.0.1',
    author='Priya',
    author_email='mgpriya03@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(os.path.join(os.path.dirname(__file__), 'requirements.txt'))
)
