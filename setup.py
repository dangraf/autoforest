from setuptools import setup, find_packages

def read_requirements(filepath):
    with open(filepath, 'r') as f:
        content = f.read().strip()
    return content.split('\n')


setup(
    name='autoforest',
    version='0.0.2',
    description='tabular data thing',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    #install_requires=read_requirements('requirements.txt')
)
# install_requires=read_requirements('requirements.txt')
