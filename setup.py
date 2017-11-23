import setuptools as setup

def find_packages():
    return ['lightnet'] + ['lightnet.'+p for p in setup.find_packages('lightnet')]

setup.setup(name='lightnet',
            version='0.0.1',
            author='EAVISE',
            description='Building blocks for recreating darknet networks in pytorch',
            long_description=open('README.md').read(),
            packages=find_packages(),
)
