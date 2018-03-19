import setuptools as setup

def find_packages():
    return ['lightnet'] + ['lightnet.'+p for p in setup.find_packages('lightnet')]

def get_version():
    with open('VERSION', 'r') as f:
        version = f.read().splitlines()[0]
    with open('lightnet/version.py', 'w') as f:
        f.write('#\n')
        f.write('#   Lightnet version: Automatically generated version file\n')
        f.write('#\n\n')
        f.write(f'__version__ = "{version}"\n')
    
    return version

setup.setup(name='lightnet',
            version=get_version(),
            author='EAVISE',
            description='Building blocks for recreating darknet networks in pytorch',
            long_description=open('README.md').read(),
            packages=find_packages(),
            test_suite='test',
            install_requires=[
                'numpy',
                'Pillow',
                'brambox',
            ],
            extras_require={
                'visual': ['visdom']
            },
)
