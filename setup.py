from setuptools import setup, find_packages

setup(
    name='MPO_NoisyRCS',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    # description='A description of your package',
    # author='',
    # author_email='your.email@example.com',
    install_requires=[
        'numpy',
        'scipy'
    ],
)