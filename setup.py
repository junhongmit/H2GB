from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='H2GB',
    version='0.1.0',
    description='A graph benchmark library for heterophilic and heterogeneous graphs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Junhong Lin',
    author_email='junhong@mit.edu',
    url='https://github.com/junhongmit/H2GB',
    packages=find_packages(include=['H2GB', 'H2GB.*']),
    entry_points={
        'console_scripts': [
            'h2gb_train = H2GB.main:run',  # Update main with the actual function name in main.py
        ],
    },
    install_requires=[
        'torch',
        'torch_geometric',
        'torch_sparse',
        'torch_scatter',
        'ogb',
        'yacs',
        'dill',
        'torchmetrics',
        'wandb',
        'seaborn',
        'gdown',
        'pytorch_lightning',
        'datatable',
        'prettytable',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)