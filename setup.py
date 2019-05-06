from setuptools import setup, find_packages
# from distutils.core import setup

setup(
    name='QAbot',
    version='0.0.1',
    author='Paul E Gradie',
    author_email='paul.e.gradie@gmail.com',
    packages=find_packages(exclude=['tests']),
    scripts=[''],
    description='Package for descriptive analysis.',
    install_requires=[
        "pandas == 0.22.0",
        "numpy == 1.14.3",
        "pathos",
        "seaborn",
        "sklearn",
        "tqdm"
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],

)

