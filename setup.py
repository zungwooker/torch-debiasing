from setuptools import setup, find_packages

setup(
    name='debiasing',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    author='zungwooker',
    author_email='zungwooker@gmail.com',
    description='A set of tools for debiasing research.',
    license='MIT',
)
