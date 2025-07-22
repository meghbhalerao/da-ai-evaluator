from setuptools import setup, find_packages

setup(
    name='da_ai_evaluator',
    version='0.0.1',
    url='https://github.com/meghbhalerao/da-ai-evaluator',
    author='Megh Bhalerao',
    author_email='megh.bhalerao@gmail.com',
    description='Code for data augmentation of human demonstrations',
    packages=find_packages(include=['da_ai_evaluator', 'da_ai_evaluator.*']),
)