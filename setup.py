from setuptools import setup, find_packages

setup(
  name = 'st-moe-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.23',
  license='MIT',
  description = 'ST - Mixture of Experts - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/st-moe-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'mixture of experts'
  ],
  install_requires=[
    'beartype',
    'CoLT5-attention>=0.10.15',
    'einops>=0.6',
    'torch>=2.0',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
