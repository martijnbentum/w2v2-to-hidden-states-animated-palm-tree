import setuptools

# Read the contents of requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
	name='w2v2_hidden_states',
	version = '0.1',
	author = 'Martijn Bentum',
	author_email = 'martijn.bentum@.ru.nl',
	description = 'a package to handle wav2vec 2.0 hidden states outputs',
	long_description = '',
	long_description_content_type = 'text/markdown',
	url = 'https://github.com/martijnbentum/w2v2-to-hidden-states-animated-palm-tree',
	packages = setuptools.find_packages(),
    install_requires = requirements,
	classifiers = ['Programming Language :: Python :: 3',
		'License :: OSI Approved :: MIT License'],
	python_requires = '>=3.8')
