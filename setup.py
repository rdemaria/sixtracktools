import setuptools

setuptools.setup(
        name='sixtracktools',
        version='0.0.9',
        description='Python tools for sixtrack',
        author='Riccardo De Maria',
        author_email='riccardo.de.maria@cern.ch',
        url='https://github.com/rdemaria/sixtracktools',
        packages=['sixtracktools'],
        package_dir={'sixtracktools': 'sixtracktools'},
        install_requires=['numpy']
)


