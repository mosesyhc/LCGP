import setuptools

setuptools.setup(
    name='lcgp',
    version='0.1',
    author='Moses Chan',
    author_email='mosesyhc@u.northwestern.edu',
    description='Latent component Gaussian process for emulation of '
                'general stochastic simulation high-dimensional outputs',
    packages=['src'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    setup_requires=[
        'setuptools>=18.0'
    ]
)
