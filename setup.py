import setuptools

requirements=[
    'torch>=2.0.1',
    'numpy>=1.18.3',
    'scipy>=1.10.1'
]

setuptools.setup(
    packages=setuptools.find_packages(exclude=['tests']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    include_package_data=True
)
