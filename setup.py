import setuptools

setuptools.setup(
    name="RankedOutliers",
    version="1.0.0",
    author="Lee Cote",
    author_email="brian.lee.cote@gmail.com",
    description="Outlier detection model that stratifies NxM Array into levels of outlierness",
    url="https://github.com/BrianCote/Ranked-Outlier-Space-Stratification/",
    project_urls={
        "Bug Tracker": "https://github.com/BrianCote/Ranked-Outlier-Space-Stratification/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={'': 'src'},
    packages=setuptools.find_packages('src'),
    python_requires=">=3.6",
)
