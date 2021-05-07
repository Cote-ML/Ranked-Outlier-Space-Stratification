import setuptools

setuptools.setup(
    name="ranked-outlier-space-stratification", # Replace with your own username
    version="0.0.1",
    author="Lee Cote",
    author_email="brian.lee.cote@gmail.com",
    description="Outlier detection model that stratifies data into levels of outlierness",
    url="https://github.com/LeeCote94/Ranked-Outlier-Space-Stratification/",
    project_urls={
        "Bug Tracker": "https://github.com/LeeCote94/Ranked-Outlier-Space-Stratification/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)