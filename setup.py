import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="procedural-caves-scouttea",
    version="0.0.1",
    author="scouttea",
    author_email="scouttea.q@gmail.com",
    description="Functions and tools for generating realistic 2d caves",
    long_description=long_description,
    url="https://https://github.com/scouttea/ProceduralCaves",
    project_urls={
        "Bug Tracker": "https://github.com/scouttea/ProceduralCaves/issues",
    },
    license='BSD 3-clause "New" or "Revised License"',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'numpy',
          'scipy',
          'numba'
      ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
