from setuptools import setup, find_packages

setup(
    name="amsd",
    version="1.0.0",
    description="Context-adaptive temporal scale weighting for overlap-aware "
                "far-field speaker diarization",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0", "numpy>=1.24", "scipy>=1.10",
        "scikit-learn>=1.3", "PyYAML>=6.0",
    ],
)
