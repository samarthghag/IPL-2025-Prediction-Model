from setuptools import setup, find_packages

setup(
    name="ipl-prediction",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.3",
        "numpy>=1.24.3",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "scikit-learn>=1.2.2",
        "xgboost>=1.7.6",
        "lightgbm>=3.3.5",
        "jupyter>=1.0.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.2",
        "tensorflow>=2.12.0",
        "plotly>=5.14.1",
    ],
    author="IPL Prediction Team",
    author_email="example@example.com",
    description="A machine learning model to predict IPL tournament outcomes",
    keywords="ipl, cricket, machine learning, prediction",
    url="https://github.com/example/ipl-prediction",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 