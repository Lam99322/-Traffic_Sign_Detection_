from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="traffic-sign-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Traffic Sign Detection System using YOLO11 and CNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/traffic-sign-detection",
    packages=find_packages(exclude=["tests", "notebooks", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "traffic-sign-detect=src.yolo_detector:main",
            "traffic-sign-train-yolo=src.training.train_yolo:main",
            "traffic-sign-train-cnn=src.training.train_cnn:main",
            "traffic-sign-server=server.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "templates/*.html", "static/**/*"],
    },
)
