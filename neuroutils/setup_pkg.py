import os

# 项目目录结构
project_name = "/home/kfchen/neuron_seg_human/neuroutils"
dirs = [
    f"{project_name}/image",
    f"{project_name}/swc",
    f"{project_name}/meta",
    f"{project_name}/visualization",
    f"{project_name}/scripts",
    f"{project_name}/config",
    f"{project_name}/utils",
    f"{project_name}/tests"
]

files_with_content = {
    f"{project_name}/__init__.py": "# neuroutils package: Utility functions for neuron segmentation, reconstruction, preprocessing, and visualization.\n",
    f"{project_name}/image/__init__.py": "",
    f"{project_name}/image/checker.py": "# Functions for checking image validity, dimensions, and content\n",
    f"{project_name}/image/preprocessor.py": "# Preprocessing functions for cropping, resizing, and denoising images\n",
    f"{project_name}/image/io.py": "# Functions for reading and writing image files (TIFF, PNG, etc.)\n",
    f"{project_name}/swc/__init__.py": "",
    f"{project_name}/swc/parser.py": "# Parse SWC files and extract neuron tree structures\n",
    f"{project_name}/swc/metrics.py": "# Compute neuron structure metrics from SWC data\n",
    f"{project_name}/swc/fixer.py": "# Functions to fix common structural issues in SWC files\n",
    f"{project_name}/meta/__init__.py": "",
    f"{project_name}/meta/validator.py": "# Validate neuron metadata (species, brain region, etc.)\n",
    f"{project_name}/meta/converter.py": "# Convert neuron metadata between JSON, CSV, and other formats\n",
    f"{project_name}/meta/schema.py": "# Define metadata schema using pydantic or dict structures\n",
    f"{project_name}/visualization/__init__.py": "",
    f"{project_name}/visualization/swc_plot.py": "# 2D and 3D plotting functions for SWC neuron structures\n",
    f"{project_name}/visualization/image_plot.py": "# Functions for overlaying images and masks for visualization\n",
    f"{project_name}/visualization/metric_plot.py": "# Visualization of neuron metrics (e.g. histograms, heatmaps)\n",
    f"{project_name}/scripts/__init__.py": "",
    f"{project_name}/scripts/batch_check.py": "# Script for batch checking image, SWC, and metadata files\n",
    f"{project_name}/scripts/auto_pipeline.py": "# One-click pipeline for neuron image processing and reconstruction\n",
    f"{project_name}/scripts/report_generator.py": "# Generate HTML or PDF reports of processed neuron data\n",
    f"{project_name}/config/__init__.py": "",
    f"{project_name}/config/settings.py": "# Default configuration settings (paths, thresholds, etc.)\n",
    f"{project_name}/utils/__init__.py": "",
    f"{project_name}/utils/logger.py": "# Logging utility for console and file outputs\n",
    f"{project_name}/utils/file_ops.py": "# File and directory operations (search, extension check, etc.)\n",
    f"{project_name}/utils/validators.py": "# Common validators for files, paths, and data formats\n",
    f"{project_name}/tests/test_image.py": "# Tests for image module\n",
    f"{project_name}/tests/test_swc.py": "# Tests for SWC module\n",
    f"{project_name}/tests/test_meta.py": "# Tests for meta module\n",
    f"{project_name}/tests/test_visualization.py": "# Tests for visualization module\n",
    f"{project_name}/setup.py": '''\
from setuptools import setup, find_packages

setup(
    name="neuroutils",
    version="0.1.0",
    description="Utility library for neuron segmentation, reconstruction, and visualization.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-image",
        "scipy"
    ],
    python_requires=">=3.7",
)
''',
    "README.md": "# neuroutils\n\nUtility functions for neuron segmentation, reconstruction, metadata validation, and visualization.",
    "requirements.txt": "numpy\npandas\nmatplotlib\nscikit-image\nscipy\n"
}

# 创建目录和文件
for d in dirs:
    os.makedirs(d, exist_ok=True)

for file_path, content in files_with_content.items():
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
