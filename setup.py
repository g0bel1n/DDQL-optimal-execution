from distutils.core import setup

setup(
    name="ddql_optimal_execution",
    packages=["ddql_optimal_execution", "ddql_optimal_execution/agent", "ddql_optimal_execution/state", "ddql_optimal_execution/experience_replay", "ddql_optimal_execution/environnement", "ddql_optimal_execution/trainer", "ddql_optimal_execution/preprocessing"],
    version="0.1.1",
    license="MIT",
    description="Double Deep Q Learning for Optimal Trading Execution",
    author="Lucas Saban",
    author_email="lucas.saban@icloud.com",
    url="https://github.com/g0bel1n/DDQL-optimal-execution",
    download_url="https://github.com/g0bel1n/DDQL-optimal-execution/archive/refs/tags/v0.1-beta.tar.gz",
    keywords=["DDQL", "OPTIMAL EXECUTION", "TRADING"],
    install_requires=["numpy", "pandas", "torch", "tqdm"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],


)
