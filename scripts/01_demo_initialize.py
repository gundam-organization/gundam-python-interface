from pathlib import Path
import sys

import numpy as np

nCpuThreads = 3
gundamLibPath = "/Users/nadrino/Documents/Work/Install/gundam/lib"
workDir = "/Users/nadrino/Documents/Work/Output/results/gundam/GundamInputOA2024"
configPath = "configOA2024.yaml"
overrideList = [
    "override/v12ProdRun45.yaml",
    "override/onlyFlux5.yaml",
    "override/noEigen.yaml",
]
dataType = "Asimov"  # "Asimov", "Toy", or "RealData"
seed = 12345

# Prefer the local checkout when running this notebook before pip installation.
# If you install this package with pip, you can remove this block.
# User configuration
repoRoot = Path.cwd().parent if Path.cwd().name == "scripts" else Path.cwd()
srcPath = repoRoot / "src"
if srcPath.exists() and str(srcPath) not in sys.path:
    sys.path.insert(0, str(srcPath))
# ~ end of this block

from gundam_interface import GundamInterface, GundamLoader, GundamRuntime  # noqa: E402

np.random.seed(seed)

runtime = GundamRuntime(
    loader=GundamLoader(gundamLibPath=gundamLibPath),
    workDir=workDir,
    nCpuThreads=nCpuThreads,
    configPath=configPath,
    overrideList=overrideList,
    dataType=dataType,
    randomSeed=seed,
)

runtime.toDict(includeConfigJsonString=False)

gundam = GundamInterface(runtime)
gundam.configure()
gundam.initialize(debugLogRedirection=True)
