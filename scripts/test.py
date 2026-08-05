
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

import sys
import numpy as np
from pathlib import Path

# Prefer the local checkout when running this notebook before pip installation.
# If you install this package with pip, you can remove this block.
# User configuration
repoRoot = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
srcPath = repoRoot / "src"
if srcPath.exists() and str(srcPath) not in sys.path:
    sys.path.insert(0, str(srcPath))
# ~ end of this block

from gundam_interface import GundamLoader, GundamRuntime, GundamInterface

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
gundam.initialize()

pars = gundam.getMinimizer().getFitParameters()
print(f"Initialized GUNDAM with {len(pars)} parameters")
for index, parameter in enumerate(pars):
    print(
        f"{index:3d}: {parameter.getName()} "
        f"prior={parameter.getPrior():.8g} "
        f"step={parameter.getStepSize():.8g} "
        f"value={parameter.getValue():.8g}"
    )

selectedPar = gundam.getMinimizer().getFitParameters()[2]
print(f"Selected parameter: {selectedPar.getSummary()}")
dial = None
ev = None
for dialEntry in gundam.getModel().getDialCacheList():
    dialList = dialEntry.getDialListAffecting(selectedPar)
    print(dialList)
    if len(dialList) == 1:
        dial = dialList[0]
        ev = dialEntry.getEvent()
        break

orig = selectedPar.getValue()

print(dial.getDialSummary())
print(ev.getSummary())

