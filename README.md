# Geometric-ventricle-reconstruction Pipeline
## Description
Blender-addon for the geometric reconstruction of time-varying 3D ventricle geometries lacking clear mitral and aortic valve interfaces.!!!

# Installation
Installation-video: !!!
- Install Blender 3.1
- Install Python

!!!

```python
def hello_world():
    print("Hello, world!")
```

## Running Blender with Python environment variables
Windows: Run Powershell
Go to the directory of Blender and run it with Python system environment
```bash
cd PATH
./blender.exe --python-use-system-env
```
## Installation of Blender case with addons
After opening Blender 3.1 using Powershell load the blend-file provided in the repository. It contains objects used in the addon.
In Blender go to Edit→Preferences→Add-ons:
- Search for looptools and tick the checkbox to install it
- Install ventricle-reconstruction-pipeline.py from repository and tick the checkbox to install it for the current blend-file
## Installation of numba in Blender Python console
Open the integrated Blender Python console
```bash
import sys
import subprocess
subprocess.call([sys.exec_prefix + '\\bin\\python.exe', '-m', 'ensurepip'])
subprocess.call([sys.exec_prefix + '\\bin\\python.exe', '-m', 'pip', 'install', 'numba'])
```
If pip is missing (Output 0 in Blender Python console):
```bash
import ensurepip
ensurepip.bootstrap()
from pip._internal import main
main(args=['install','numba'])
```
## Usage
Video-tutorial:!!!\
otherwise: \
- Import ventricle geometries: File→Import→.STL→All files
!!!






## Visualization of distance to original
During the usage of the pipeline the longitudinal shift is saved as a variable bound to the respective object (ventricle 0 ... X). The user has to re-import the raw data and rename it to 'ref_obj'. While the reconstructed object is selected pressing the button 'Color minimal distance to raw object' will compute the minimal distance from each face-center of the reconstructed ventricle to any face-center of the reference object. The faces of the object are then colored with the distances which are normalized with the maximum value resulting in a scale from 0 to 1 (blue→white→red). To view the colors select 'Material Preview' in Blender (top right in 3D Viewport). 

## Authors and acknowledgment
- Author: Daniel Verhülsdonk
- Acknowledgment: Jan-Niklas Thiel
- Michael Neidlin

## License
!!!

## Project status
Development has been terminated. 
