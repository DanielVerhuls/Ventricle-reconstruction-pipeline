# Geometric-ventricle-reconstruction Pipeline

## Description
Blender-addon for the geometric reconstruction of time-varying 3D ventricle geometries lacking clear mitral and aortic valve interfaces.!!!

# Installation
!!!

```python
def hello_world():
    print("Hello, world!")
```

!!! Njit Installation in Blender Python console
```bash
import sys
import subprocess
subprocess.call([sys.exec_prefix + '\\bin\\python.exe', '-m', 'ensurepip'])
subprocess.call([sys.exec_prefix + '\\bin\\python.exe', '-m', 'pip', 'install', 'numba'])
```
If pip is missing (Output 0 i´n Blender Python console):
```bash
import ensurepip
ensurepip.bootstrap()
from pip._internal import main
main(args=['install','numba'])
```

## Usage




Looptools package is used for this application.
!!!


## Visualization of distance to original
During the usage of the pipeline the longitudinal shift is saved as a variable bound to the respective object (ventricle 0 ... X). The user has to re-import the raw data and rename it to 'ref_obj'. While the reconstructed object is selected pressing the button 'Color minimal distance to raw object' will compute the minimal distance from each face-center of the reconstructed ventricle to any face-center of the reference object. The faces of the object are then colored with the distances which are normalized with the maximum value resulting in a scale from 0 to 1 (blue->white->red). To view the colors select 'Material Preview' in Blender (top right in 3D Viewport). 

## Authors and acknowledgment
Daniel Verhülsdonk
Jan-Niklas Thiel
Michael Neidlin

## License
!!!

## Project status
Development has slowed down. 
