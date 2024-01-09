# Geometric-ventricle-reconstruction Pipeline
## Description
Blender-addon for the geometric reconstruction of time-varying 3D ventricle geometries lacking clear mitral and aortic valve interfaces.!!!

# Installation
Installation-video: !!!
- Install Blender 3.1
- Install Python
- Install pip: https://pip.pypa.io/en/stable/installation/
- Install Python Packages bpy, open3D and numba in console
```bash
python.exe -m pip install --upgrade pip
pip install bpy
pip install open3d
pip install numba
```
## Running Blender with Python environment variables
Windows: Run Powershell
Go to the directory of Blender and run it with Python system environment
```bash
cd PATH
./blender.exe --python-use-system-env
```
## Installation of numba in Blender Python console
After opening Blender 3.1 using Powershell load the blend-file provided in the repository. It contains objects used in the addon.
Open the integrated Blender Python console.
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
## Installation of Blender case with addons
In Blender go to Edit→Preferences→Add-ons:
- Search for looptools and tick the checkbox to install it
- Install ventricle-reconstruction-pipeline.py from repository and tick the checkbox to install it for the current blend-file
After that a new category should appear on the right side of the 3D Viewport called 'GVR-Pipeline'. Clicking it will open panels containing buttons,etc. used for the pipeline.
# Usage
Video-tutorial:!!!\
Open blend-file provided in the repository and follow the following steps.
## Import ventricle geometries 
File→Import→.STL→All files
Other file formats are also possible. But some formats e.g. wavefront (.obj) have to be imported individually.
## Setup pipeline
1. Sort volumes\
    1.1 Select all volumes
    1.2 Click button 'Sort volumes'
This restructures the list of selected objects such that the object with the smallest volume is the first object and all objects that were before that object are concatenated in the original order at the end of the object list. It also changes the names of the objects to the naming convention ventricle 0 ... X.

2. Setup ventricle position and rotation\
!!!
3. Setup valves\
!!!
4. Setup algorithm variables\
!!!
5. Select approach\
!!!
## Run pipeline
Select all ventricles and either run all steps with the button Quick reconstruction or do the following steps for a more comprehensive execution of the pipeline:
1. Remove basal region\
Press button 'Remove basal region' !!! description
2. Create basal region\
!!!
3. Connect basal and apical parts\
!!!
4. Add atrium, aorta and valves\
!!!
## Export files
Select all files to export in Blender.
File→Export→.STL→
- tick ASCII checkbox
- Batch Mode Object
- tick selection Only checkbox
- keep the other options at default
- Leave name empty\
→export STL
## Optional: Visualization of distance to original
During the usage of the pipeline the longitudinal shift is saved as a variable bound to the respective object (ventricle 0 ... X). The user has to re-import the raw data and rename it to 'ref_obj'. While the reconstructed object is selected pressing the button 'Color minimal distance to raw object' will compute the minimal distance from each face-center of the reconstructed ventricle to any face-center of the reference object resulting in a 3d-representation of the Hausdorff distance (https://cgm.cs.mcgill.ca/~godfried/teaching/cg-projects/98/normand/main.html). The faces of the object are then colored with the distances which are normalized with the maximum value resulting in a scale from 0 to 1 (blue→white→red). To view the colors select 'Material Preview' in Blender (top right in 3D Viewport). 
# Authors and acknowledgment
- Author: Daniel Verhülsdonk
- Supervision by: Jan-Niklas Thiel and Michael Neidlin 

# Project status
Development has been terminated. 

# License
MIT License

Copyright (c) [2024] [Institute of Applied Medical Engineering - Cardiovascular Engineering (AME-CVE) RWTH Aachen]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.