This repo contains code to:
1. walk a folder to collect information about a dicom exam. Only 1 dicom exam shoudl be in this folder system. It then selects the 'Best' series, which for this purpose is: 1) AXial, 2) slice thickness between 0.625 and 6mm, preferred 2mm, preferred SOFT kernel, required no IV Contrast. It then writes this out to a folder
2. This folder is converted to nifti using dcm2niix
3. The non-brain tissues are removed using run_synthstrip (synthstrip.py requires special docker environment)
4. AI model applied--can use blast, which is the industry reference but I find that not as good as the model developed here.
5. 
