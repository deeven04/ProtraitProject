# PortraitProject

This repository contains the implementation of **Light Maps** for relighting portraits. The code is based on the paper ```LightMaps.pdf```.

## Steps for Execution:
1. Follow the ```Step 1``` mentioned in [DSINE GitHub](https://github.com/baegwangbin/DSINE) to install the required dependencies and set up the necessary files.
2. Replace ```test_minimal.py``` with ```normals.py``` in the folder ```projects/dsine/```.
3. Save the file ```light_maps.py``` in the same folder.
4. Save the input images in the folder ```projects/samples/img/```.
5. Update the input image path in ```light_maps.py``` to point to your desired image. For example:
   ```python
   file_path = os.path.abspath('samples/img/input_img.hdr')
6. Navigate to the folder ```projects/dsine/```, and run the following command:
   ```python
   python light_maps.py ./experiments/exp001_cvpr2024/dsine.txt
7. The output images are save in ```projects/samples/output/```
