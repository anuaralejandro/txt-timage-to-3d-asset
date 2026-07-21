set CUDA_PATH=
set CUDA_PATH_V13_0=
set PATH=%PATH:C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;=%
set PATH=%PATH:C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64;=%
set CUDA_MODULE_LOADING=EAGER
C:\Users\datam\.conda\envs\enarmgpu\python.exe ComfyUI\main.py --windows-standalone-build --listen 0.0.0.0
echo If you see this and ComfyUI did not start try updating your Nvidia Drivers to the latest. If you get a c10.dll error you need to install vc redist that you can find: https://aka.ms/vc14/vc_redist.x64.exe
pause
