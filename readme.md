A video target detect project based on color, shape, SURF with robust detection accuracy.

***
## 1. Requirements:
* Microsoft Visual Studio 2013
* OPENCV2.4.9

## 2. Configure Windows:
* Add new user variable, name:“OPENCV2_DIR”, value: "C:\opencv\build\"(or other path that you installed OPENCV2.4.9)
* Select Path(create one if does not exist), add: ";%OPENCV2_DIR%\x86\vc12\bin"

## 3. Configure Visual Studio Project:
### configure either in debug or release
* Create new C++ project(File > Project... > Visual C++ > Win32 Console Application)
* Open "Project Properties"(Solution Explorer > Properties > Configuration > All Configuration)
* Select "Additional Include Directories"(Configuration Properties > C/C++ > General), add "$(OPENCV2_DIR)\Include" in right section
* Select "Additional Library Directories"(Configuration Properties > C/C++ > Linker > General), add "$(OPENCV2_DIR)\x86\vc12\lib"
* Select "Additional Dependencies"(Configuration Properties > C/C++ > Linker > Input ), add:(for release just remove "d" before "." in filename)

    >opencv_contrib249d.lib  
    >opencv_core249d.lib  
    opencv_features2d249d.lib  
    opencv_flann249d.lib  
    opencv_gpu249d.lib  
    opencv_highgui249d.lib  
    opencv_imgproc249d.lib  
    opencv_legacy249d.lib  
    opencv_ml249d.lib  
    opencv_nonfree249d.lib  
    opencv_objdetect249d.lib  
    opencv_ocl249d.lib  
    opencv_photo249d.lib  
    opencv_stitching249d.lib  
    opencv_superres249d.lib  
    opencv_ts249d.lib  
    opencv_video249d.lib  
    opencv_videostab249d.lib  

*__PS__: all these libraries (lib files) are not required and you can choose your required library*
