# 配置引导 #
***
## 1. 环境：##
* Microsoft Visual Studio 2013
* OPENCV2.4.9

## 2. Windows配置：##
* 添加环境变量，名称：“OPENCV2_DIR”，值：“C:\opencv\build\”（或者其他你安装OPENCV2.4.9的目录）
* 在用户变量（或者系统变量）下面选择变量名Path，添加值：“;%OPENCV2_DIR%\x86\vc12\bin”

## 3. Visual Studio 工程配置:
### 可以配置debug或者release
* 创建新的C++工程(File > Project... > Visual C++ > Win32 Console Application)
* 打开 "工程属性"(Solution Explorer > Properties > Configuration > All Configuration)
* 选择 "附加包含目录"(Configuration Properties > C/C++ > General), 在右边添加 "$(OPENCV2_DIR)\Include"
* 选择 "附加库目录"(Configuration Properties > C/C++ > Linker > General), 添加 "$(OPENCV2_DIR)\x86\vc12\lib"
* 选择 "附加依赖项"(Configuration Properties > C/C++ > Linker > Input ), 添加:(对于release版本，只要把库文件名末尾的"d"去掉即可)
>opencv_contrib249d.lib  
opencv_core249d.lib  
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

*__注意__: 这些库文件不一定都要添加, 添加你需要的项目即可*
