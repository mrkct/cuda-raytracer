# CUDA-Accelerated Raytracer

Simple path tracer that uses CUDA to accelerate the image rendering.
This was a project made for the course "GPU Computing" at UNIMI.

This is a raytracer based on the one presented in ["Raytracing in One Weekend"](https://raytracing.github.io/books/RayTracingInOneWeekend.html), except it uses the GPU to parallelize the computation of light rays. This raytracer uses less 1/10 of the time that the book's implementation takes to render the sample image at 1920x1080p with 256 samples per pixels.

A report (in Italian only) detailing how raytracing works, how to parallelize it via CUDA and a bunch of profiling stats is available here: [relazione.pdf](./relazione.pdf)

## How to use it

First you need to install the CUDA Toolkit. Once that is done and you are sure `nvcc` works correctly, run:

    make all
    ./rt -o out
    open out_000000000.png

This will render a scene with the default settings (640x480, 128 samples per pixel). You can change these settings with the appropriate flags `--width`, `--height` and `--samples`. 

The scene to render is hardcoded in the source, to change them you need to modify the line `create_X_scene` in `raytracer.cu` to call the appropriate function that will fill up the `struct Scene` object that will be used in the render. You can also change the camera position in the `render_scene` function call.

The code was made to allow generating multiple frames of the same scene while changing the camera position, this allows to create the frames and then assemble them into a video with `ffmpeg`. This is why the output filename have 9 numbers appended. After a while I realized the videos are not that impressive so I abbandoned that idea, even though the code is still there.

### How to use it without an NVIDIA GPU

Google Colab offers free GPU access, although you are forced to use it through a Jupyter notebook. It's not really comfortable, but it does work and it's how I developed this project.

Make a new notebook and execute this code in a cell:

    %cd /content
    !rm -rf cuda-raytracer
    !git clone https://github.com/mrkct/cuda-raytracer.git
    %cd cuda-raytracer
    !make all

    !./rt --width 1920 --height 1080 --samples 128 --output out

    from IPython.display import Image
    Image(filename='out_000000000.png')

## Sample renderings

![Render di Esempio](report/images/renders/planets.png)
![Render di Esempio](report/images/renders/floating_spheres.png)

## Come compilare ed eseguire
Per compilare ed eseguire il raytracer, assumendo che il toolkit CUDA sia
installato, basta eseguire:

    make all
    ./rt

Di default il progetto è compilato con la Compute Capability 7.5, per modificarla
basta andare nel Makefile e cambiare la variabile `NVCCFLAGS`.

Per chi non possiede una GPU NVIDIA è possibile usare Google Colab crendo una cella nel notebook ed eseguendo il seguente codice:

    %cd /content
    !rm -rf cuda-raytracer
    !git clone https://github.com/mrkct/cuda-raytracer.git
    %cd cuda-raytracer
    !make clean all

    !./rt

    from IPython.display import Image
    Image(filename='_000000000.png')
