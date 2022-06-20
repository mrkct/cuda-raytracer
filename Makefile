NAME = rt
NVCC = nvcc
NVCCFLAGS = -arch=sm_37 -I. --compiler-options -Wall --compiler-options -Wextra -Wno-deprecated-gpu-targets

OBJS = \
	stb_image_write_impl.o \
	main.o \
	raytracer/util/CudaHelpers.o \
	raytracer/Raytracer.o \
	raytracer/geometry/Sphere.o \
	raytracer/Camera.o \
	raytracer/HittableList.o \
	raytracer/util/DeviceRNG.o \
	raytracer/material/Lambertian.o \
	raytracer/material/Metal.o \
	raytracer/material/Dielectric.o \

.PHONY = all clean

all: $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(NAME) $(OBJS)

test-run: all
	./$(NAME)

# Compile differently because it gives a bunch of warnings we don't care about
stb_image_write_impl.o: stb_image_write_impl.cpp
	$(CXX) -c -o stb_image_write_impl.o stb_image_write_impl.cpp

%.o: %.cpp
	$(NVCC) -x cu $(NVCCFLAGS) -dc $< -o $@

clean:
	rm -f $(OBJS)
	rm -f $(NAME)