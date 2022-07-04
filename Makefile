NAME = rt
NVCC = nvcc
NVCCFLAGS = \
	-lineinfo -arch=sm_37 -I. \
	--compiler-options -Wall --compiler-options -Wextra \
	--maxrregcount 64 -Xptxas -v \
	-Wno-deprecated-gpu-targets \

OBJS = \
	camera.o \
	main.o \
	raytracer.o \
	geometry/sphere.o \
	materials/dielectric.o \
	materials/lambertian.o \
	materials/material.o \
	materials/metal.o \
	scenes/scene.o \
	scenes/single_sphere.o \
	scenes/planets.o \
	scenes/test_scene.o \
	textures/image_texture.o \
	textures/texture.o \
	textures/checkered_texture.o \
	textures/solid_texture.o \
	util/arg_parsing.o \
	util/check_cuda_errors.o \
	util/framebuffer.o \
	util/rng.o \
	util/stb_image_write_impl.o \
	util/time_measurement.o \
	util/vec3.o \

.PHONY = all clean

all: $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(NAME) $(OBJS)

test-run: all
	./$(NAME)

# Compile differently because it gives a bunch of warnings we don't care about
util/stb_image_write_impl.o: util/stb_image_write_impl.c
	$(CXX) -c -o util/stb_image_write_impl.o util/stb_image_write_impl.c

%.o: %.cu
	$(NVCC) -x cu $(NVCCFLAGS) -dc $< -o $@

clean:
	rm -f $(OBJS)
	rm -f $(NAME)