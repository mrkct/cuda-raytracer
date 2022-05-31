NAME = rt
NVCC = nvcc
NVCCFLAGS = -arch=sm_37 --compiler-options -Wall --compiler-options -Wextra 

OBJS = \
	main.o \
	stb_image_write_impl.o \

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