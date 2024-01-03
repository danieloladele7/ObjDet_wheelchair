#!/bin/bash
read -p "Do you want to build Docker? (y/n): " build_docker

pwd_var=$(pwd)

# MUST enable jetson toolkit for host device: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
if [ "$build_docker" == "y" ]; then
    # BUILD DOCKER. Where: jetson_nano/docker:0.1 = container_name
    docker login
    # set privileges: https://docs.docker.com/build/building/multi-platform/ and https://github.com/tonistiigi/binfmt
    docker run --privileged --rm tonistiigi/binfmt --install all
    echo "binfmt has been installed"
    
    # display binfmt versions
    docker run --privileged --rm tonistiigi/binfmt --version
    
    # sometimes you might need to add credentials: https://github.com/docker/buildx/issues/1335
    # docker run --privileged multiarch/qemu-user-static:latest --reset -p yes --credential yes
    
    # create a new buildx. Its adviceable not to use the default builder: https://www.youtube.com/watch?v=hWSHtHasJUI
    docker buildx create --name jetsonbuilder
    echo "jetsonbuilder created"
    
    # use the created builder
    docker buildx use jetsonbuilder
    
    # inspect builder to make sure it is correctly configured. --bootstrap ensures that the driver is running before we inspect it
    docker buildx inspect --bootstrap
	
    # ensure to login and create the repo before the next step
	docker buildx build \
        --platform linux/amd64,linux/arm64,linux/arm/v7 \
        -t dannyola/nvcr.io-nano-multi-arch:0.1 \
        -f DockerFile.JetsonNano \
        --push\
        .
    # inspect image:
    docker buildx imagetools inspect dannyola/nvcr.io-nano-multi-arch:0.1
    exit 0
fi

# jetson_nano/docker:0.1
read -p "Run Docker on Windows or Linux? (W/L): " run_docker

if [ "$run_docker" == "W" ]; then
    # set privileges: https://docs.docker.com/build/building/multi-platform/ and https://github.com/tonistiigi/binfmt
    docker run --privileged --rm tonistiigi/binfmt --install all
    echo "binfmt has been installed"

    docker run -it --rm --name jetson_nano \
        --mount type=bind,source="$pwd_var",target=/src \
        dannyola/nvcr.io-nano-multi-arch:0.1 \
        bash
fi

if [ "$run_docker" == "L" ]; then
    # set privileges: https://docs.docker.com/build/building/multi-platform/ and https://github.com/tonistiigi/binfmt
    docker run --privileged --rm tonistiigi/binfmt --install all
    echo "binfmt has been installed"

    docker run -it --rm --name jetson_nano \
        --mount type=bind,source="$pwd_var",target=/src \
		--env="DISPLAY" \
		--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        dannyola/nvcr.io-nano-multi-arch:0.1 \
        bash
fi

# sudo docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix nvcr.io/nvidia/l4t-tensorrt:r8.0.1-runtime