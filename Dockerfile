
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    make \
    libomp-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY CMakeLists.txt .
COPY include ./include
COPY src ./src
COPY kernels ./kernels
COPY tests ./tests

RUN cmake -S . -B build \
    -DCMAKE_GENERATOR="Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DASTRA_ENABLE_OPENMP=ON \
    -DASTRA_ENABLE_OPENCL=ON \
    -DASTRA_BUILD_TESTS=ON \
    && cmake --build build --parallel 4 \
    && ctest --test-dir build --output-on-failure \
    && cmake --install build --prefix /usr/local

FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV OCL_ICD_VENDORS=/etc/OpenCL/vendors
ENV RUSTICL_ENABLE=radeonsi

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ocl-icd-libopencl1 \
    mesa-opencl-icd \
    clinfo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/bin/astra-nbody /usr/local/bin/astra-nbody
COPY --from=builder /usr/local/share/astra-nbody/kernels /usr/local/share/astra-nbody/kernels

ENTRYPOINT ["astra-nbody"]
CMD ["--backend", "auto", "--particles", "1024", "--steps", "100"]
