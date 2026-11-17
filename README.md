# ASTRA N-BODY

ASTRA N-BODY is a three-dimensional gravitational simulation focused on parallel computing and GPU acceleration.

## Requirements

- Docker
- Docker Compose

## Build

```bash
docker compose build
```

## Run

```bash
docker compose up
```

## Custom simulation

```bash
docker compose run --rm astra-nbody \
  --particles 2048 \
  --steps 500 \
  --dt 0.001
```

## Available parameters

| Parameter   | Default | Description                   |
| ----------- | ------: | ----------------------------- |
| --particles |    1024 | Number of simulated particles |
| --steps     |     100 | Simulation iterations         |
| --dt        |   0.001 | Integration time step         |
| --softening |    0.01 | Gravitational softening       |
| --seed      |      42 | Random generator seed         |
| --help      |       - | Display usage information     |

## Architecture

- C++20
- Newtonian gravitational model
- Velocity Verlet integration
- CPU computation
- OpenMP parallel processing
- OpenCL kernel development
- CMake
- Docker

## License

MIT
