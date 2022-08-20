# utils
Here I put my utilisation functions together.

Compile and install, here assume that we have [doctest](https://github.com/doctest/doctest) for unit test installed at `UTILS_ROOT` (we also have `UTILS_ROOT` setup as an environment variable). `make install` installs all header files into `${UTILS_ROOT}/utils` (header only, so no libraries here).

1. `mkdir build`
2. `cmake ..`
3. `make all -j`
4. `make install`
