from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME


cdef Coord NULL_COORD = (-1, -1)

cdef double getTime():
    cdef timespec ts
    clock_gettime(CLOCK_REALTIME, &ts)
    return ts.tv_sec + (ts.tv_nsec / 1000000000.)