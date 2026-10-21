// Python ABI only. The four APP2 headers beside this file are unmodified.
#include <cmath>
#include <cstdint>
#include <exception>
#include <mutex>
#include <streambuf>
#include "fastmarching_dt.h"

class QuietOutput {
    class Sink : public std::streambuf {
        int overflow(int c) override { return traits_type::not_eof(c); }
    } sink;
    std::streambuf* previous;
public:
    QuietOutput() : previous(std::cout.rdbuf(&sink)) {}
    ~QuietOutput() { std::cout.rdbuf(previous); }
};

extern "C" int app2_gsdt(const void* input, float* output, int x, int y, int z,
                          int dtype, int connectivity, int threshold) {
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    QuietOutput quiet;
    try {
        float* phi = output; // APP2 writes directly into the caller's array.
        bool ok = false;
        if (dtype == 1)
            ok = fastmarching_dt((uint8_t*)input, phi, x, y, z, connectivity, threshold);
        else if (dtype == 2)
            ok = fastmarching_dt((uint16_t*)input, phi, x, y, z, connectivity, threshold);
        else if (dtype == 4)
            ok = fastmarching_dt((float*)input, phi, x, y, z, connectivity, threshold);
        return ok ? 0 : 1;
    } catch (const std::exception&) { return 2; }
    catch (...) { return 3; }
}
