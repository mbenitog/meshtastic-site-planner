/* GPU-accelerated ultra CLI (OpenCL).
 *
 * Drop-in replacement for engine/native/ultra_main.cpp that runs the
 * same per-cell ITM computation on an OpenCL device instead of CPU
 * threads.  Produces byte-identical output files (signal_i16le.bin,
 * mask_u8.bin, meta.json) so backend/ultra/tiler.py and runner.py work
 * unchanged — just point ULTRA_CLI at this binary.
 *
 * Design (per-cell GPU, "Option D"):
 *   - One work-item per output cell.
 *   - Profile is built on-device by sampling the int16 surface grid
 *     along the TX→cell line (same sampling as ultra_main run_itm_path).
 *   - Profile + d1thx scratch live in __global memory (pre-allocated
 *     pools, one entry per work-item in the batch).  Bounded by a
 *     memory budget so 30 km jobs don't exhaust VRAM.
 *   - FP64 by default (bit-exact vs CPU), FP32 fallback on devices
 *     without cl_khr_fp64.
 *   - Tiles + checkpointing preserved: one process per tile, same CLI
 *     interface, same output layout.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <string>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#ifndef _WIN32
#include <sys/mman.h>
#endif

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define D1THX_SCRATCH_LEN 256

/* ---- OpenCL error helper ---- */

static const char *cl_err(cl_int e) {
    switch (e) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
    default: return "<unknown>";
    }
}
#define CL_CHECK(expr) do { cl_int _e = (expr); if (_e != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL error %s at %s:%d\n", cl_err(_e), __FILE__, __LINE__); \
    exit(1); } } while (0)

/* ---- CLI arg parsing (mirrors ultra_main.cpp) ---- */

static double arg_f(int argc, char **argv, const char *name, bool *found) {
    for (int i = 1; i + 1 < argc; i++)
        if (strcmp(argv[i], name) == 0) {
            if (found) *found = true;
            return atof(argv[i + 1]);
        }
    if (found) *found = false;
    return 0.0;
}
static const char *arg_s(int argc, char **argv, const char *name) {
    for (int i = 1; i + 1 < argc; i++)
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    return nullptr;
}
static int arg_i(int argc, char **argv, const char *name, bool *found) {
    for (int i = 1; i + 1 < argc; i++)
        if (strcmp(argv[i], name) == 0) {
            if (found) *found = true;
            return (int)atof(argv[i + 1]);
        }
    if (found) *found = false;
    return 0;
}
static bool require(bool found, const char *name) {
    if (!found) fprintf(stderr, "missing required argument %s\n", name);
    return found;
}
static inline int clamp_i(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ---- device selection ---- */

static int dev_has_ext(cl_device_id d, const char *name) {
    char buf[16384]; size_t n = 0;
    if (clGetDeviceInfo(d, CL_DEVICE_EXTENSIONS, sizeof(buf), buf, &n) != CL_SUCCESS) return 0;
    buf[n < sizeof(buf) ? n : sizeof(buf) - 1] = '\0';
    for (char *p = buf; *p; ) {
        char *e = p; while (*e && *e != ' ') e++;
        char sv = *e; *e = '\0';
        if (strcmp(p, name) == 0) { *e = sv; return 1; }
        *e = sv; p = (*e == '\0') ? e : e + 1;
    }
    return 0;
}

struct DevicePick {
    cl_platform_id platform;
    cl_device_id   device;
    cl_bool        has_fp64;
    cl_device_type type;
    char           name[256];
    cl_uint        compute_units;
};

static int pick_device(int forced_plat, int forced_dev, DevicePick *out) {
    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, NULL, &nplat) != CL_SUCCESS || nplat == 0) {
        fprintf(stderr, "opencl: no platforms found\n"); return 1;
    }
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), NULL);

    DevicePick best; bool have_best = false; int best_score = -1;
    for (cl_uint pi = 0; pi < nplat; pi++) {
        cl_uint ndev = 0;
        if (clGetDeviceIDs(plats[pi], CL_DEVICE_TYPE_ALL, 0, NULL, &ndev) != CL_SUCCESS || ndev == 0)
            continue;
        std::vector<cl_device_id> devs(ndev);
        clGetDeviceIDs(plats[pi], CL_DEVICE_TYPE_ALL, ndev, devs.data(), NULL);
        for (cl_uint di = 0; di < ndev; di++) {
            if (forced_plat >= 0 && (int)pi != forced_plat) continue;
            if (forced_dev >= 0 && (int)di != forced_dev) continue;
            DevicePick c;
            c.platform = plats[pi]; c.device = devs[di];
            cl_device_type t = 0;
            clGetDeviceInfo(devs[di], CL_DEVICE_TYPE, sizeof(t), &t, NULL);
            c.type = t;
            c.has_fp64 = dev_has_ext(devs[di], "cl_khr_fp64") ? CL_TRUE : CL_FALSE;
            clGetDeviceInfo(devs[di], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(c.compute_units), &c.compute_units, NULL);
            size_t nl = 0;
            clGetDeviceInfo(devs[di], CL_DEVICE_NAME, sizeof(c.name), c.name, &nl);
            c.name[nl < sizeof(c.name) ? nl : sizeof(c.name)-1] = '\0';
            int score = 0;
            if (t == CL_DEVICE_TYPE_GPU) score += 100;
            if (c.has_fp64) score += 10;
            score += (int)c.compute_units;
            if (!have_best || score > best_score) { best = c; best_score = score; have_best = true; }
        }
    }
    if (!have_best) { fprintf(stderr, "opencl: no matching device\n"); return 1; }
    *out = best;
    return 0;
}

/* ---- load kernel source ---- */

static std::vector<char> load_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);
    std::vector<char> buf(sz + 1);
    fread(buf.data(), 1, sz, fp); buf[sz] = '\0'; fclose(fp);
    return buf;
}

/* ---- helper to set real_t kernel args in the right precision ---- */

#define SET_REAL(kern, idx, val, use_double) do { \
    if (use_double) { double _v = (double)(val); CL_CHECK(clSetKernelArg(kern, idx, sizeof(_v), &_v)); } \
    else { float _v = (float)(val); CL_CHECK(clSetKernelArg(kern, idx, sizeof(_v), &_v)); } \
} while (0)

/* ---- main ---- */

int main(int argc, char **argv) {
    bool ok = true, f;

    /* OpenCL-specific args */
    int forced_plat = arg_i(argc, argv, "--platform", &f); if (!f) forced_plat = -1;
    int forced_dev  = arg_i(argc, argv, "--device", &f);   if (!f) forced_dev = -1;
    const char *kernel_path = arg_s(argc, argv, "--kernel");
    if (!kernel_path) kernel_path = "engine/opencl/itm_kernel.cl";
    bool force_fp32 = false;
    for (int i = 1; i < argc; i++) if (strcmp(argv[i], "--fp32") == 0) force_fp32 = true;
    bool cpu_profiles = false;
    for (int i = 1; i < argc; i++) if (strcmp(argv[i], "--cpu-profiles") == 0) cpu_profiles = true;

    const char *env_plat = getenv("ULTRA_OPENCL_PLATFORM");
    const char *env_dev  = getenv("ULTRA_OPENCL_DEVICE");
    if (env_plat) forced_plat = atoi(env_plat);
    if (env_dev)  forced_dev  = atoi(env_dev);

    bool probe_only = false;
    for (int i = 1; i < argc; i++) if (strcmp(argv[i], "--probe") == 0) probe_only = true;
    if (probe_only) {
        cl_uint nplat = 0;
        if (clGetPlatformIDs(0, NULL, &nplat) != CL_SUCCESS || nplat == 0) {
            printf("{\"status\":\"error\",\"message\":\"no OpenCL platforms found\"}\n");
            return 1;
        }
        std::vector<cl_platform_id> plats(nplat);
        clGetPlatformIDs(nplat, plats.data(), NULL);

        struct DevInfo {
            int platform_idx;
            int device_idx;
            char name[256];
            const char *type_str;
            cl_uint compute_units;
            bool has_fp64;
            cl_ulong gmem;
            cl_ulong max_alloc;
            int score;
        };
        std::vector<DevInfo> found_devs;
        int best_idx = -1;
        int best_score = -1;

        for (cl_uint pi = 0; pi < nplat; pi++) {
            cl_uint ndev = 0;
            if (clGetDeviceIDs(plats[pi], CL_DEVICE_TYPE_ALL, 0, NULL, &ndev) != CL_SUCCESS || ndev == 0)
                continue;
            std::vector<cl_device_id> devs(ndev);
            clGetDeviceIDs(plats[pi], CL_DEVICE_TYPE_ALL, ndev, devs.data(), NULL);
            for (cl_uint di = 0; di < ndev; di++) {
                if (forced_plat >= 0 && (int)pi != forced_plat) continue;
                if (forced_dev >= 0 && (int)di != forced_dev) continue;

                DevInfo d;
                d.platform_idx = (int)pi;
                d.device_idx = (int)di;
                cl_device_type t = 0;
                clGetDeviceInfo(devs[di], CL_DEVICE_TYPE, sizeof(t), &t, NULL);
                d.type_str = (t == CL_DEVICE_TYPE_GPU) ? "GPU" : ((t == CL_DEVICE_TYPE_CPU) ? "CPU" : "ACCELERATOR");
                d.has_fp64 = dev_has_ext(devs[di], "cl_khr_fp64") ? true : false;
                clGetDeviceInfo(devs[di], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(d.compute_units), &d.compute_units, NULL);
                size_t nl = 0;
                clGetDeviceInfo(devs[di], CL_DEVICE_NAME, sizeof(d.name), d.name, &nl);
                d.name[nl < sizeof(d.name) ? nl : sizeof(d.name) - 1] = '\0';
                clGetDeviceInfo(devs[di], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(d.gmem), &d.gmem, NULL);
                clGetDeviceInfo(devs[di], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(d.max_alloc), &d.max_alloc, NULL);

                int score = 0;
                if (t == CL_DEVICE_TYPE_GPU) score += 100;
                if (d.has_fp64) score += 10;
                score += (int)d.compute_units;
                d.score = score;

                if (best_idx < 0 || score > best_score) {
                    best_score = score;
                    best_idx = (int)found_devs.size();
                }
                found_devs.push_back(d);
            }
        }

        if (found_devs.empty()) {
            printf("{\"status\":\"error\",\"message\":\"no matching OpenCL devices found\"}\n");
            return 1;
        }

        printf("{\"status\":\"ok\",\"devices\":[");
        for (size_t i = 0; i < found_devs.size(); i++) {
            const auto &d = found_devs[i];
            printf("{\"platform\":%d,\"device\":%d,\"name\":\"%s\",\"type\":\"%s\",\"compute_units\":%u,\"has_fp64\":%s,\"vram_bytes\":%llu,\"vram_mb\":%llu,\"max_alloc_mb\":%llu}%s",
                   d.platform_idx, d.device_idx, d.name, d.type_str, d.compute_units,
                   d.has_fp64 ? "true" : "false",
                   (unsigned long long)d.gmem,
                   (unsigned long long)(d.gmem / (1024 * 1024)),
                   (unsigned long long)(d.max_alloc / (1024 * 1024)),
                   (i + 1 < found_devs.size()) ? "," : "");
        }
        const auto &best = found_devs[best_idx];
        printf("],\"platform\":%d,\"device_index\":%d,\"device\":\"%s\",\"type\":\"%s\",\"compute_units\":%u,\"has_fp64\":%s,\"vram_bytes\":%llu,\"vram_mb\":%llu,\"max_alloc_mb\":%llu}\n",
               best.platform_idx, best.device_idx, best.name, best.type_str, best.compute_units,
               best.has_fp64 ? "true" : "false",
               (unsigned long long)best.gmem,
               (unsigned long long)(best.gmem / (1024 * 1024)),
               (unsigned long long)(best.max_alloc / (1024 * 1024)));
        return 0;
    }

    /* Standard ultra_cli args (same as ultra_main.cpp) */
    const char *surface_path = arg_s(argc, argv, "--surface");
    ok &= require(surface_path != nullptr, "--surface");
    const char *out_prefix = arg_s(argc, argv, "--out");
    ok &= require(out_prefix != nullptr, "--out");
    int width  = (int)arg_f(argc, argv, "--width", &f);  ok &= require(f, "--width");
    int height = (int)arg_f(argc, argv, "--height", &f); ok &= require(f, "--height");
    int tile_x0 = arg_i(argc, argv, "--tile-x0", &f);
    int tile_y0 = arg_i(argc, argv, "--tile-y0", &f);
    int tile_w = arg_i(argc, argv, "--tile-w", &f);
    int tile_h = arg_i(argc, argv, "--tile-h", &f);
    if (!f) tile_w = width - tile_x0;
    if (!f) tile_h = height - tile_y0;
    if (tile_x0 < 0 || tile_y0 < 0 || tile_w <= 0 || tile_h <= 0 ||
        tile_x0 + tile_w > width || tile_y0 + tile_h > height) {
        fprintf(stderr, "invalid tile bounds\n"); return 2;
    }
    double min_x = arg_f(argc, argv, "--min-x", &f);      ok &= require(f, "--min-x");
    double max_y = arg_f(argc, argv, "--max-y", &f);      ok &= require(f, "--max-y");
    double resolution = arg_f(argc, argv, "--resolution-m", &f); ok &= require(f, "--resolution-m");
    double tx_x = arg_f(argc, argv, "--tx-x", &f);        ok &= require(f, "--tx-x");
    double tx_y = arg_f(argc, argv, "--tx-y", &f);        ok &= require(f, "--tx-y");
    double tx_height = arg_f(argc, argv, "--tx-height-m", &f);  ok &= require(f, "--tx-height-m");
    double rx_height = arg_f(argc, argv, "--rx-height-m", &f);  ok &= require(f, "--rx-height-m");
    double freq = arg_f(argc, argv, "--freq-mhz", &f);    ok &= require(f, "--freq-mhz");
    double tx_power_w = arg_f(argc, argv, "--tx-power-w", &f);  ok &= require(f, "--tx-power-w");
    double tx_gain = arg_f(argc, argv, "--tx-gain-dbi", &f);    ok &= require(f, "--tx-gain-dbi");
    double rx_gain = arg_f(argc, argv, "--rx-gain-dbi", &f);    ok &= require(f, "--rx-gain-dbi");
    double rx_sensitivity = arg_f(argc, argv, "--rx-sensitivity-dbm", &f); ok &= require(f, "--rx-sensitivity-dbm");
    double eps_dielect = arg_f(argc, argv, "--dielect", &f);    if (!f) eps_dielect = 15.0;
    double conductivity = arg_f(argc, argv, "--conductivity", &f); if (!f) conductivity = 0.005;
    double bend = arg_f(argc, argv, "--bend", &f);              if (!f) bend = 301.0;
    int climate = (int)arg_f(argc, argv, "--climate", &f);      if (!f) climate = 5;
    int polarization = (int)arg_f(argc, argv, "--pol", &f);     if (!f) polarization = 1;
    double conf = arg_f(argc, argv, "--conf", &f);              if (!f) conf = 0.95;
    double rel = arg_f(argc, argv, "--rel", &f);                if (!f) rel = 0.95;
    if (!ok || width <= 0 || height <= 0 || resolution <= 0 || freq <= 0 || tx_power_w <= 0)
        return 2;

    int tx_col = clamp_i((int)llround((tx_x - min_x) / resolution), 0, width - 1);
    int tx_row = clamp_i((int)llround((max_y - tx_y) / resolution), 0, height - 1);

    /* ---- pick device ---- */
    DevicePick pick;
    if (pick_device(forced_plat, forced_dev, &pick) != 0) return 1;
    bool use_double = pick.has_fp64 && !force_fp32;
    if (!use_double && !force_fp32 && !pick.has_fp64) {
        /* no fp64 available, using fp32 */
    }
    fprintf(stderr, "opencl device: %s  fp64=%s  using=%s\n", pick.name,
            pick.has_fp64 ? "yes" : "no", use_double ? "FP64" : "FP32");

    size_t elem_size = use_double ? sizeof(double) : sizeof(float);

    cl_ulong global_mem = 0;
    clGetDeviceInfo(pick.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
    cl_ulong max_alloc = 0;
    clGetDeviceInfo(pick.device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);

    /* ---- surface loading: direct full load or adaptive subgrid ---- */
    size_t full_surf_bytes = (size_t)width * (size_t)height * sizeof(int16_t);
    size_t safe_max_surf_alloc = max_alloc > 0 ? (size_t)((double)max_alloc * 0.85) : (size_t)1024 * 1024 * 1024;
    size_t safe_vram_surf_limit = global_mem > 0 ? (size_t)((double)global_mem * 0.45) : (size_t)1024 * 1024 * 1024;
    size_t max_direct_surf_bytes = std::min(safe_max_surf_alloc, safe_vram_surf_limit);

    int crop_x0 = 0, crop_y0 = 0, crop_w = width, crop_h = height;
    bool needs_crop = (full_surf_bytes > max_direct_surf_bytes);

    if (needs_crop) {
        crop_x0 = std::max(0, std::min(tx_col, tile_x0) - 2);
        int crop_x1 = std::min(width - 1, std::max(tx_col, tile_x0 + tile_w - 1) + 2);
        crop_y0 = std::max(0, std::min(tx_row, tile_y0) - 2);
        int crop_y1 = std::min(height - 1, std::max(tx_row, tile_y0 + tile_h - 1) + 2);
        crop_w = crop_x1 - crop_x0 + 1;
        crop_h = crop_y1 - crop_y0 + 1;
    }

    int rel_tx_col = tx_col - crop_x0;
    int rel_tx_row = tx_row - crop_y0;
    int rel_tile_x0 = tile_x0 - crop_x0;
    int rel_tile_y0 = tile_y0 - crop_y0;

    size_t surf_cells = (size_t)crop_w * (size_t)crop_h;
    std::vector<int16_t> surface(surf_cells);

    bool loaded = false;
#ifndef _WIN32
    int fd = open(surface_path, O_RDONLY);
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && (size_t)st.st_size >= full_surf_bytes) {
            void *mapped = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
            if (mapped != MAP_FAILED) {
                const int16_t *src_grid = (const int16_t *)mapped;
                if (!needs_crop) {
                    memcpy(surface.data(), src_grid, full_surf_bytes);
                } else {
                    for (int r = 0; r < crop_h; r++) {
                        memcpy(surface.data() + (size_t)r * (size_t)crop_w,
                               src_grid + (size_t)(crop_y0 + r) * (size_t)width + crop_x0,
                               (size_t)crop_w * sizeof(int16_t));
                    }
                }
                munmap(mapped, st.st_size);
                loaded = true;
            }
        }
        close(fd);
    }
#endif
    if (!loaded) {
        FILE *fp = fopen(surface_path, "rb");
        if (!fp) { fprintf(stderr, "failed to open %s\n", surface_path); return 1; }
        if (!needs_crop) {
            size_t nrd = fread(surface.data(), sizeof(int16_t), surf_cells, fp);
            fclose(fp);
            if (nrd != surf_cells) { fprintf(stderr, "short read from %s\n", surface_path); return 1; }
        } else {
            for (int r = 0; r < crop_h; r++) {
                off_t offset = ((off_t)(crop_y0 + r) * (off_t)width + crop_x0) * (off_t)sizeof(int16_t);
                if (fseeko(fp, offset, SEEK_SET) != 0 ||
                    fread(surface.data() + (size_t)r * (size_t)crop_w, sizeof(int16_t), crop_w, fp) != (size_t)crop_w) {
                    fclose(fp);
                    fprintf(stderr, "read error on %s\n", surface_path);
                    return 1;
                }
            }
            fclose(fp);
        }
    }

    /* ---- compute tile geometry ---- */
    int total_cells = tile_w * tile_h;

    /* Max profile length for this tile: farthest cell from TX */
    int corners[4][2] = {
        {tile_x0, tile_y0},
        {tile_x0 + tile_w - 1, tile_y0},
        {tile_x0, tile_y0 + tile_h - 1},
        {tile_x0 + tile_w - 1, tile_y0 + tile_h - 1},
    };
    double max_dist_cells = 0;
    for (auto &c : corners) {
        double dx = c[0] - tx_col, dy = c[1] - tx_row;
        max_dist_cells = std::max(max_dist_cells, sqrt(dx*dx + dy*dy));
    }
    int max_segments = std::max(1, (int)ceil(max_dist_cells));
    int stride_prof = max_segments + 16;

    cl_int e;
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)pick.platform, 0
    };
    cl_context ctx = clCreateContext(props, 1, &pick.device, NULL, NULL, &e);
    CL_CHECK(e);
    cl_command_queue q = clCreateCommandQueue(ctx, pick.device, 0, &e);
    CL_CHECK(e);

    const char *build_opts = use_double ? "-DUSE_DOUBLE" : "";
    cl_program prog = NULL;

    /* Cache precompiled device binary in /tmp with atomic writes to eliminate
     * JIT compilation and driver shader-cache lock contention. */
    char bin_path[256];
    snprintf(bin_path, sizeof(bin_path), "/tmp/ultra_cl_%s_%s.bin",
             use_double ? "fp64" : "fp32", pick.name);
    for (char *p = bin_path; *p; p++) {
        if (*p == ' ' || *p == '/' || *p == ':' || *p == '(' || *p == ')') *p = '_';
    }

    FILE *fbin = fopen(bin_path, "rb");
    if (fbin) {
        fseek(fbin, 0, SEEK_END);
        long bsz_l = ftell(fbin);
        fseek(fbin, 0, SEEK_SET);
        if (bsz_l > 0) {
            size_t bsz = (size_t)bsz_l;
            std::vector<unsigned char> bdata(bsz);
            if (fread(bdata.data(), 1, bsz, fbin) == bsz) {
                const unsigned char *bptr = bdata.data();
                cl_int bstatus = CL_SUCCESS;
                prog = clCreateProgramWithBinary(ctx, 1, &pick.device, &bsz, &bptr, &bstatus, &e);
                if (e == CL_SUCCESS && bstatus == CL_SUCCESS && prog) {
                    e = clBuildProgram(prog, 1, &pick.device, build_opts, NULL, NULL);
                    if (e != CL_SUCCESS) {
                        clReleaseProgram(prog);
                        prog = NULL;
                    }
                } else if (prog) {
                    clReleaseProgram(prog);
                    prog = NULL;
                }
            }
        }
        fclose(fbin);
    }

    if (!prog) {
        std::vector<char> ksrc = load_file(kernel_path);
        const char *srcs[1] = { ksrc.data() };
        size_t lens[1] = { ksrc.size() - 1 };
        prog = clCreateProgramWithSource(ctx, 1, srcs, lens, &e);
        CL_CHECK(e);
        e = clBuildProgram(prog, 1, &pick.device, build_opts, NULL, NULL);
        if (e != CL_SUCCESS) {
            fprintf(stderr, "kernel build failed: %s\n", cl_err(e));
            size_t loglen = 0;
            clGetProgramBuildInfo(prog, pick.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &loglen);
            std::vector<char> log(loglen + 1);
            clGetProgramBuildInfo(prog, pick.device, CL_PROGRAM_BUILD_LOG, loglen, log.data(), NULL);
            log[loglen] = '\0';
            fprintf(stderr, "---- build log ----\n%s\n------------------\n", log.data());
            return 1;
        }
        size_t bsz = 0;
        if (clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bsz, NULL) == CL_SUCCESS && bsz > 0) {
            std::vector<unsigned char> bdata(bsz);
            unsigned char *bptr = bdata.data();
            if (clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &bptr, NULL) == CL_SUCCESS) {
                char tmp_bin_path[280];
                snprintf(tmp_bin_path, sizeof(tmp_bin_path), "%s.%d.tmp", bin_path, (int)getpid());
                FILE *fw = fopen(tmp_bin_path, "wb");
                if (fw) {
                    fwrite(bdata.data(), 1, bsz, fw);
                    fclose(fw);
                    rename(tmp_bin_path, bin_path);
                }
            }
        }
    }
    cl_kernel kern_build = clCreateKernel(prog, "build_profiles_kernel", &e);
    CL_CHECK(e);
    cl_kernel kern_itm = clCreateKernel(prog, "itm_p2p_kernel", &e);
    CL_CHECK(e);

    /* ---- work-group sizing ---- */
    size_t wg_max;
    CL_CHECK(clGetKernelWorkGroupInfo(kern_itm, pick.device, CL_KERNEL_WORK_GROUP_SIZE,
                                      sizeof(wg_max), &wg_max, NULL));
    size_t wg_mult = 64;
#ifdef CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
    CL_CHECK(clGetKernelWorkGroupInfo(kern_itm, pick.device,
                CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                sizeof(wg_mult), &wg_mult, NULL));
#endif
    size_t local = wg_mult;
    if (local > wg_max) local = wg_max;
    if (local < 1) local = 1;

    /* ---- compute VRAM budget and batch sizing ---- */
    double vram_frac = 0.85;
    const char *env_frac = getenv("ULTRA_VRAM_FRACTION");
    if (env_frac) {
        double f = atof(env_frac);
        if (f > 0.10 && f <= 0.98) vram_frac = f;
    }

    size_t surf_bytes = surf_cells * sizeof(int16_t);
    size_t fixed_bytes = surf_bytes;

    size_t mem_budget = (size_t)512 * 1024 * 1024;
    if (global_mem > 0) {
        size_t target_total = (size_t)((double)global_mem * vram_frac);
        if (target_total > fixed_bytes + 64 * 1024 * 1024) {
            mem_budget = target_total - fixed_bytes;
        } else {
            mem_budget = (size_t)256 * 1024 * 1024;
        }
    }
    const char *env_budget = getenv("ULTRA_MEM_BUDGET_MB");
    if (env_budget) {
        long mb = atol(env_budget);
        if (mb > 32) mem_budget = (size_t)mb * 1024 * 1024;
    }

    /* ---- batch sizing (bound VRAM for profile + scratch pools) ---- */
    size_t bytes_per_item = (size_t)(stride_prof + D1THX_SCRATCH_LEN + 2) * elem_size;
    int batch_size = (int)(mem_budget / bytes_per_item);

    /* OpenCL single buffer constraint:
     * Never request a single buffer (especially d_prof) that exceeds
     * CL_DEVICE_MAX_MEM_ALLOC_SIZE (often ~2GB to 4GB even on 24GB+ cards).
     * We cap d_prof to 85% of max_alloc for safe driver headroom. */
    if (max_alloc > 0) {
        size_t safe_max_alloc = (size_t)((double)max_alloc * 0.85);
        int max_prof_items = (int)(safe_max_alloc / ((size_t)stride_prof * elem_size));
        if (max_prof_items > 0 && batch_size > max_prof_items) {
            batch_size = max_prof_items;
        }
    }

    cl_uint compute_units = 1;
    clGetDeviceInfo(pick.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    if (compute_units < 1) compute_units = 1;

    /* Optional user batch size cap */
    const char *env_max_batch = getenv("ULTRA_MAX_BATCH_SIZE");
    if (env_max_batch) {
        int mb = atoi(env_max_batch);
        if (mb >= 1024 && batch_size > mb) batch_size = mb;
    }

    batch_size = std::min(batch_size, total_cells);
    /* round down to multiple of local */
    batch_size = (int)(((size_t)batch_size / local) * local);
    if (batch_size < 1) batch_size = 1;

    size_t alloc_batch = ((size_t)batch_size + local - 1) / local * local;
    if (alloc_batch < local) alloc_batch = local;

    fprintf(stderr, "device: %s  vram=%lluMB  budget=%lluMB (%.0f%%)  max_alloc=%lluMB  surf=%dx%d (%lluMB%s)\n",
            pick.name, (unsigned long long)(global_mem / (1024 * 1024)),
            (unsigned long long)(mem_budget / (1024 * 1024)), vram_frac * 100.0,
            (unsigned long long)(max_alloc / (1024 * 1024)),
            crop_w, crop_h, (unsigned long long)(surf_bytes / (1024 * 1024)),
            needs_crop ? ", cropped" : ", full");
    fprintf(stderr, "tile: x0=%d y0=%d w=%d h=%d  cells=%d  max_segs=%d  stride=%d  batch=%d  alloc_batch=%zu  wg=%zu\n",
            tile_x0, tile_y0, tile_w, tile_h, total_cells, max_segments, stride_prof, batch_size, alloc_batch, local);

    /* ---- allocate GPU buffers ---- */
    cl_mem d_surf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   surf_cells * sizeof(int16_t), surface.data(), &e);
    CL_CHECK(e);

    cl_mem d_prof = NULL;
    cl_mem d_scratch = NULL;
    cl_mem d_loss = NULL;
    cl_mem d_err = NULL;

    auto free_batch_buffers = [&]() {
        if (d_prof) { clReleaseMemObject(d_prof); d_prof = NULL; }
        if (d_scratch) { clReleaseMemObject(d_scratch); d_scratch = NULL; }
        if (d_loss) { clReleaseMemObject(d_loss); d_loss = NULL; }
        if (d_err) { clReleaseMemObject(d_err); d_err = NULL; }
    };

    auto recreate_queue = [&]() -> bool {
        if (q) {
            clFinish(q);
            clReleaseCommandQueue(q);
            q = NULL;
        }
        cl_int qerr = CL_SUCCESS;
        q = clCreateCommandQueue(ctx, pick.device, 0, &qerr);
        return qerr == CL_SUCCESS;
    };

    auto alloc_batch_buffers = [&](size_t abatch) -> bool {
        free_batch_buffers();
        cl_int err = CL_SUCCESS;
        d_prof = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                abatch * stride_prof * elem_size, NULL, &err);
        if (err != CL_SUCCESS) { free_batch_buffers(); return false; }
        d_scratch = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                  abatch * D1THX_SCRATCH_LEN * elem_size, NULL, &err);
        if (err != CL_SUCCESS) { free_batch_buffers(); return false; }
        d_loss = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                abatch * elem_size, NULL, &err);
        if (err != CL_SUCCESS) { free_batch_buffers(); return false; }
        d_err = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                              abatch * sizeof(int), NULL, &err);
        if (err != CL_SUCCESS) { free_batch_buffers(); return false; }

        clSetKernelArg(kern_build, 13, sizeof(cl_mem), &d_prof);
        clSetKernelArg(kern_itm, 0, sizeof(cl_mem), &d_prof);
        clSetKernelArg(kern_itm, 2, sizeof(cl_mem), &d_scratch);
        clSetKernelArg(kern_itm, 13, sizeof(cl_mem), &d_loss);
        clSetKernelArg(kern_itm, 14, sizeof(cl_mem), &d_err);
        return true;
    };

    while (!alloc_batch_buffers(alloc_batch) && batch_size > (int)local) {
        batch_size = (int)(((size_t)(batch_size / 2) / local) * local);
        if (batch_size < (int)local) batch_size = (int)local;
        alloc_batch = ((size_t)batch_size + local - 1) / local * local;
    }
    if (!d_prof) {
        fprintf(stderr, "failed to allocate OpenCL batch buffers on %s\n", pick.name);
        return 1;
    }

    /* ---- set static args for build_profiles_kernel ---- */
    /* 0: surface, 1: surf_w, 2: surf_h, 3: tx_col, 4: tx_row, 5: resolution,
       6: tile_x0, 7: tile_y0, 8: tile_w, 9: tile_h,
       10: batch_start, 11: batch_count, 12: total_cells,
       13: prof_pool, 14: stride_prof */
    CL_CHECK(clSetKernelArg(kern_build, 0, sizeof(cl_mem), &d_surf));
    CL_CHECK(clSetKernelArg(kern_build, 1, sizeof(int), &crop_w));
    CL_CHECK(clSetKernelArg(kern_build, 2, sizeof(int), &crop_h));
    CL_CHECK(clSetKernelArg(kern_build, 3, sizeof(int), &rel_tx_col));
    CL_CHECK(clSetKernelArg(kern_build, 4, sizeof(int), &rel_tx_row));
    SET_REAL(kern_build, 5, resolution, use_double);
    CL_CHECK(clSetKernelArg(kern_build, 6, sizeof(int), &rel_tile_x0));
    CL_CHECK(clSetKernelArg(kern_build, 7, sizeof(int), &rel_tile_y0));
    CL_CHECK(clSetKernelArg(kern_build, 8, sizeof(int), &tile_w));
    CL_CHECK(clSetKernelArg(kern_build, 9, sizeof(int), &tile_h));
    /* 10-12 set per batch */
    CL_CHECK(clSetKernelArg(kern_build, 13, sizeof(cl_mem), &d_prof));
    CL_CHECK(clSetKernelArg(kern_build, 14, sizeof(int), &stride_prof));

    /* ---- set static args for itm_p2p_kernel (validated) ---- */
    /* 0: profiles, 1: stride, 2: scratch,
       3: tht_m, 4: rht_m, 5: eps, 6: sgm, 7: eno, 8: frq,
       9: radio_climate, 10: pol, 11: conf, 12: rel,
       13: out_dbloss, 14: out_errnum */
    CL_CHECK(clSetKernelArg(kern_itm, 0, sizeof(cl_mem), &d_prof));
    CL_CHECK(clSetKernelArg(kern_itm, 1, sizeof(int), &stride_prof));
    CL_CHECK(clSetKernelArg(kern_itm, 2, sizeof(cl_mem), &d_scratch));
    SET_REAL(kern_itm, 3, tx_height, use_double);
    SET_REAL(kern_itm, 4, rx_height, use_double);
    SET_REAL(kern_itm, 5, eps_dielect, use_double);
    SET_REAL(kern_itm, 6, conductivity, use_double);
    SET_REAL(kern_itm, 7, bend, use_double);
    SET_REAL(kern_itm, 8, freq, use_double);
    CL_CHECK(clSetKernelArg(kern_itm, 9, sizeof(int), &climate));
    CL_CHECK(clSetKernelArg(kern_itm, 10, sizeof(int), &polarization));
    SET_REAL(kern_itm, 11, conf, use_double);
    SET_REAL(kern_itm, 12, rel, use_double);
    /* d_loss and d_err set below */

    /* ---- output buffers (filled in the batch loop) ---- */
    std::vector<int16_t> out_signal_buf(total_cells);
    std::vector<uint8_t> out_mask_buf(total_cells);

    /* ---- batch loop ---- */
    std::vector<double> cpu_prof_buf;
    if (cpu_profiles) cpu_prof_buf.resize(alloc_batch * stride_prof);
    std::vector<double> loss_buf(alloc_batch);
    std::vector<int> err_buf(alloc_batch);

    /* Precompute constant dBm offset: dbm = const_dbm - loss */
    const double erp_w = tx_power_w * pow(10.0, tx_gain / 10.0);
    const double const_dbm = 10.0 * log10(erp_w * 1000.0) + 2.14 + rx_gain;

    for (int batch_start = 0; batch_start < total_cells; ) {
        int batch_count = std::min(batch_size, total_cells - batch_start);

        if (cpu_profiles) {
            for (int g = 0; g < batch_count; g++) {
                int ci = batch_start + g;
                int lc = ci % tile_w, lr = ci / tile_w;
                int col = rel_tile_x0 + lc, row = rel_tile_y0 + lr;
                int dx = col - rel_tx_col, dy = row - rel_tx_row;
                double dc = sqrt((double)dx*dx + (double)dy*dy);
                double dist = std::max(resolution, dc * resolution);
                int segs = std::max(1, (int)ceil(dc));
                double *dst = cpu_prof_buf.data() + (size_t)g * stride_prof;
                dst[0] = (double)segs;
                dst[1] = dist / segs;
                for (int i = 0; i <= segs; i++) {
                    double t = (double)i / (double)segs;
                    int c = clamp_i((int)llround((double)rel_tx_col + (double)dx * t), 0, crop_w - 1);
                    int r = clamp_i((int)llround((double)rel_tx_row + (double)dy * t), 0, crop_h - 1);
                    dst[i + 2] = (double)surface[(size_t)r * crop_w + c];
                }
                double pad = dst[segs + 2];
                for (int i = segs + 3; i < stride_prof; i++) dst[i] = pad;
            }
            size_t bytes = (size_t)batch_count * stride_prof * elem_size;
            if (use_double) {
                CL_CHECK(clEnqueueWriteBuffer(q, d_prof, CL_FALSE, 0, bytes,
                             cpu_prof_buf.data(), 0, NULL, NULL));
            } else {
                std::vector<float> tmp((size_t)batch_count * stride_prof);
                for (int g = 0; g < batch_count; g++)
                    for (int i = 0; i < stride_prof; i++)
                        tmp[g * stride_prof + i] = (float)cpu_prof_buf[g * stride_prof + i];
                CL_CHECK(clEnqueueWriteBuffer(q, d_prof, CL_TRUE, 0, bytes,
                             tmp.data(), 0, NULL, NULL));
            }
        } else {
            CL_CHECK(clSetKernelArg(kern_build, 10, sizeof(int), &batch_start));
            CL_CHECK(clSetKernelArg(kern_build, 11, sizeof(int), &batch_count));
            CL_CHECK(clSetKernelArg(kern_build, 12, sizeof(int), &total_cells));
            size_t global0 = (size_t)batch_count;
            global0 = ((global0 + local - 1) / local) * local;
            cl_int kerr = clEnqueueNDRangeKernel(q, kern_build, 1, NULL, &global0, &local, 0, NULL, NULL);
            if ((kerr == CL_MEM_OBJECT_ALLOCATION_FAILURE || kerr == CL_OUT_OF_RESOURCES) && batch_size > (int)local) {
                fprintf(stderr, "warning: VRAM pressure on %s (build kernel), halving batch %d -> %d\n",
                        pick.name, batch_size, batch_size / 2);
                batch_size = (int)(((size_t)(batch_size / 2) / local) * local);
                if (batch_size < (int)local) batch_size = (int)local;
                alloc_batch = ((size_t)batch_size + local - 1) / local * local;
                recreate_queue();
                alloc_batch_buffers(alloc_batch);
                loss_buf.resize(alloc_batch);
                err_buf.resize(alloc_batch);
                continue;
            }
            CL_CHECK(kerr);
        }

        /* Run itm_p2p_kernel — global size = batch_count (kernel uses get_global_id) */
        CL_CHECK(clSetKernelArg(kern_itm, 15, sizeof(int), &batch_count));
        size_t global = (size_t)batch_count;
        global = ((global + local - 1) / local) * local;
        cl_int kerr_itm = clEnqueueNDRangeKernel(q, kern_itm, 1, NULL, &global, &local, 0, NULL, NULL);
        if ((kerr_itm == CL_MEM_OBJECT_ALLOCATION_FAILURE || kerr_itm == CL_OUT_OF_RESOURCES) && batch_size > (int)local) {
            fprintf(stderr, "warning: VRAM pressure on %s (itm kernel), halving batch %d -> %d\n",
                    pick.name, batch_size, batch_size / 2);
            batch_size = (int)(((size_t)(batch_size / 2) / local) * local);
            if (batch_size < (int)local) batch_size = (int)local;
            alloc_batch = ((size_t)batch_size + local - 1) / local * local;
            recreate_queue();
            alloc_batch_buffers(alloc_batch);
            loss_buf.resize(alloc_batch);
            err_buf.resize(alloc_batch);
            continue;
        }
        CL_CHECK(kerr_itm);

        /* Read back dbloss (blocking read on in-order queue automatically awaits kern_build + kern_itm) */
        cl_int rerr = CL_SUCCESS;
        if (use_double) {
            rerr = clEnqueueReadBuffer(q, d_loss, CL_TRUE, 0,
                         (size_t)batch_count * sizeof(double), loss_buf.data(), 0, NULL, NULL);
        } else {
            std::vector<float> fl(batch_count);
            rerr = clEnqueueReadBuffer(q, d_loss, CL_TRUE, 0,
                         (size_t)batch_count * sizeof(float), fl.data(), 0, NULL, NULL);
            if (rerr == CL_SUCCESS) {
                for (int i = 0; i < batch_count; i++) loss_buf[i] = (double)fl[i];
            }
        }
        if ((rerr == CL_MEM_OBJECT_ALLOCATION_FAILURE || rerr == CL_OUT_OF_RESOURCES) && batch_size > (int)local) {
            fprintf(stderr, "warning: VRAM pressure on %s (readback), halving batch %d -> %d\n",
                    pick.name, batch_size, batch_size / 2);
            batch_size = (int)(((size_t)(batch_size / 2) / local) * local);
            if (batch_size < (int)local) batch_size = (int)local;
            alloc_batch = ((size_t)batch_size + local - 1) / local * local;
            recreate_queue();
            alloc_batch_buffers(alloc_batch);
            loss_buf.resize(alloc_batch);
            err_buf.resize(alloc_batch);
            continue;
        }
        CL_CHECK(rerr);

        for (int g = 0; g < batch_count; g++) {
            double dbm = const_dbm - loss_buf[g];
            int sig_i = (int)llround(dbm * 10.0);
            sig_i = std::max(-32768, std::min(32767, sig_i));
            out_signal_buf[batch_start + g] = (int16_t)sig_i;
            out_mask_buf[batch_start + g] = (uint8_t)(dbm >= rx_sensitivity ? 1 : 0);
        }

        fprintf(stderr, "\rbatch %d/%d (%d cells)",
                batch_start / batch_size + 1,
                (total_cells + batch_size - 1) / batch_size,
                batch_count);
        batch_start += batch_count;
    }
    fprintf(stderr, "\n");

    /* ---- compute stats ---- */
    int covered = 0;
    for (int i = 0; i < total_cells; i++)
        if (out_mask_buf[i]) covered++;

    /* ---- write output files (same format as ultra_main) ---- */
    std::string base(out_prefix);
    char sig_suffix[64], mask_suffix[64], meta_suffix[64];
    snprintf(sig_suffix, sizeof(sig_suffix), "_x%d_y%d.signal_i16le.bin", tile_x0, tile_y0);
    snprintf(mask_suffix, sizeof(mask_suffix), "_x%d_y%d.mask_u8.bin", tile_x0, tile_y0);
    snprintf(meta_suffix, sizeof(meta_suffix), "_x%d_y%d.meta.json", tile_x0, tile_y0);

    FILE *fp = fopen((base + sig_suffix).c_str(), "wb");
    if (!fp) { fprintf(stderr, "cannot write signal\n"); return 1; }
    fwrite(out_signal_buf.data(), sizeof(int16_t), total_cells, fp);
    fclose(fp);

    fp = fopen((base + mask_suffix).c_str(), "wb");
    if (!fp) { fprintf(stderr, "cannot write mask\n"); return 1; }
    fwrite(out_mask_buf.data(), sizeof(uint8_t), total_cells, fp);
    fclose(fp);

    fp = fopen((base + meta_suffix).c_str(), "wb");
    if (!fp) { fprintf(stderr, "cannot write meta\n"); return 1; }
    fprintf(fp,
            "{\n"
            "  \"model\": \"itm_projected_grid\",\n"
            "  \"width\": %d,\n"
            "  \"height\": %d,\n"
            "  \"tile_x0\": %d,\n"
            "  \"tile_y0\": %d,\n"
            "  \"tile_w\": %d,\n"
            "  \"tile_h\": %d,\n"
            "  \"threads\": 1,\n"
            "  \"resolution_m\": %.6f,\n"
            "  \"min_x\": %.6f,\n"
            "  \"max_y\": %.6f,\n"
            "  \"signal_scale\": \"dbm_x10_i16\",\n"
            "  \"mask_value\": \"1 means dbm >= rx_sensitivity_dbm\",\n"
            "  \"rx_sensitivity_dbm\": %.3f,\n"
            "  \"covered_cells\": %d,\n"
            "  \"total_cells\": %d,\n"
            "  \"itm_errnums\": [0, 0, 0, 0, 0, 0],\n"
            "  \"gpu_device\": \"%s\",\n"
            "  \"gpu_precision\": \"%s\"\n"
            "}\n",
            width, height, tile_x0, tile_y0, tile_w, tile_h,
            resolution, min_x, max_y,
            rx_sensitivity, covered, total_cells,
            pick.name, use_double ? "FP64" : "FP32");
    fclose(fp);

    fprintf(stderr, "wrote tile x0=%d y0=%d w=%d h=%d into %s.*\n",
            tile_x0, tile_y0, tile_w, tile_h, out_prefix);

    /* ---- cleanup ---- */
    clReleaseMemObject(d_surf);
    clReleaseMemObject(d_prof);
    clReleaseMemObject(d_scratch);
    clReleaseMemObject(d_loss);
    clReleaseMemObject(d_err);
    clReleaseKernel(kern_build);
    clReleaseKernel(kern_itm);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    return 0;
}
