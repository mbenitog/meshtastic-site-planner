/* Phase 2 validation harness: CPU-vs-GPU parity for point_to_point_ITM.
 *
 * Loads the projected 2.5 m surface (same int16 raw format as ultra_cli),
 * builds terrain profiles for a sample of cells exactly the way
 * engine/native/ultra_main.cpp does (run_itm_path), then:
 *
 *   1. Runs the CPU reference (splat/itwom3.0.cpp :: point_to_point_ITM,
 *      FP64) for every sampled profile.
 *   2. Uploads the profiles to the OpenCL device and runs itm_p2p_kernel
 *      in two variants when available: FP64 (cl_khr_fp64) and FP32.
 *   3. Compares each GPU variant against the CPU reference cell-by-cell
 *      and reports max/mean abs error and % within {0.1, 0.5, 1.0} dB.
 *
 * Build (see engine/build_opencl.sh):
 *   macOS: clang++ -O2 -std=gnu++11 -framework OpenCL \
 *            engine/opencl/itm_validate.cpp splat/itwom3.0.cpp \
 *            -o engine/build/itm_validate
 *   Linux: g++ -O2 -std=gnu++11 -lOpenCL \
 *            engine/opencl/itm_validate.cpp splat/itwom3.0.cpp \
 *            -o engine/build/itm_validate
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

void point_to_point_ITM(double elev[], double tht_m, double rht_m,
                        double eps_dielect, double sgm_conductivity,
                        double eno_ns_surfref, double frq_mhz,
                        int radio_climate, int pol, double conf, double rel,
                        double &dbloss, char *strmode, int &errnum);

/* ---- CLI helpers (mirrors ultra_main.cpp) ---- */

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

/* ---- OpenCL error helper ---- */

static const char *cl_err(cl_int e) {
    switch (e) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
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

/* ---- device selection ---- */

struct DevicePick {
    cl_platform_id platform;
    cl_device_id device;
    cl_bool has_fp64;
    cl_device_type type;
    char name[256];
};

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

static int pick_device(int forced_plat, int forced_dev, DevicePick *out) {
    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, NULL, &nplat) != CL_SUCCESS || nplat == 0) {
        fprintf(stderr, "no OpenCL platforms\n"); return 1;
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
            size_t nl = 0;
            clGetDeviceInfo(devs[di], CL_DEVICE_NAME, sizeof(c.name), c.name, &nl);
            c.name[nl < sizeof(c.name) ? nl : sizeof(c.name)-1] = '\0';
            /* prefer GPU, then fp64, then first seen */
            int score = 0;
            if (t == CL_DEVICE_TYPE_GPU) score += 100;
            if (c.has_fp64) score += 10;
            if (!have_best || score > best_score) { best = c; best_score = score; have_best = true; }
        }
    }
    if (!have_best) { fprintf(stderr, "no matching OpenCL device\n"); return 1; }
    *out = best;
    return 0;
}

/* ---- build a single profile (mirror ultra_main run_itm_path) ---- */

struct ProfileBuild {
    std::vector<double> elev;   // length = segments + 16
    int segments;
};

static ProfileBuild build_profile(const std::vector<int16_t> &surface, int width,
                                  int height, double resolution,
                                  int tx_col, int tx_row, int rx_col, int rx_row) {
    int dx = rx_col - tx_col, dy = rx_row - tx_row;
    double dist_cells = sqrt((double)dx*dx + (double)dy*dy);
    double dist = std::max(resolution, dist_cells * resolution);
    int segments = std::max(1, (int)ceil(dist_cells));
    ProfileBuild pb;
    pb.segments = segments;
    pb.elev.resize((size_t)segments + 16);
    pb.elev[0] = (double)segments;
    pb.elev[1] = dist / (double)segments;
    for (int i = 0; i <= segments; i++) {
        double t = (double)i / (double)segments;
        int col = clamp_i((int)llround((double)tx_col + (double)dx * t), 0, width - 1);
        int row = clamp_i((int)llround((double)tx_row + (double)dy * t), 0, height - 1);
        pb.elev[(size_t)i + 2] = (double)surface[(size_t)row * (size_t)width + (size_t)col];
    }
    for (size_t i = (size_t)segments + 3; i < pb.elev.size(); i++)
        pb.elev[i] = pb.elev[(size_t)segments + 2];
    return pb;
}

/* ---- run one precision variant on the device ---- */

struct VariantResult {
    std::vector<double> dbloss;
    std::vector<int> errnum;
    bool ran;
};

static VariantResult run_variant(const DevicePick &pick, const char *kernel_src,
                                 size_t src_len, int use_double,
                                 const std::vector<double> &profiles_f64,
                                 int num_cells, int stride,
                                 double tht_m, double rht_m,
                                 double eps, double sgm, double eno,
                                 double frq, int climate, int pol,
                                 double conf, double rel,
                                 int wg_multiple, size_t max_wg) {
    VariantResult R;
    R.ran = false;
    R.dbloss.resize(num_cells, 0.0);
    R.errnum.resize(num_cells, 0);

    if (use_double && !pick.has_fp64) return R;

    cl_int e;
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)pick.platform, 0
    };
    cl_context ctx = clCreateContext(props, 1, &pick.device, NULL, NULL, &e);
    CL_CHECK(e);
    cl_command_queue q = clCreateCommandQueue(ctx, pick.device, 0, &e);
    CL_CHECK(e);

    const char *srcs[1] = { kernel_src };
    size_t lens[1] = { src_len };
    cl_program prog = clCreateProgramWithSource(ctx, 1, srcs, lens, &e);
    CL_CHECK(e);
    const char *opts = use_double ? "-DUSE_DOUBLE" : "";
    e = clBuildProgram(prog, 1, &pick.device, opts, NULL, NULL);
    if (e != CL_SUCCESS) {
        fprintf(stderr, "build failed (%s variant): %s\n",
                use_double ? "FP64" : "FP32", cl_err(e));
        size_t loglen = 0;
        clGetProgramBuildInfo(prog, pick.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &loglen);
        std::vector<char> log(loglen + 1);
        clGetProgramBuildInfo(prog, pick.device, CL_PROGRAM_BUILD_LOG, loglen, log.data(), NULL);
        log[loglen] = '\0';
        fprintf(stderr, "---- build log ----\n%s\n------------------\n", log.data());
        clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx);
        return R;
    }
    cl_kernel kern = clCreateKernel(prog, "itm_p2p_kernel", &e);
    CL_CHECK(e);

    size_t wg;
    CL_CHECK(clGetKernelWorkGroupInfo(kern, pick.device, CL_KERNEL_WORK_GROUP_SIZE,
                                      sizeof(wg), &wg, NULL));
    if (wg > max_wg) wg = max_wg;
    size_t local = (wg_multiple > 0) ? (size_t)wg_multiple : 64;
    if (local > wg) local = wg;
    size_t global = ((num_cells + local - 1) / local) * local;

    /* upload profiles in the variant's precision */
    size_t elem_bytes = use_double ? sizeof(double) : sizeof(float);
    std::vector<unsigned char> prof_bytes((size_t)num_cells * stride * elem_bytes);
    for (int c = 0; c < num_cells; c++) {
        const double *src = profiles_f64.data() + (size_t)c * stride;
        if (use_double) {
            memcpy(prof_bytes.data() + (size_t)c * stride * elem_bytes, src, stride * elem_bytes);
        } else {
            float *dst = (float*)(prof_bytes.data() + (size_t)c * stride * elem_bytes);
            for (int i = 0; i < stride; i++) dst[i] = (float)src[i];
        }
    }
    cl_mem d_prof = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   prof_bytes.size(), prof_bytes.data(), &e);
    CL_CHECK(e);
    size_t scratch_elems = (size_t)num_cells * 256;   /* D1THX_SCRATCH_LEN */
    cl_mem d_scratch = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                      scratch_elems * elem_bytes, NULL, &e);
    CL_CHECK(e);
    cl_mem d_loss = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                   (size_t)num_cells * elem_bytes, NULL, &e);
    CL_CHECK(e);
    cl_mem d_err = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                  (size_t)num_cells * sizeof(int), NULL, &e);
    CL_CHECK(e);

    if (use_double) {
        double _tht=tht_m,_rht=rht_m,_eps=eps,_sgm=sgm,_eno=eno,_frq=frq,_conf=conf,_rel=rel;
        CL_CHECK(clSetKernelArg(kern, 0, sizeof(d_prof), &d_prof));
        CL_CHECK(clSetKernelArg(kern, 1, sizeof(int), &stride));
        CL_CHECK(clSetKernelArg(kern, 2, sizeof(d_scratch), &d_scratch));
        CL_CHECK(clSetKernelArg(kern, 3, sizeof(double), &_tht));
        CL_CHECK(clSetKernelArg(kern, 4, sizeof(double), &_rht));
        CL_CHECK(clSetKernelArg(kern, 5, sizeof(double), &_eps));
        CL_CHECK(clSetKernelArg(kern, 6, sizeof(double), &_sgm));
        CL_CHECK(clSetKernelArg(kern, 7, sizeof(double), &_eno));
        CL_CHECK(clSetKernelArg(kern, 8, sizeof(double), &_frq));
        CL_CHECK(clSetKernelArg(kern, 9, sizeof(int), &climate));
        CL_CHECK(clSetKernelArg(kern,10, sizeof(int), &pol));
        CL_CHECK(clSetKernelArg(kern,11, sizeof(double), &_conf));
        CL_CHECK(clSetKernelArg(kern,12, sizeof(double), &_rel));
        CL_CHECK(clSetKernelArg(kern,13, sizeof(d_loss), &d_loss));
        CL_CHECK(clSetKernelArg(kern,14, sizeof(d_err), &d_err));
    } else {
        float _tht=(float)tht_m,_rht=(float)rht_m,_eps=(float)eps,_sgm=(float)sgm,
              _eno=(float)eno,_frq=(float)frq,_conf=(float)conf,_rel=(float)rel;
        CL_CHECK(clSetKernelArg(kern, 0, sizeof(d_prof), &d_prof));
        CL_CHECK(clSetKernelArg(kern, 1, sizeof(int), &stride));
        CL_CHECK(clSetKernelArg(kern, 2, sizeof(d_scratch), &d_scratch));
        CL_CHECK(clSetKernelArg(kern, 3, sizeof(float), &_tht));
        CL_CHECK(clSetKernelArg(kern, 4, sizeof(float), &_rht));
        CL_CHECK(clSetKernelArg(kern, 5, sizeof(float), &_eps));
        CL_CHECK(clSetKernelArg(kern, 6, sizeof(float), &_sgm));
        CL_CHECK(clSetKernelArg(kern, 7, sizeof(float), &_eno));
        CL_CHECK(clSetKernelArg(kern, 8, sizeof(float), &_frq));
        CL_CHECK(clSetKernelArg(kern, 9, sizeof(int), &climate));
        CL_CHECK(clSetKernelArg(kern,10, sizeof(int), &pol));
        CL_CHECK(clSetKernelArg(kern,11, sizeof(float), &_conf));
        CL_CHECK(clSetKernelArg(kern,12, sizeof(float), &_rel));
        CL_CHECK(clSetKernelArg(kern,13, sizeof(d_loss), &d_loss));
        CL_CHECK(clSetKernelArg(kern,14, sizeof(d_err), &d_err));
    }

    CL_CHECK(clEnqueueNDRangeKernel(q, kern, 1, NULL, &global, &local, 0, NULL, NULL));
    CL_CHECK(clFinish(q));

    if (use_double) {
        CL_CHECK(clEnqueueReadBuffer(q, d_loss, CL_TRUE, 0,
                     (size_t)num_cells * sizeof(double), R.dbloss.data(), 0, NULL, NULL));
    } else {
        std::vector<float> fl(num_cells);
        CL_CHECK(clEnqueueReadBuffer(q, d_loss, CL_TRUE, 0,
                     (size_t)num_cells * sizeof(float), fl.data(), 0, NULL, NULL));
        for (int i = 0; i < num_cells; i++) R.dbloss[i] = (double)fl[i];
    }
    CL_CHECK(clEnqueueReadBuffer(q, d_err, CL_TRUE, 0,
                 (size_t)num_cells * sizeof(int), R.errnum.data(), 0, NULL, NULL));

    clReleaseMemObject(d_prof); clReleaseMemObject(d_scratch);
    clReleaseMemObject(d_loss); clReleaseMemObject(d_err);
    clReleaseKernel(kern); clReleaseProgram(prog);
    clReleaseCommandQueue(q); clReleaseContext(ctx);
    R.ran = true;
    return R;
}

/* ---- comparison report ---- */

static void report(const char *label, const std::vector<double> &cpu,
                   const std::vector<int> &cpu_err,
                   const VariantResult &vr) {
    int n = (int)cpu.size();
    double maxd = 0, sumd = 0;
    int within01 = 0, within05 = 0, within1 = 0, err_mismatch = 0;
    double worst_cpu = 0, worst_gpu = 0; int worst_i = -1;
    for (int i = 0; i < n; i++) {
        double d = fabs(cpu[i] - vr.dbloss[i]);
        if (d > maxd) { maxd = d; worst_cpu = cpu[i]; worst_gpu = vr.dbloss[i]; worst_i = i; }
        sumd += d;
        if (d <= 0.1) within01++;
        if (d <= 0.5) within05++;
        if (d <= 1.0) within1++;
        if (cpu_err[i] != vr.errnum[i]) err_mismatch++;
    }
    printf("  [%s] cells=%d  max|d|=%.6f dB  mean|d|=%.6f dB\n",
           label, n, maxd, sumd / n);
    printf("      within 0.1 dB: %.2f%%   within 0.5 dB: %.2f%%   within 1.0 dB: %.2f%%\n",
           100.0*within01/n, 100.0*within05/n, 100.0*within1/n);
    printf("      errnum mismatches: %d / %d\n", err_mismatch, n);
    if (worst_i >= 0)
        printf("      worst: cell #%d  cpu=%.6f  gpu=%.6f  (d=%.6f)\n",
               worst_i, worst_cpu, worst_gpu, maxd);
}

int main(int argc, char **argv) {
    bool ok = true, f;
    const char *surface_path = arg_s(argc, argv, "--surface");
    ok &= require(surface_path != nullptr, "--surface");
    int width = (int)arg_f(argc, argv, "--width", &f); ok &= require(f, "--width");
    int height = (int)arg_f(argc, argv, "--height", &f); ok &= require(f, "--height");
    double min_x = arg_f(argc, argv, "--min-x", &f); ok &= require(f, "--min-x");
    double max_y = arg_f(argc, argv, "--max-y", &f); ok &= require(f, "--max-y");
    double resolution = arg_f(argc, argv, "--resolution-m", &f); ok &= require(f, "--resolution-m");
    double tx_x = arg_f(argc, argv, "--tx-x", &f); ok &= require(f, "--tx-x");
    double tx_y = arg_f(argc, argv, "--tx-y", &f); ok &= require(f, "--tx-y");
    double tx_height = arg_f(argc, argv, "--tx-height-m", &f); ok &= require(f, "--tx-height-m");
    double rx_height = arg_f(argc, argv, "--rx-height-m", &f); ok &= require(f, "--rx-height-m");
    double freq = arg_f(argc, argv, "--freq-mhz", &f); ok &= require(f, "--freq-mhz");
    double eps = arg_f(argc, argv, "--dielect", &f); if (!f) eps = 15.0;
    double conductivity = arg_f(argc, argv, "--conductivity", &f); if (!f) conductivity = 0.005;
    double bend = arg_f(argc, argv, "--bend", &f); if (!f) bend = 301.0;
    int climate = (int)arg_f(argc, argv, "--climate", &f); if (!f) climate = 5;
    int polarization = (int)arg_f(argc, argv, "--pol", &f); if (!f) polarization = 1;
    double conf = arg_f(argc, argv, "--conf", &f); if (!f) conf = 0.95;
    double rel = arg_f(argc, argv, "--rel", &f); if (!f) rel = 0.95;
    int sample_step = arg_i(argc, argv, "--sample-step", &f); if (!f) sample_step = 16;
    int max_cells = arg_i(argc, argv, "--max-cells", &f); if (!f) max_cells = 2000;
    int forced_plat = arg_i(argc, argv, "--platform", &f); if (!f) forced_plat = -1;
    int forced_dev = arg_i(argc, argv, "--device", &f); if (!f) forced_dev = -1;
    const char *env_plat = getenv("ULTRA_OPENCL_PLATFORM");
    const char *env_dev = getenv("ULTRA_OPENCL_DEVICE");
    if (env_plat) forced_plat = atoi(env_plat);
    if (env_dev) forced_dev = atoi(env_dev);
    const char *kernel_path = arg_s(argc, argv, "--kernel");
    if (!kernel_path) kernel_path = "engine/opencl/itm_kernel.cl";
    bool no_fp64 = false;
    for (int i = 1; i < argc; i++) if (strcmp(argv[i], "--no-fp64") == 0) no_fp64 = true;
    if (!ok) return 2;

    /* load surface */
    std::vector<int16_t> surface((size_t)width * (size_t)height);
    FILE *fp = fopen(surface_path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", surface_path); return 1; }
    size_t nrd = fread(surface.data(), sizeof(int16_t), surface.size(), fp);
    fclose(fp);
    if (nrd != surface.size()) { fprintf(stderr, "short read\n"); return 1; }

    int tx_col = clamp_i((int)llround((tx_x - min_x) / resolution), 0, width - 1);
    int tx_row = clamp_i((int)llround((max_y - tx_y) / resolution), 0, height - 1);

    /* sample cells */
    std::vector<int> cell_col, cell_row;
    for (int row = 0; row < height && (int)cell_col.size() < max_cells; row += sample_step) {
        for (int col = 0; col < width && (int)cell_col.size() < max_cells; col += sample_step) {
            if (col == tx_col && row == tx_row) continue;   /* skip tx itself */
            cell_col.push_back(col);
            cell_row.push_back(row);
        }
    }
    int num_cells = (int)cell_col.size();
    fprintf(stderr, "sampled %d cells (step=%d)\n", num_cells, sample_step);

    /* build profiles */
    int max_segments = 0;
    std::vector<ProfileBuild> pbs(num_cells);
    for (int c = 0; c < num_cells; c++) {
        pbs[c] = build_profile(surface, width, height, resolution,
                               tx_col, tx_row, cell_col[c], cell_row[c]);
        max_segments = std::max(max_segments, pbs[c].segments);
    }
    int stride = max_segments + 16;   /* common stride */
    fprintf(stderr, "max segments=%d  profile stride=%d\n", max_segments, stride);

    std::vector<double> profiles_f64((size_t)num_cells * stride, 0.0);
    for (int c = 0; c < num_cells; c++) {
        double *dst = profiles_f64.data() + (size_t)c * stride;
        memcpy(dst, pbs[c].elev.data(), pbs[c].elev.size() * sizeof(double));
        /* pad the rest of the stride with the last height */
        double pad = pbs[c].elev.back();
        for (int i = (int)pbs[c].elev.size(); i < stride; i++) dst[i] = pad;
    }

    /* CPU reference (FP64) */
    std::vector<double> cpu_loss(num_cells, 0.0);
    std::vector<int> cpu_err(num_cells, 0);
    for (int c = 0; c < num_cells; c++) {
        char mode[100] = {0};
        int errnum = 0;
        double loss = 0.0;
        /* point_to_point_ITM uses its own internal static state which is
         * reset each call via mdp; pass a fresh profile copy. */
        point_to_point_ITM(pbs[c].elev.data(), tx_height, rx_height, eps,
                           conductivity, bend, freq, climate, polarization,
                           conf, rel, loss, mode, errnum);
        cpu_loss[c] = loss;
        cpu_err[c] = errnum;
    }
    fprintf(stderr, "CPU reference done (%d cells)\n", num_cells);

    /* load kernel source */
    FILE *kf = fopen(kernel_path, "rb");
    if (!kf) { fprintf(stderr, "cannot open kernel %s\n", kernel_path); return 1; }
    fseek(kf, 0, SEEK_END); long ks = ftell(kf); fseek(kf, 0, SEEK_SET);
    std::vector<char> ksrc(ks + 1);
    fread(ksrc.data(), 1, ks, kf); ksrc[ks] = '\0'; fclose(kf);

    /* pick device */
    DevicePick pick;
    if (pick_device(forced_plat, forced_dev, &pick) != 0) return 1;
    fprintf(stderr, "device: %s  fp64=%s  type=%s\n", pick.name,
            pick.has_fp64 ? "yes" : "no",
            pick.type == CL_DEVICE_TYPE_GPU ? "GPU" :
            pick.type == CL_DEVICE_TYPE_CPU ? "CPU" : "OTHER");

    size_t max_wg = 256;
    clGetDeviceInfo(pick.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL);
    int wg_mult = 0;
    /* we will refine wg_mult with kernel query inside run_variant */

    printf("=== ITM point_to_point GPU vs CPU parity ===\n");
    printf("cells=%d  stride=%d  device=%s\n", num_cells, stride, pick.name);

    /* FP64 variant */
    if (!no_fp64 && pick.has_fp64) {
        VariantResult vr = run_variant(pick, ksrc.data(), ks, 1,
                                       profiles_f64, num_cells, stride,
                                       tx_height, rx_height, eps, conductivity, bend,
                                       freq, climate, polarization, conf, rel,
                                       wg_mult, max_wg);
        if (vr.ran) report("FP64", cpu_loss, cpu_err, vr);
        else printf("  [FP64] did not run\n");
    } else {
        printf("  [FP64] skipped (no cl_khr_fp64 or --no-fp64)\n");
    }

    /* FP32 variant */
    {
        VariantResult vr = run_variant(pick, ksrc.data(), ks, 0,
                                       profiles_f64, num_cells, stride,
                                       tx_height, rx_height, eps, conductivity, bend,
                                       freq, climate, polarization, conf, rel,
                                       wg_mult, max_wg);
        if (vr.ran) report("FP32", cpu_loss, cpu_err, vr);
        else printf("  [FP32] did not run\n");
    }

    return 0;
}
