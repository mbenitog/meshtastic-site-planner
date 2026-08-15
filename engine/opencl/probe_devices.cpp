/* OpenCL device probe.
 *
 * Standalone diagnostic utility: enumerates every platform and device
 * reachable through the OpenCL ICD loader and dumps the capability table
 * that the ultra GPU backend needs to choose its code path at runtime
 * (FP64 support, local memory size, image support, work-group limits, ...).
 *
 * Build:
 *   macOS:  clang++ -O2 -std=gnu++11 -framework OpenCL \
 *             -o engine/build/opencl_probe engine/opencl/probe_devices.cpp
 *   Linux:  g++ -O2 -std=gnu++11 -lOpenCL \
 *             -o engine/build/opencl_probe engine/opencl/probe_devices.cpp
 *
 * This is deliberately defensive: every clGetDeviceInfo call is checked
 * and a failure for one field is reported but does not abort the scan.
 * The whole point of the probe is to run on weird/old drivers.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

static const char *err_str(cl_int e) {
    switch (e) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_PROPERTY: return "CL_INVALID_PROPERTY";
    case CL_INVALID_IMAGE_DESCRIPTOR: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case CL_INVALID_COMPILER_OPTIONS: return "CL_INVALID_COMPILER_OPTIONS";
    case CL_INVALID_LINKER_OPTIONS: return "CL_INVALID_LINKER_OPTIONS";
    case CL_INVALID_DEVICE_PARTITION_COUNT: return "CL_INVALID_DEVICE_PARTITION_COUNT";
    default: return "<unknown>";
    }
}

/* ---- small typed query helpers ------------------------------------------- */

static void q_str(cl_device_id d, cl_device_info what, const char *label) {
    char buf[4096];
    size_t n = 0;
    cl_int e = clGetDeviceInfo(d, what, sizeof(buf), buf, &n);
    if (e == CL_SUCCESS) {
        buf[n < sizeof(buf) ? n : sizeof(buf) - 1] = '\0';
        /* strip trailing newline */
        while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = '\0';
        printf("  %-34s %s\n", label, buf);
    } else {
        printf("  %-34s <error: %s>\n", label, err_str(e));
    }
}

static void q_uint(cl_device_id d, cl_device_info what, const char *label) {
    cl_uint v = 0;
    cl_int e = clGetDeviceInfo(d, what, sizeof(v), &v, NULL);
    if (e == CL_SUCCESS)
        printf("  %-34s %u\n", label, v);
    else
        printf("  %-34s <error: %s>\n", label, err_str(e));
}

static void q_ulong(cl_device_id d, cl_device_info what, const char *label) {
    cl_ulong v = 0;
    cl_int e = clGetDeviceInfo(d, what, sizeof(v), &v, NULL);
    if (e == CL_SUCCESS)
        printf("  %-34s %llu\n", label, (unsigned long long)v);
    else
        printf("  %-34s <error: %s>\n", label, err_str(e));
}

static void q_size(cl_device_id d, cl_device_info what, const char *label) {
    size_t v = 0;
    cl_int e = clGetDeviceInfo(d, what, sizeof(v), &v, NULL);
    if (e == CL_SUCCESS)
        printf("  %-34s %zu\n", label, v);
    else
        printf("  %-34s <error: %s>\n", label, err_str(e));
}

static void q_bool(cl_device_id d, cl_device_info what, const char *label) {
    cl_bool v = CL_FALSE;
    cl_int e = clGetDeviceInfo(d, what, sizeof(v), &v, NULL);
    if (e == CL_SUCCESS)
        printf("  %-34s %s\n", label, v ? "yes" : "no");
    else
        printf("  %-34s <error: %s>\n", label, err_str(e));
}

static const char *dev_type_str(cl_device_type t) {
    switch (t) {
    case CL_DEVICE_TYPE_CPU: return "CPU";
    case CL_DEVICE_TYPE_GPU: return "GPU";
    case CL_DEVICE_TYPE_ACCELERATOR: return "ACCELERATOR";
    case CL_DEVICE_TYPE_DEFAULT: return "DEFAULT";
    case CL_DEVICE_TYPE_CUSTOM: return "CUSTOM";
    default: return "<mixed/unknown>";
    }
}

static void q_devtype(cl_device_id d, cl_device_info what, const char *label) {
    cl_device_type v = 0;
    cl_int e = clGetDeviceInfo(d, what, sizeof(v), &v, NULL);
    if (e == CL_SUCCESS)
        printf("  %-34s %s (0x%lx)\n", label, dev_type_str(v), (unsigned long)v);
    else
        printf("  %-34s <error: %s>\n", label, err_str(e));
}

static void q_local_mem_type(cl_device_id d, cl_device_info what, const char *label) {
    cl_device_local_mem_type v = 0;
    cl_int e = clGetDeviceInfo(d, what, sizeof(v), &v, NULL);
    if (e == CL_SUCCESS)
        printf("  %-34s %s\n", label,
               v == CL_LOCAL ? "LOCAL (on-chip)" :
               v == CL_GLOBAL ? "GLOBAL (emulated)" : "NONE");
    else
        printf("  %-34s <error: %s>\n", label, err_str(e));
}

static void q_work_item_sizes(cl_device_id d, cl_device_info what, const char *label) {
    cl_uint dims = 0;
    clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(dims), &dims, NULL);
    if (dims == 0) {
        printf("  %-34s <dims=0>\n", label);
        return;
    }
    size_t *sz = (size_t *)malloc(dims * sizeof(size_t));
    cl_int e = clGetDeviceInfo(d, what, dims * sizeof(size_t), sz, NULL);
    if (e == CL_SUCCESS) {
        printf("  %-34s [", label);
        for (cl_uint i = 0; i < dims; i++)
            printf("%s%zu", i ? ", " : "", sz[i]);
        printf("]\n");
    } else {
        printf("  %-34s <error: %s>\n", label, err_str(e));
    }
    free(sz);
}

static int has_extension(cl_device_id d, const char *name) {
    char buf[16384];
    size_t n = 0;
    if (clGetDeviceInfo(d, CL_DEVICE_EXTENSIONS, sizeof(buf), buf, &n) != CL_SUCCESS)
        return 0;
    buf[n < sizeof(buf) ? n : sizeof(buf) - 1] = '\0';
    /* tokenize on spaces */
    for (char *p = buf; *p; ) {
        char *e = p;
        while (*e && *e != ' ') e++;
        char saved = *e;
        *e = '\0';
        if (strcmp(p, name) == 0) { *e = saved; return 1; }
        *e = saved;
        p = (*e == '\0') ? e : e + 1;
    }
    return 0;
}

static void q_extensions(cl_device_id d, const char *label) {
    char buf[16384];
    size_t n = 0;
    cl_int e = clGetDeviceInfo(d, CL_DEVICE_EXTENSIONS, sizeof(buf), buf, &n);
    if (e != CL_SUCCESS) {
        printf("  %-34s <error: %s>\n", label, err_str(e));
        return;
    }
    buf[n < sizeof(buf) ? n : sizeof(buf) - 1] = '\0';
    printf("  %-34s %s\n", label, buf);
    /* Highlight the ones the GPU backend actually keys on. */
    printf("  %-34s fp64=%s  fp16=%s  subgroups=%s  int64_atomics=%s  images=%s\n",
           "  -> capability flags",
           has_extension(d, "cl_khr_fp64") ? "YES" : "no",
           has_extension(d, "cl_khr_fp16") ? "YES" : "no",
           has_extension(d, "cl_khr_subgroups") ? "YES" : "no",
           has_extension(d, "cl_khr_int64_base_atomics") ? "YES" : "no",
           has_extension(d, "cl_khr_image2d_from_buffer") ? "YES(khr)" : "core");
}

/* ---- image format probing ------------------------------------------------ */

static const char *channel_order_str(cl_channel_order o) {
    switch (o) {
    case CL_R: return "R";
    case CL_A: return "A";
    case CL_RG: return "RG";
    case CL_RA: return "RA";
    case CL_RGB: return "RGB";
    case CL_RGBA: return "RGBA";
    case CL_BGRA: return "BGRA";
    case CL_INTENSITY: return "INTENSITY";
    case CL_LUMINANCE: return "LUMINANCE";
    case CL_Rx: return "Rx";
    case CL_RGx: return "RGx";
    case CL_RGBx: return "RGBx";
#ifdef CL_sRGB
    case CL_sRGB: return "sRGB";
    case CL_sRGBx: return "sRGBx";
    case CL_sRGBA: return "sRGBA";
    case CL_sBGRA: return "sBGRA";
    case CL_ABGR: return "ABGR";
#endif
    default: return "?";
    }
}

static const char *channel_type_str(cl_channel_type t) {
    switch (t) {
    case CL_SNORM_INT8: return "SNORM_INT8";
    case CL_SNORM_INT16: return "SNORM_INT16";
    case CL_UNORM_INT8: return "UNORM_INT8";
    case CL_UNORM_INT16: return "UNORM_INT16";
    case CL_UNORM_SHORT_565: return "UNORM_SHORT_565";
    case CL_UNORM_SHORT_555: return "UNORM_SHORT_555";
    case CL_UNORM_INT_101010: return "UNORM_INT_101010";
    case CL_SIGNED_INT8: return "SIGNED_INT8";
    case CL_SIGNED_INT16: return "SIGNED_INT16";
    case CL_SIGNED_INT32: return "SIGNED_INT32";
    case CL_UNSIGNED_INT8: return "UNSIGNED_INT8";
    case CL_UNSIGNED_INT16: return "UNSIGNED_INT16";
    case CL_UNSIGNED_INT32: return "UNSIGNED_INT32";
    case CL_HALF_FLOAT: return "HALF_FLOAT";
    case CL_FLOAT: return "FLOAT";
#ifdef CL_UNORM_INT_101010_2
    case CL_UNORM_INT_101010_2: return "UNORM_INT_101010_2";
#endif
    default: return "?";
    }
}

static void probe_image_formats(cl_device_id d, cl_context ctx) {
    if (!ctx) {
        printf("  image formats (READ_ONLY 2D):      <no context>\n");
        return;
    }
    cl_uint num = 0;
    cl_image_format formats[256];
    cl_int e = clGetSupportedImageFormats(ctx, CL_MEM_READ_ONLY,
                                         CL_MEM_OBJECT_IMAGE2D,
                                         256, formats, &num);
    if (e != CL_SUCCESS) {
        printf("  image formats (READ_ONLY 2D):      <error: %s>\n", err_str(e));
        return;
    }
    printf("  image formats (READ_ONLY 2D):      %u supported\n", num);
    int saw_signed16_r = 0, saw_signed16_rgba = 0, saw_float_r = 0, saw_float_rgba = 0;
    for (cl_uint i = 0; i < num; i++) {
        const char *o = channel_order_str(formats[i].image_channel_order);
        const char *t = channel_type_str(formats[i].image_channel_data_type);
        printf("    [%2u] %-10s / %s\n", i, o, t);
        if (formats[i].image_channel_data_type == CL_SIGNED_INT16) {
            if (formats[i].image_channel_order == CL_R) saw_signed16_r = 1;
            if (formats[i].image_channel_order == CL_RGBA) saw_signed16_rgba = 1;
        }
        if (formats[i].image_channel_data_type == CL_FLOAT) {
            if (formats[i].image_channel_order == CL_R) saw_float_r = 1;
            if (formats[i].image_channel_order == CL_RGBA) saw_float_rgba = 1;
        }
    }
    printf("  -> surface-relevant:  SIGNED_INT16 R=%s RGBA=%s   FLOAT R=%s RGBA=%s\n",
           saw_signed16_r ? "YES" : "no",
           saw_signed16_rgba ? "YES" : "no",
           saw_float_r ? "YES" : "no",
           saw_float_rgba ? "YES" : "no");
}

/* ---- per-device report --------------------------------------------------- */

static void report_device(cl_device_id d, int idx, cl_platform_id plat) {
    printf("=== Device %d ===\n", idx);
    q_str(d, CL_DEVICE_NAME, "name");
    q_str(d, CL_DEVICE_VENDOR, "vendor");
    q_devtype(d, CL_DEVICE_TYPE, "type");
    q_str(d, CL_DEVICE_VERSION, "device version (OpenCL)");
    q_str(d, CL_DEVICE_OPENCL_C_VERSION, "OpenCL C version");
    q_str(d, CL_DRIVER_VERSION, "driver version");
    q_str(d, CL_DEVICE_PROFILE, "profile");

    q_bool(d, CL_DEVICE_AVAILABLE, "available");
    q_bool(d, CL_DEVICE_COMPILER_AVAILABLE, "compiler available");
    q_bool(d, CL_DEVICE_ENDIAN_LITTLE, "little endian");
    q_bool(d, CL_DEVICE_ERROR_CORRECTION_SUPPORT, "ECC");

    q_uint(d, CL_DEVICE_MAX_COMPUTE_UNITS, "max compute units");
    q_uint(d, CL_DEVICE_MAX_CLOCK_FREQUENCY, "max clock (MHz)");
    q_uint(d, CL_DEVICE_MAX_PARAMETER_SIZE, "max parameter size (B)");
    q_uint(d, CL_DEVICE_MAX_CONSTANT_ARGS, "max constant args");
    q_ulong(d, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "max constant buffer (B)");

    q_size(d, CL_DEVICE_MAX_WORK_GROUP_SIZE, "max work-group size");
    q_uint(d, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "max work-item dims");
    q_work_item_sizes(d, CL_DEVICE_MAX_WORK_ITEM_SIZES, "max work-item sizes");
    q_uint(d, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "vec width char");
    q_uint(d, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "vec width short");
    q_uint(d, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "vec width int");
    q_uint(d, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "vec width long");
    q_uint(d, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "vec width float");
    q_uint(d, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "vec width double");

    q_ulong(d, CL_DEVICE_GLOBAL_MEM_SIZE, "global mem (B)");
    q_ulong(d, CL_DEVICE_MAX_MEM_ALLOC_SIZE, "max alloc (B)");
    q_ulong(d, CL_DEVICE_LOCAL_MEM_SIZE, "local mem (B)");
    q_local_mem_type(d, CL_DEVICE_LOCAL_MEM_TYPE, "local mem type");
#ifdef CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE
    q_ulong(d, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE, "max global var (B)");
#endif

    q_bool(d, CL_DEVICE_IMAGE_SUPPORT, "image support");
    q_size(d, CL_DEVICE_IMAGE2D_MAX_WIDTH, "image2d max width");
    q_size(d, CL_DEVICE_IMAGE2D_MAX_HEIGHT, "image2d max height");
    q_size(d, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, "image max buffer size");
    q_uint(d, CL_DEVICE_MAX_READ_IMAGE_ARGS, "max read image args");
    q_uint(d, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, "max write image args");

    q_extensions(d, "extensions");

    /* Try to build a context to probe image formats (and as a smoke test
     * that the device is actually usable). */
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)plat, 0
    };
    cl_int e;
    cl_context ctx = clCreateContext(props, 1, &d, NULL, NULL, &e);
    if (e != CL_SUCCESS) {
        printf("  context create:                   <error: %s>\n", err_str(e));
    } else {
        probe_image_formats(d, ctx);
        /* preferred work-group multiple needs a compiled kernel; skip here,
         * the runtime will query it via clGetKernelWorkGroupInfo. */
        clReleaseContext(ctx);
    }
    printf("\n");
}

/* ---- platform report ----------------------------------------------------- */

static void report_platform(cl_platform_id plat, int idx) {
    char buf[4096];
    size_t n = 0;
    printf("########## Platform %d ##########\n", idx);
    static const struct { cl_platform_info what; const char *label; } fields[] = {
        { CL_PLATFORM_NAME, "name" },
        { CL_PLATFORM_VENDOR, "vendor" },
        { CL_PLATFORM_VERSION, "version" },
        { CL_PLATFORM_PROFILE, "profile" },
        { CL_PLATFORM_EXTENSIONS, "extensions" },
    };
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); i++) {
        n = 0;
        cl_int e = clGetPlatformInfo(plat, fields[i].what,
                                     sizeof(buf), buf, &n);
        if (e == CL_SUCCESS) {
            buf[n < sizeof(buf) ? n : sizeof(buf) - 1] = '\0';
            while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = '\0';
            printf("  %-12s %s\n", fields[i].label, buf);
        } else {
            printf("  %-12s <error: %s>\n", fields[i].label, err_str(e));
        }
    }
    printf("\n");

    /* Enumerate all device types we care about. We query ALL_DEVICES so we
     * also see CPU-via-OpenCL (useful as a fallback target). */
    cl_uint ndev = 0;
    cl_int e = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, NULL, &ndev);
    if (e == CL_DEVICE_NOT_FOUND) {
        printf("  (no devices on this platform)\n\n");
        return;
    }
    if (e != CL_SUCCESS) {
        printf("  clGetDeviceIDs: <error: %s>\n\n", err_str(e));
        return;
    }
    if (ndev == 0) {
        printf("  (no devices on this platform)\n\n");
        return;
    }
    cl_device_id *devs = (cl_device_id *)malloc(ndev * sizeof(cl_device_id));
    e = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, ndev, devs, NULL);
    if (e != CL_SUCCESS) {
        printf("  clGetDeviceIDs(alloc): <error: %s>\n\n", err_str(e));
        free(devs);
        return;
    }
    for (cl_uint i = 0; i < ndev; i++)
        report_device(devs[i], (int)i, plat);
    free(devs);
}

int main(int argc, char **argv) {
    int json_mode = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) json_mode = 1;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("usage: opencl_probe [--json]\n");
            printf("  Dumps every OpenCL platform/device and the capability\n");
            printf("  fields the ultra GPU backend keys on at runtime.\n");
            return 0;
        }
    }
    if (json_mode) {
        fprintf(stderr, "note: --json not yet implemented; emitting text.\n");
    }

    cl_uint nplat = 0;
    cl_int e = clGetPlatformIDs(0, NULL, &nplat);
    if (e != CL_SUCCESS) {
        fprintf(stderr, "clGetPlatformIDs failed: %s\n", err_str(e));
        fprintf(stderr, "No OpenCL ICD loader / platform found.\n");
        fprintf(stderr, "On Linux install 'ocl-icd-opencl-dev' (dev) and an\n");
        fprintf(stderr, "ICD from the GPU vendor (runtime).\n");
        return 1;
    }
    if (nplat == 0) {
        fprintf(stderr, "No OpenCL platforms found.\n");
        return 1;
    }
    cl_platform_id *plats = (cl_platform_id *)malloc(nplat * sizeof(cl_platform_id));
    e = clGetPlatformIDs(nplat, plats, NULL);
    if (e != CL_SUCCESS) {
        fprintf(stderr, "clGetPlatformIDs(alloc) failed: %s\n", err_str(e));
        free(plats);
        return 1;
    }
    printf("OpenCL probe: %u platform(s)\n\n", nplat);
    for (cl_uint i = 0; i < nplat; i++)
        report_platform(plats[i], (int)i);
    free(plats);
    return 0;
}
