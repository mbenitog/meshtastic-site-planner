/* Prototype native runner for the backend ultra DSM path.
 *
 * This intentionally does not touch the SPLAT parity CLI. It consumes the
 * projected 2.5 m surface artifact produced by backend/ultra and writes a
 * first RF raster using free-space path loss plus a terrain-obstruction
 * penalty. The output contract lets the backend exercise native execution now;
 * the propagation core can be replaced with a full projected-grid ITM runner
 * without changing the artifact boundary.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

static double arg_f(int argc, char **argv, const char *name, bool *found) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], name) == 0) {
            if (found)
                *found = true;
            return atof(argv[i + 1]);
        }
    }
    if (found)
        *found = false;
    return 0.0;
}

static const char *arg_s(int argc, char **argv, const char *name) {
    for (int i = 1; i + 1 < argc; i++)
        if (strcmp(argv[i], name) == 0)
            return argv[i + 1];
    return nullptr;
}

static bool require(bool found, const char *name) {
    if (!found)
        fprintf(stderr, "missing required argument %s\n", name);
    return found;
}

static int16_t sample_nearest(const std::vector<int16_t> &surface, int width,
                              int height, double min_x, double max_y,
                              double resolution, double x, double y) {
    int col = (int)llround((x - min_x) / resolution);
    int row = (int)llround((max_y - y) / resolution);
    col = std::max(0, std::min(width - 1, col));
    row = std::max(0, std::min(height - 1, row));
    return surface[(size_t)row * (size_t)width + (size_t)col];
}

static bool obstructed(const std::vector<int16_t> &surface, int width,
                       int height, double min_x, double max_y,
                       double resolution, double tx_x, double tx_y,
                       double tx_z, double rx_x, double rx_y, double rx_z) {
    double dx = rx_x - tx_x;
    double dy = rx_y - tx_y;
    double dist = sqrt(dx * dx + dy * dy);
    int steps = std::max(1, (int)floor(dist / resolution));

    for (int i = 1; i < steps; i++) {
        double t = (double)i / (double)steps;
        double x = tx_x + dx * t;
        double y = tx_y + dy * t;
        double los_z = tx_z + (rx_z - tx_z) * t;
        if ((double)sample_nearest(surface, width, height, min_x, max_y,
                                   resolution, x, y) > los_z)
            return true;
    }
    return false;
}

int main(int argc, char **argv) {
    bool ok = true, f;
    const char *surface_path = arg_s(argc, argv, "--surface");
    ok &= require(surface_path != nullptr, "--surface");
    const char *out_prefix = arg_s(argc, argv, "--out");
    ok &= require(out_prefix != nullptr, "--out");
    int width = (int)arg_f(argc, argv, "--width", &f);
    ok &= require(f, "--width");
    int height = (int)arg_f(argc, argv, "--height", &f);
    ok &= require(f, "--height");
    double min_x = arg_f(argc, argv, "--min-x", &f);
    ok &= require(f, "--min-x");
    double max_y = arg_f(argc, argv, "--max-y", &f);
    ok &= require(f, "--max-y");
    double resolution = arg_f(argc, argv, "--resolution-m", &f);
    ok &= require(f, "--resolution-m");
    double tx_x = arg_f(argc, argv, "--tx-x", &f);
    ok &= require(f, "--tx-x");
    double tx_y = arg_f(argc, argv, "--tx-y", &f);
    ok &= require(f, "--tx-y");
    double tx_height = arg_f(argc, argv, "--tx-height-m", &f);
    ok &= require(f, "--tx-height-m");
    double rx_height = arg_f(argc, argv, "--rx-height-m", &f);
    ok &= require(f, "--rx-height-m");
    double freq = arg_f(argc, argv, "--freq-mhz", &f);
    ok &= require(f, "--freq-mhz");
    double tx_power_w = arg_f(argc, argv, "--tx-power-w", &f);
    ok &= require(f, "--tx-power-w");
    double tx_gain = arg_f(argc, argv, "--tx-gain-dbi", &f);
    ok &= require(f, "--tx-gain-dbi");
    double rx_gain = arg_f(argc, argv, "--rx-gain-dbi", &f);
    ok &= require(f, "--rx-gain-dbi");
    double rx_sensitivity = arg_f(argc, argv, "--rx-sensitivity-dbm", &f);
    ok &= require(f, "--rx-sensitivity-dbm");
    if (!ok || width <= 0 || height <= 0 || resolution <= 0 || freq <= 0 || tx_power_w <= 0)
        return 2;

    std::vector<int16_t> surface((size_t)width * (size_t)height);
    FILE *fp = fopen(surface_path, "rb");
    if (!fp) {
        fprintf(stderr, "failed to open %s\n", surface_path);
        return 1;
    }
    size_t n = fread(surface.data(), sizeof(int16_t), surface.size(), fp);
    fclose(fp);
    if (n != surface.size()) {
        fprintf(stderr, "short read from %s\n", surface_path);
        return 1;
    }

    double tx_ground = (double)sample_nearest(surface, width, height, min_x, max_y,
                                             resolution, tx_x, tx_y);
    double tx_z = tx_ground + tx_height;
    double tx_dbm = 10.0 * log10(tx_power_w) + 30.0 + tx_gain + rx_gain;
    std::vector<int16_t> signal(surface.size());
    std::vector<uint8_t> mask(surface.size());
    int covered = 0;

    for (int row = 0; row < height; row++) {
        double y = max_y - (double)row * resolution;
        for (int col = 0; col < width; col++) {
            double x = min_x + (double)col * resolution;
            size_t idx = (size_t)row * (size_t)width + (size_t)col;
            double dx = x - tx_x;
            double dy = y - tx_y;
            double dist_m = std::max(1.0, sqrt(dx * dx + dy * dy));
            double rx_z = (double)surface[idx] + rx_height;
            double fspl = 32.44 + 20.0 * log10(freq) + 20.0 * log10(dist_m / 1000.0);
            double penalty = obstructed(surface, width, height, min_x, max_y,
                                        resolution, tx_x, tx_y, tx_z, x, y, rx_z)
                                 ? 30.0
                                 : 0.0;
            double dbm = tx_dbm - fspl - penalty;
            int value = (int)llround(dbm * 10.0);
            signal[idx] = (int16_t)std::max(-32768, std::min(32767, value));
            mask[idx] = dbm >= rx_sensitivity ? 1 : 0;
            if (mask[idx])
                covered++;
        }
    }

    std::string base(out_prefix);
    fp = fopen((base + ".signal_i16le.bin").c_str(), "wb");
    if (!fp)
        return 1;
    fwrite(signal.data(), sizeof(int16_t), signal.size(), fp);
    fclose(fp);

    fp = fopen((base + ".mask_u8.bin").c_str(), "wb");
    if (!fp)
        return 1;
    fwrite(mask.data(), sizeof(uint8_t), mask.size(), fp);
    fclose(fp);

    fp = fopen((base + ".meta.json").c_str(), "wb");
    if (!fp)
        return 1;
    fprintf(fp,
            "{\n"
            "  \"model\": \"prototype_fspl_los\",\n"
            "  \"width\": %d,\n"
            "  \"height\": %d,\n"
            "  \"resolution_m\": %.6f,\n"
            "  \"min_x\": %.6f,\n"
            "  \"max_y\": %.6f,\n"
            "  \"signal_scale\": \"dbm_x10_i16\",\n"
            "  \"mask_value\": \"1 means dbm >= rx_sensitivity_dbm\",\n"
            "  \"rx_sensitivity_dbm\": %.3f,\n"
            "  \"covered_cells\": %d,\n"
            "  \"total_cells\": %d\n"
            "}\n",
            width, height, resolution, min_x, max_y, rx_sensitivity, covered,
            width * height);
    fclose(fp);

    fprintf(stderr, "wrote %s.{signal_i16le.bin,mask_u8.bin,meta.json}\n", out_prefix);
    return 0;
}
