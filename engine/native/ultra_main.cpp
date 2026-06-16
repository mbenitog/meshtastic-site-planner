/* Native runner for the backend ultra DSM path.
 *
 * This intentionally does not touch the SPLAT parity CLI. It consumes the
 * projected 2.5 m surface artifact produced by backend/ultra and writes an RF
 * raster using SPLAT!'s point_to_point_ITM model over measured terrain profiles.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

void point_to_point_ITM(double elev[], double tht_m, double rht_m,
                        double eps_dielect, double sgm_conductivity,
                        double eno_ns_surfref, double frq_mhz,
                        int radio_climate, int pol, double conf, double rel,
                        double &dbloss, char *strmode, int &errnum);

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

static int arg_i(int argc, char **argv, const char *name, bool *found) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], name) == 0) {
            if (found)
                *found = true;
            return (int)atof(argv[i + 1]);
        }
    }
    if (found)
        *found = false;
    return 0;
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

static inline int clamp_i(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

struct ItmScratch {
    std::vector<double> elev;
    char mode[100];
};

static double run_itm_path(const std::vector<int16_t> &surface, int width,
                           int height, double resolution,
                           int tx_col, int tx_row,
                           double tx_height, int rx_col, int rx_row,
                           double rx_height, double freq, double tx_power_w,
                           double tx_gain, double rx_gain,
                           double eps_dielect, double conductivity,
                           double bend, int climate, int polarization,
                           double conf, double rel, int err_counts[6],
                           ItmScratch &scratch) {
    int dx_cells = rx_col - tx_col;
    int dy_cells = rx_row - tx_row;
    double dist_cells = std::sqrt((double)dx_cells * (double)dx_cells +
                                  (double)dy_cells * (double)dy_cells);
    double dist = std::max(resolution, dist_cells * resolution);
    int segments = std::max(1, (int)ceil(dist_cells));
    scratch.elev.resize((size_t)segments + 16);
    scratch.elev[0] = (double)segments;
    scratch.elev[1] = dist / (double)segments;

    for (int i = 0; i <= segments; i++) {
        double t = (double)i / (double)segments;
        int col = clamp_i((int)llround((double)tx_col + (double)dx_cells * t), 0, width - 1);
        int row = clamp_i((int)llround((double)tx_row + (double)dy_cells * t), 0, height - 1);
        scratch.elev[(size_t)i + 2] = (double)surface[(size_t)row * (size_t)width + (size_t)col];
    }
    for (size_t i = (size_t)segments + 3; i < scratch.elev.size(); i++)
        scratch.elev[i] = scratch.elev[(size_t)segments + 2];

    double loss = 0.0;
    int errnum = 0;
    point_to_point_ITM(scratch.elev.data(), tx_height, rx_height, eps_dielect,
                       conductivity, bend, freq, climate, polarization, conf,
                       rel, loss, scratch.mode, errnum);
    if (errnum >= 0 && errnum < 5)
        err_counts[errnum]++;
    else
        err_counts[5]++;

    double erp_w = tx_power_w * pow(10.0, tx_gain / 10.0);
    double rxp_w = erp_w / pow(10.0, (loss - 2.14) / 10.0);
    return 10.0 * log10(rxp_w * 1000.0) + rx_gain;
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
    int tile_x0 = arg_i(argc, argv, "--tile-x0", &f);
    int tile_y0 = arg_i(argc, argv, "--tile-y0", &f);
    int tile_w = arg_i(argc, argv, "--tile-w", &f);
    int tile_h = arg_i(argc, argv, "--tile-h", &f);
    if (!f) tile_w = width - tile_x0;
    if (!f) tile_h = height - tile_y0;
    if (tile_x0 < 0 || tile_y0 < 0 || tile_w <= 0 || tile_h <= 0 ||
        tile_x0 + tile_w > width || tile_y0 + tile_h > height) {
        fprintf(stderr, "invalid tile bounds\n");
        return 2;
    }
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
    double eps_dielect = arg_f(argc, argv, "--dielect", &f);
    if (!f)
        eps_dielect = 15.0;
    double conductivity = arg_f(argc, argv, "--conductivity", &f);
    if (!f)
        conductivity = 0.005;
    double bend = arg_f(argc, argv, "--bend", &f);
    if (!f)
        bend = 301.0;
    int climate = (int)arg_f(argc, argv, "--climate", &f);
    if (!f)
        climate = 5;
    int polarization = (int)arg_f(argc, argv, "--pol", &f);
    if (!f)
        polarization = 1;
    double conf = arg_f(argc, argv, "--conf", &f);
    if (!f)
        conf = 0.95;
    double rel = arg_f(argc, argv, "--rel", &f);
    if (!f)
        rel = 0.95;
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

    std::vector<int16_t> signal(surface.size());
    std::vector<uint8_t> mask(surface.size());
    int covered = 0;
    int err_counts[6] = {0, 0, 0, 0, 0, 0};
    int tx_col = clamp_i((int)llround((tx_x - min_x) / resolution), 0, width - 1);
    int tx_row = clamp_i((int)llround((max_y - tx_y) / resolution), 0, height - 1);
    ItmScratch scratch;

    for (int row = tile_y0; row < tile_y0 + tile_h; row++) {
        for (int col = tile_x0; col < tile_x0 + tile_w; col++) {
            size_t idx = (size_t)row * (size_t)width + (size_t)col;
            double dbm = run_itm_path(surface, width, height, resolution,
                                      tx_col, tx_row, tx_height, col, row,
                                      rx_height, freq, tx_power_w, tx_gain,
                                      rx_gain, eps_dielect, conductivity, bend,
                                      climate, polarization, conf, rel,
                                      err_counts, scratch);
            int value = (int)llround(dbm * 10.0);
            signal[idx] = (int16_t)std::max(-32768, std::min(32767, value));
            mask[idx] = dbm >= rx_sensitivity ? 1 : 0;
            if (mask[idx])
                covered++;
        }
    }

    std::string base(out_prefix);
    char signal_suffix[64];
    snprintf(signal_suffix, sizeof(signal_suffix), "_x%d_y%d.signal_i16le.bin", tile_x0, tile_y0);
    char mask_suffix[64];
    snprintf(mask_suffix, sizeof(mask_suffix), "_x%d_y%d.mask_u8.bin", tile_x0, tile_y0);
    char meta_suffix[64];
    snprintf(meta_suffix, sizeof(meta_suffix), "_x%d_y%d.meta.json", tile_x0, tile_y0);
    fp = fopen((base + signal_suffix).c_str(), "wb");
    if (!fp)
        return 1;
    fwrite(signal.data() + (size_t)tile_y0 * (size_t)width + (size_t)tile_x0,
           sizeof(int16_t), (size_t)tile_w * (size_t)tile_h, fp);
    fclose(fp);

    fp = fopen((base + mask_suffix).c_str(), "wb");
    if (!fp)
        return 1;
    fwrite(mask.data() + (size_t)tile_y0 * (size_t)width + (size_t)tile_x0,
           sizeof(uint8_t), (size_t)tile_w * (size_t)tile_h, fp);
    fclose(fp);

    fp = fopen((base + meta_suffix).c_str(), "wb");
    if (!fp)
        return 1;
    fprintf(fp,
            "{\n"
            "  \"model\": \"itm_projected_grid\",\n"
            "  \"width\": %d,\n"
            "  \"height\": %d,\n"
            "  \"tile_x0\": %d,\n"
            "  \"tile_y0\": %d,\n"
            "  \"tile_w\": %d,\n"
            "  \"tile_h\": %d,\n"
            "  \"resolution_m\": %.6f,\n"
            "  \"min_x\": %.6f,\n"
            "  \"max_y\": %.6f,\n"
            "  \"signal_scale\": \"dbm_x10_i16\",\n"
            "  \"mask_value\": \"1 means dbm >= rx_sensitivity_dbm\",\n"
            "  \"rx_sensitivity_dbm\": %.3f,\n"
            "  \"covered_cells\": %d,\n"
            "  \"total_cells\": %d,\n"
            "  \"itm_errnums\": [%d, %d, %d, %d, %d, %d]\n"
            "}\n",
            width, height, tile_x0, tile_y0, tile_w, tile_h,
            resolution, min_x, max_y,
            rx_sensitivity, covered,
            tile_w * tile_h, err_counts[0], err_counts[1], err_counts[2],
            err_counts[3], err_counts[4], err_counts[5]);
    fclose(fp);

    fprintf(stderr, "wrote tile x0=%d y0=%d w=%d h=%d into %s.*\n",
            tile_x0, tile_y0, tile_w, tile_h, out_prefix);
    return 0;
}
