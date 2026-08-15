/* OpenCL C port of SPLAT! ITM (Longley-Rice) point-to-point model.
 *
 * This is the ITM path of splat/itwom3.0.cpp (point_to_point_ITM and the
 * functions it transitively calls: qlrps, qlra, qlrpfl, hzns, z1sq1, d1thx,
 * qtile, lrprop, alos, adiff, ascat, avar, curve, aknfe, fht, h0f, ahd,
 * qerfi, qerf, abq_alos). The ITWOM "2" variants (alos2, adiff2, lrprop2,
 * saalos, ...) are NOT used by point_to_point_ITM and are intentionally
 * omitted.
 *
 * Precision is selected at build time:
 *   -DUSE_DOUBLE  -> real_t = double   (requires cl_khr_fp64)
 *   (undefined)   -> real_t = float
 *
 * Porting notes (see docs/gpu-ultra-backend.md):
 *   * C++ function-scope `static` locals are forbidden in OpenCL C and are
 *     hoisted into an `itm_state` struct held privately by each work-item.
 *     This is semantically equivalent because ultra_main resets prop.mdp=-1
 *     on every call, so every point_to_point_ITM invocation re-initialises
 *     all of that state before use.
 *   * d1thx's dynamic `new[]` is replaced by a fixed-size __global scratch
 *     buffer (one per work-item). Max scratch entries = 256 (n = 10*ka-5,
 *     ka in [4,25] -> n in [35,245]).
 *   * complex<double> is represented as real2_t (.x = re, .y = im) with
 *     hand-written cdiv/csqrtc/cabq/cabsv helpers, to avoid relying on
 *     complex built-ins whose availability/behaviour varies across drivers.
 *   * bool -> int (OpenCL C 1.2 has no bool).
 *   * No -cl-fast-relaxed-math: kept IEEE-strict for cross-vendor parity.
 *
 * OpenCL C 1.2 is the assumed baseline.
 */

/* Disable FMA contraction: the CPU reference (splat/itwom3.0.cpp compiled
 * with g++/clang++ -O2) does NOT contract a*b+c into fma(a,b,c), so to
 * match it bit-for-bit the GPU kernel must not either.  Without this
 * pragma some compilers contract FP64 expressions when the kernel is
 * large enough to allow cross-function inlining, producing subtly
 * different roundings. */
#pragma OPENCL FP_CONTRACT off

#ifdef USE_DOUBLE
#  define real_t double
#  define real2_t double2
#else
#  define real_t float
#  define real2_t float2
#endif

#define THIRD ((real_t)0.3333333333333333)

/* ----------------------------------------------------------------------- */
/* complex helpers (real2_t = (re, im))                                    */
/* ----------------------------------------------------------------------- */

static real_t cabq(real2_t r) { return r.x*r.x + r.y*r.y; }         /* |r|^2 (abq_alos) */
static real_t cabsv(real2_t r) { return sqrt(r.x*r.x + r.y*r.y); }  /* |r|  (abs)        */
static real2_t cdiv(real2_t a, real2_t b)
{
    real_t d = b.x*b.x + b.y*b.y;
    return (real2_t)((a.x*b.x + a.y*b.y) / d, (a.y*b.x - a.x*b.y) / d);
}
static real2_t csqrtc(real2_t z)
{
    real_t r = sqrt(z.x*z.x + z.y*z.y);
    real_t re = sqrt(fmax((real_t)0.0, (r + z.x) * (real_t)0.5));
    real_t im = sqrt(fmax((real_t)0.0, (r - z.x) * (real_t)0.5));
    if (z.y < (real_t)0.0) im = -im;
    return (real2_t)(re, im);
}

/* ----------------------------------------------------------------------- */
/* min/max (faithful if/else, no fmin/fmax NaN semantics)                  */
/* ----------------------------------------------------------------------- */

static int imin(int i, int j) { return i < j ? i : j; }
static int imax(int i, int j) { return i > j ? i : j; }
static real_t rmin(real_t a, real_t b) { return a < b ? a : b; }
static real_t rmax(real_t a, real_t b) { return a > b ? a : b; }

/* FORTRAN_DIM(x,y) = x-y if x>y else 0  (== fdim) */
#define FDIM(x,y) ((x) > (y) ? ((x) - (y)) : (real_t)0.0)

/* ----------------------------------------------------------------------- */
/* structs (mirror splat/itwom3.0.cpp, double -> real_t)                   */
/* ----------------------------------------------------------------------- */

typedef struct {
    real_t aref;
    real_t dist;
    real_t hg[2];
    real_t rch[2];
    real_t wn;
    real_t dh;
    real_t dhd;
    real_t ens;
    real_t encc;
    real_t cch;
    real_t cd;
    real_t gme;
    real_t zgndreal;
    real_t zgndimag;
    real_t he[2];
    real_t dl[2];
    real_t the[2];
    real_t tiw;
    real_t ght;
    real_t ghr;
    real_t rph;
    real_t hht;
    real_t hhr;
    real_t tgh;
    real_t tsgh;
    real_t thera;
    real_t thenr;
    int rpl;
    int kwx;
    int mdp;
    int ptx;
    int los;
} prop_type;

typedef struct {
    real_t sgc;
    int lvar;
    int mdvar;
    int klim;
} propv_type;

typedef struct {
    real_t dlsa;
    real_t dx;
    real_t ael;
    real_t ak1;
    real_t ak2;
    real_t aed;
    real_t emd;
    real_t aes;
    real_t ems;
    real_t dls[2];
    real_t dla;
    real_t tha;
} propa_type;

typedef struct {
    /* adiff */
    real_t wd1, xd1, afo, qk, aht, xht;
    /* alos */
    real_t wls;
    /* ascat */
    real_t ad, rr, etq, h0s;
    /* lrprop */
    int wlos, wscat;
    real_t dmin, xae;
} itm_state;

/* ----------------------------------------------------------------------- */
/* pure math helpers                                                       */
/* ----------------------------------------------------------------------- */

static real_t aknfe(real_t v2)
{
    if (v2 < (real_t)5.76)
        return (real_t)6.02 + (real_t)9.11 * sqrt(v2) - (real_t)1.27 * v2;
    return (real_t)12.953 + (real_t)10.0 * log10(v2);
}

static real_t fht(real_t x, real_t pk)
{
    real_t w, fhtv;
    if (x < (real_t)200.0) {
        w = -log(pk);
        if (pk < (real_t)1.0e-5 || x*w*w*w > (real_t)5495.0) {
            fhtv = (real_t)-117.0;
            if (x > (real_t)1.0)
                fhtv = (real_t)40.0 * log10(x) + fhtv;
        } else {
            fhtv = (real_t)2.5e-5 * x * x / pk - (real_t)8.686 * w - (real_t)15.0;
        }
    } else {
        fhtv = (real_t)0.05751 * x - (real_t)10.0 * log10(x);
        if (x < (real_t)2000.0) {
            w = (real_t)0.0134 * x * exp((real_t)-0.005 * x);
            fhtv = ((real_t)1.0 - w) * fhtv + w * ((real_t)40.0 * log10(x) - (real_t)117.0);
        }
    }
    return fhtv;
}

static real_t h0f(real_t r, real_t et)
{
    real_t a[5] = {(real_t)25.0, (real_t)80.0, (real_t)177.0, (real_t)395.0, (real_t)705.0};
    real_t b[5] = {(real_t)24.0, (real_t)45.0, (real_t)68.0, (real_t)80.0, (real_t)105.0};
    real_t q, x, h0fv, temp;
    int it = (int)et;
    if (it <= 0) { it = 1; q = (real_t)0.0; }
    else if (it >= 5) { it = 5; q = (real_t)0.0; }
    else q = et - it;
    temp = (real_t)1.0 / r;
    x = temp * temp;
    h0fv = (real_t)4.343 * log((a[it-1]*x + b[it-1]) * x + (real_t)1.0);
    if (q != (real_t)0.0)
        h0fv = ((real_t)1.0 - q) * h0fv + q * (real_t)4.343 * log((a[it]*x + b[it]) * x + (real_t)1.0);
    return h0fv;
}

static real_t ahd(real_t td)
{
    int i;
    real_t a[3] = {(real_t)133.4, (real_t)104.6, (real_t)71.8};
    real_t b[3] = {(real_t)0.332e-3, (real_t)0.212e-3, (real_t)0.157e-3};
    real_t c[3] = {(real_t)-4.343, (real_t)-1.086, (real_t)2.171};
    if (td <= (real_t)10e3) i = 0;
    else if (td <= (real_t)70e3) i = 1;
    else i = 2;
    return a[i] + b[i]*td + c[i]*log(td);
}

static real_t curve(real_t c1, real_t c2, real_t x1, real_t x2, real_t x3, real_t de)
{
    real_t temp1 = (de - x2) / x3;
    real_t temp2 = de / x1;
    temp1 *= temp1;
    temp2 *= temp2;
    return (c1 + c2 / ((real_t)1.0 + temp1)) * temp2 / ((real_t)1.0 + temp2);
}

static real_t qerfi(real_t q)
{
    real_t x, t, v;
    real_t c0=(real_t)2.515516698, c1=(real_t)0.802853, c2=(real_t)0.010328;
    real_t d1=(real_t)1.432788, d2=(real_t)0.189269, d3=(real_t)0.001308;
    x = (real_t)0.5 - q;
    t = rmax((real_t)0.5 - fabs(x), (real_t)0.000001);
    t = sqrt((real_t)-2.0 * log(t));
    v = t - ((c2*t + c1)*t + c0) / (((d3*t + d2)*t + d1)*t + (real_t)1.0);
    if (x < (real_t)0.0) v = -v;
    return v;
}

static real_t qerf(real_t z)
{
    real_t b1=(real_t)0.319381530, b2=(real_t)-0.356563782, b3=(real_t)1.781477937;
    real_t b4=(real_t)-1.821255987, b5=(real_t)1.330274429;
    real_t rp=(real_t)4.317008, rrt2pi=(real_t)0.398942280;
    real_t t, x, qerfv;
    x = z;
    t = fabs(x);
    if (t >= (real_t)10.0) {
        qerfv = (real_t)0.0;
    } else {
        t = rp / (t + rp);
        qerfv = exp((real_t)-0.5 * x * x) * rrt2pi * ((((b5*t + b4)*t + b3)*t + b2)*t + b1) * t;
    }
    if (x < (real_t)0.0) qerfv = (real_t)1.0 - qerfv;
    return qerfv;
}

/* ----------------------------------------------------------------------- */
/* qlrps                                                                   */
/* ----------------------------------------------------------------------- */

static void qlrps(real_t fmhz, real_t zsys, real_t en0, int ipol,
                  real_t eps, real_t sgm, private prop_type* prop)
{
    real_t gma = (real_t)157e-9;
    prop->wn = fmhz / (real_t)47.7;
    prop->ens = en0;
    if (zsys != (real_t)0.0)
        prop->ens *= exp(-zsys / (real_t)9460.0);
    prop->gme = gma * ((real_t)1.0 - (real_t)0.04665 * exp(prop->ens / (real_t)179.3));
    real2_t zq = (real2_t)(eps, (real_t)376.62 * sgm / prop->wn);
    real2_t zgm1 = (real2_t)(zq.x - (real_t)1.0, zq.y);
    real2_t prop_zgnd = csqrtc(zgm1);
    if (ipol != 0)
        prop_zgnd = cdiv(prop_zgnd, zq);
    prop->zgndreal = prop_zgnd.x;
    prop->zgndimag = prop_zgnd.y;
}

/* ----------------------------------------------------------------------- */
/* alos / adiff / ascat (state in itm_state)                               */
/* ----------------------------------------------------------------------- */

static real_t alos(real_t d, private prop_type* prop, private propa_type* propa,
                   private itm_state* st)
{
    real2_t prop_zgnd = (real2_t)(prop->zgndreal, prop->zgndimag);
    real2_t r;
    real_t s, sps, q, alosv;
    if (d == (real_t)0.0) {
        st->wls = (real_t)0.021 / ((real_t)0.021 + prop->wn * prop->dh / rmax((real_t)10e3, propa->dlsa));
        alosv = (real_t)0.0;
    } else {
        q = ((real_t)1.0 - (real_t)0.8 * exp(-d / (real_t)50e3)) * prop->dh;
        s = (real_t)0.78 * q * exp(-pow(q / (real_t)16.0, (real_t)0.25));
        q = prop->he[0] + prop->he[1];
        sps = q / sqrt(d*d + q*q);
        real2_t num = (real2_t)(sps - prop_zgnd.x, -prop_zgnd.y);
        real2_t den = (real2_t)(sps + prop_zgnd.x,  prop_zgnd.y);
        r = cdiv(num, den) * exp(-rmin((real_t)10.0, prop->wn * s * sps));
        q = cabq(r);
        if (q < (real_t)0.25 || q < sps)
            r = r * sqrt(sps / q);
        alosv = propa->emd * d + propa->aed;
        q = prop->wn * prop->he[0] * prop->he[1] * (real_t)2.0 / d;
        if (q > (real_t)1.57)
            q = (real_t)3.14 - (real_t)2.4649 / q;
        alosv = ((real_t)-4.343 * log(cabq((real2_t)(cos(q), -sin(q)) + r)) - alosv) * st->wls + alosv;
    }
    return alosv;
}

static real_t adiff(real_t d, private prop_type* prop, private propa_type* propa,
                    private itm_state* st)
{
    real2_t prop_zgnd = (real2_t)(prop->zgndreal, prop->zgndimag);
    real_t a, q, pk, ds, th, wa, ar, wd, adiffv;
    if (d == (real_t)0.0) {
        q = prop->hg[0] * prop->hg[1];
        st->qk = prop->he[0] * prop->he[1] - q;
        if (prop->mdp < 0)
            q += (real_t)10.0;
        st->wd1 = sqrt((real_t)1.0 + st->qk / q);
        st->xd1 = propa->dla + propa->tha / prop->gme;
        q = ((real_t)1.0 - (real_t)0.8 * exp(-propa->dlsa / (real_t)50e3)) * prop->dh;
        q *= (real_t)0.78 * exp(-pow(q / (real_t)16.0, (real_t)0.25));
        st->afo = rmin((real_t)15.0, (real_t)2.171 * log((real_t)1.0 + (real_t)4.77e-4 * prop->hg[0] * prop->hg[1] * prop->wn * q));
        st->qk = (real_t)1.0 / cabsv(prop_zgnd);
        st->aht = (real_t)20.0;
        st->xht = (real_t)0.0;
        for (int j = 0; j < 2; ++j) {
            a = (real_t)0.5 * (prop->dl[j] * prop->dl[j]) / prop->he[j];
            wa = pow(a * prop->wn, THIRD);
            pk = st->qk / wa;
            q = ((real_t)1.607 - pk) * (real_t)151.0 * wa * prop->dl[j] / a;
            st->xht += q;
            st->aht += fht(q, pk);
        }
        adiffv = (real_t)0.0;
    } else {
        th = propa->tha + d * prop->gme;
        ds = d - propa->dla;
        q = (real_t)0.0795775 * prop->wn * ds * th * th;
        adiffv = aknfe(q * prop->dl[0] / (ds + prop->dl[0])) + aknfe(q * prop->dl[1] / (ds + prop->dl[1]));
        a = ds / th;
        wa = pow(a * prop->wn, THIRD);
        pk = st->qk / wa;
        q = ((real_t)1.607 - pk) * (real_t)151.0 * wa * th + st->xht;
        ar = (real_t)0.05751 * q - (real_t)4.343 * log(q) - st->aht;
        q = (st->wd1 + st->xd1 / d) * rmin(((real_t)1.0 - (real_t)0.8 * exp(-d / (real_t)50e3)) * prop->dh * prop->wn, (real_t)6283.2);
        wd = (real_t)25.1 / ((real_t)25.1 + sqrt(q));
        adiffv = ar * wd + ((real_t)1.0 - wd) * adiffv + st->afo;
    }
    return adiffv;
}

static real_t ascat(real_t d, private prop_type* prop, private propa_type* propa,
                    private itm_state* st)
{
    real_t h0, r1, r2, z0, ss, et, ett, th, q, ascatv, temp;
    if (d == (real_t)0.0) {
        st->ad = prop->dl[0] - prop->dl[1];
        st->rr = prop->he[1] / prop->rch[0];
        if (st->ad < (real_t)0.0) {
            st->ad = -st->ad;
            st->rr = (real_t)1.0 / st->rr;
        }
        st->etq = ((real_t)5.67e-6 * prop->ens - (real_t)2.32e-3) * prop->ens + (real_t)0.031;
        st->h0s = (real_t)-15.0;
        ascatv = (real_t)0.0;
    } else {
        if (st->h0s > (real_t)15.0) {
            h0 = st->h0s;
        } else {
            th = prop->the[0] + prop->the[1] + d * prop->gme;
            r2 = (real_t)2.0 * prop->wn * th;
            r1 = r2 * prop->he[0];
            r2 *= prop->he[1];
            if (r1 < (real_t)0.2 && r2 < (real_t)0.2)
                return (real_t)1001.0;
            ss = (d - st->ad) / (d + st->ad);
            q = st->rr / ss;
            ss = rmax((real_t)0.1, ss);
            q = rmin(rmax((real_t)0.1, q), (real_t)10.0);
            z0 = (d - st->ad) * (d + st->ad) * th * (real_t)0.25 / d;
            temp = rmin((real_t)1.7, z0 / (real_t)8.0e3);
            temp = temp*temp*temp*temp*temp*temp;
            et = (st->etq * exp(-temp) + (real_t)1.0) * z0 / (real_t)1.7556e3;
            ett = rmax(et, (real_t)1.0);
            h0 = (h0f(r1, ett) + h0f(r2, ett)) * (real_t)0.5;
            h0 += rmin(h0, ((real_t)1.38 - log(ett)) * log(ss) * log(q) * (real_t)0.49);
            h0 = FDIM(h0, (real_t)0.0);
            if (et < (real_t)1.0) {
                temp = ((real_t)1.0 + (real_t)1.4142/r1) * ((real_t)1.0 + (real_t)1.4142/r2);
                h0 = et*h0 + ((real_t)1.0 - et) * (real_t)4.343 * log((temp*temp) * (r1+r2) / (r1+r2+(real_t)2.8284));
            }
            if (h0 > (real_t)15.0 && st->h0s >= (real_t)0.0)
                h0 = st->h0s;
        }
        st->h0s = h0;
        th = propa->tha + d * prop->gme;
        ascatv = ahd(th*d) + (real_t)4.343 * log((real_t)47.7 * prop->wn * (th*th*th*th))
                 - (real_t)0.1 * (prop->ens - (real_t)301.0) * exp(-th*d / (real_t)40e3) + h0;
    }
    return ascatv;
}

/* ----------------------------------------------------------------------- */
/* lrprop                                                                  */
/* ----------------------------------------------------------------------- */

static void lrprop(real_t d, private prop_type* prop, private propa_type* propa,
                   private itm_state* st)
{
    real_t a0, a1, a2, a3, a4, a5, a6;
    real_t d0, d1, d2, d3, d4, d5, d6;
    int wq;
    real_t q;
    int j;

    if (prop->mdp != 0) {
        for (j = 0; j < 2; j++)
            propa->dls[j] = sqrt((real_t)2.0 * prop->he[j] / prop->gme);
        propa->dlsa = propa->dls[0] + propa->dls[1];
        propa->dla = prop->dl[0] + prop->dl[1];
        propa->tha = rmax(prop->the[0] + prop->the[1], -propa->dla * prop->gme);
        st->wlos = 0;
        st->wscat = 0;
        if (prop->wn < (real_t)0.838 || prop->wn > (real_t)210.0)
            prop->kwx = imax(prop->kwx, 1);
        for (j = 0; j < 2; j++)
            if (prop->hg[j] < (real_t)1.0 || prop->hg[j] > (real_t)1000.0)
                prop->kwx = imax(prop->kwx, 1);
        for (j = 0; j < 2; j++)
            if (fabs(prop->the[j]) > (real_t)200e-3 || prop->dl[j] < (real_t)0.1 * propa->dls[j] || prop->dl[j] > (real_t)3.0 * propa->dls[j])
                prop->kwx = imax(prop->kwx, 3);
        {
            real2_t prop_zgnd = (real2_t)(prop->zgndreal, prop->zgndimag);
            if (prop->ens < (real_t)250.0 || prop->ens > (real_t)400.0 || prop->gme < (real_t)75e-9 || prop->gme > (real_t)250e-9
                || prop_zgnd.x <= fabs(prop_zgnd.y) /* real() <= abs(imag()) */
                || prop->wn < (real_t)0.419 || prop->wn > (real_t)420.0)
                prop->kwx = 4;
        }
        for (j = 0; j < 2; j++)
            if (prop->hg[j] < (real_t)0.5 || prop->hg[j] > (real_t)3000.0)
                prop->kwx = 4;
        st->dmin = fabs(prop->he[0] - prop->he[1]) / (real_t)200e-3;
        q = adiff((real_t)0.0, prop, propa, st);
        st->xae = pow(prop->wn * (prop->gme * prop->gme), -THIRD);
        d3 = rmax(propa->dlsa, (real_t)1.3787 * st->xae + propa->dla);
        d4 = d3 + (real_t)2.7574 * st->xae;
        a3 = adiff(d3, prop, propa, st);
        a4 = adiff(d4, prop, propa, st);
        propa->emd = (a4 - a3) / (d4 - d3);
        propa->aed = a3 - propa->emd * d3;
    }

    if (prop->mdp >= 0) {
        prop->mdp = 0;
        prop->dist = d;
    }

    if (prop->dist > (real_t)0.0) {
        if (prop->dist > (real_t)1000e3)
            prop->kwx = imax(prop->kwx, 1);
        if (prop->dist < st->dmin)
            prop->kwx = imax(prop->kwx, 3);
        if (prop->dist < (real_t)1e3 || prop->dist > (real_t)2000e3)
            prop->kwx = 4;
    }

    if (prop->dist < propa->dlsa) {
        if (!st->wlos) {
            q = alos((real_t)0.0, prop, propa, st);
            d2 = propa->dlsa;
            a2 = propa->aed + d2 * propa->emd;
            d0 = (real_t)1.908 * prop->wn * prop->he[0] * prop->he[1];
            if (propa->aed >= (real_t)0.0) {
                d0 = rmin(d0, (real_t)0.5 * propa->dla);
                d1 = d0 + (real_t)0.25 * (propa->dla - d0);
            } else {
                d1 = rmax(-propa->aed / propa->emd, (real_t)0.25 * propa->dla);
            }
            a1 = alos(d1, prop, propa, st);
            wq = 0;
            if (d0 < d1) {
                a0 = alos(d0, prop, propa, st);
                q = log(d2 / d0);
                propa->ak2 = rmax((real_t)0.0, ((d2-d0)*(a1-a0) - (d1-d0)*(a2-a0)) / ((d2-d0)*log(d1/d0) - (d1-d0)*q));
                wq = (propa->aed >= (real_t)0.0) || (propa->ak2 > (real_t)0.0);
                if (wq) {
                    propa->ak1 = (a2 - a0 - propa->ak2 * q) / (d2 - d0);
                    if (propa->ak1 < (real_t)0.0) {
                        propa->ak1 = (real_t)0.0;
                        propa->ak2 = FDIM(a2, a0) / q;
                        if (propa->ak2 == (real_t)0.0)
                            propa->ak1 = propa->emd;
                    }
                } else {
                    propa->ak2 = (real_t)0.0;
                    propa->ak1 = (a2 - a1) / (d2 - d1);
                    if (propa->ak1 <= (real_t)0.0)
                        propa->ak1 = propa->emd;
                }
            } else {
                propa->ak1 = (a2 - a1) / (d2 - d1);
                propa->ak2 = (real_t)0.0;
                if (propa->ak1 <= (real_t)0.0)
                    propa->ak1 = propa->emd;
            }
            propa->ael = a2 - propa->ak1 * d2 - propa->ak2 * log(d2);
            st->wlos = 1;
        }
        if (prop->dist > (real_t)0.0)
            prop->aref = propa->ael + propa->ak1 * prop->dist + propa->ak2 * log(prop->dist);
    }

    if (prop->dist <= (real_t)0.0 || prop->dist >= propa->dlsa) {
        if (!st->wscat) {
            q = ascat((real_t)0.0, prop, propa, st);
            d5 = propa->dla + (real_t)200e3;
            d6 = d5 + (real_t)200e3;
            a6 = ascat(d6, prop, propa, st);
            a5 = ascat(d5, prop, propa, st);
            if (a5 < (real_t)1000.0) {
                propa->ems = (a6 - a5) / (real_t)200e3;
                propa->dx = rmax(propa->dlsa, rmax(propa->dla + (real_t)0.3 * st->xae * log((real_t)47.7 * prop->wn),
                                  (a5 - propa->aed - propa->ems * d5) / (propa->emd - propa->ems)));
                propa->aes = (propa->emd - propa->ems) * propa->dx + propa->aed;
            } else {
                propa->ems = propa->emd;
                propa->aes = propa->aed;
                propa->dx = (real_t)10e6;
            }
            st->wscat = 1;
        }
        if (prop->dist > propa->dx)
            prop->aref = propa->aes + propa->ems * prop->dist;
        else
            prop->aref = propa->aed + propa->emd * prop->dist;
    }

    prop->aref = rmax(prop->aref, (real_t)0.0);
}

/* ----------------------------------------------------------------------- */
/* hzns / z1sq1 / qtile / d1thx  (operate on __global profile/scratch)     */
/* ----------------------------------------------------------------------- */

static void hzns(__global real_t* pfl, private prop_type* prop)
{
    int np = (int)pfl[0];
    real_t xi = pfl[1];
    real_t za = pfl[2] + prop->hg[0];
    real_t zb = pfl[np+2] + prop->hg[1];
    real_t qc = (real_t)0.5 * prop->gme;
    real_t q = qc * prop->dist;
    int wq;
    real_t sa, sb;
    prop->the[1] = (zb - za) / prop->dist;
    prop->the[0] = prop->the[1] - q;
    prop->the[1] = -prop->the[1] - q;
    prop->dl[0] = prop->dist;
    prop->dl[1] = prop->dist;
    if (np >= 2) {
        sa = (real_t)0.0;
        sb = prop->dist;
        wq = 1;
        for (int i = 1; i < np; i++) {
            sa += xi;
            sb -= xi;
            q = pfl[i+2] - (qc*sa + prop->the[0])*sa - za;
            if (q > (real_t)0.0) {
                prop->the[0] += q / sa;
                prop->dl[0] = sa;
                wq = 0;
            }
            if (!wq) {
                q = pfl[i+2] - (qc*sb + prop->the[1])*sb - zb;
                if (q > (real_t)0.0) {
                    prop->the[1] += q / sb;
                    prop->dl[1] = sb;
                }
            }
        }
    }
}

static void z1sq1(__global real_t* z, real_t x1, real_t x2,
                  private real_t* z0, private real_t* zn)
{
    real_t xn, xa, xb, x, a, b;
    int n, ja, jb;
    xn = z[0];
    xa = (int)FDIM(x1 / z[1], (real_t)0.0);
    xb = xn - (int)FDIM(xn, x2 / z[1]);
    if (xb <= xa) {
        xa = FDIM(xa, (real_t)1.0);
        xb = xn - FDIM(xn, xb + (real_t)1.0);
    }
    ja = (int)xa;
    jb = (int)xb;
    n = jb - ja;
    xa = xb - xa;
    x = (real_t)-0.5 * xa;
    xb += x;
    a = (real_t)0.5 * (z[ja+2] + z[jb+2]);
    b = (real_t)0.5 * (z[ja+2] - z[jb+2]) * x;
    for (int i = 2; i <= n; ++i) {
        ++ja;
        x += (real_t)1.0;
        a += z[ja+2];
        b += z[ja+2] * x;
    }
    a /= xa;
    b = b * (real_t)12.0 / ((xa*xa + (real_t)2.0) * xa);
    *z0 = a - b * xb;
    *zn = a + b * (xn - xb);
}

static real_t qtile(int nn, __global real_t* a, int ir)
{
    real_t q = (real_t)0.0, r;
    int m, n, i, j, j1 = 0, i0 = 0, k;
    int done = 0;
    int goto10 = 1;
    m = 0;
    n = nn;
    k = imin(imax(0, ir), n);
    while (!done) {
        if (goto10) {
            q = a[k];
            i0 = m;
            j1 = n;
        }
        i = i0;
        while (i <= n && a[i] >= q) i++;
        if (i > n) i = n;
        j = j1;
        while (j >= m && a[j] <= q) j--;
        if (j < m) j = m;
        if (i < j) {
            r = a[i]; a[i] = a[j]; a[j] = r;
            i0 = i + 1;
            j1 = j - 1;
            goto10 = 0;
        } else if (i < k) {
            a[k] = a[i];
            a[i] = q;
            m = i + 1;
            goto10 = 1;
        } else if (j > k) {
            a[k] = a[j];
            a[j] = q;
            n = j - 1;
            goto10 = 1;
        } else {
            done = 1;
        }
    }
    return q;
}

#define D1THX_SCRATCH_LEN 256   /* n = 10*ka-5, ka in [4,25] -> n in [35,245]; +2 */

static real_t d1thx(__global real_t* pfl, real_t x1, real_t x2,
                    __global real_t* s)
{
    int np, ka, kb, n, k, j;
    real_t d1thxv, sn, xa, xb;
    np = (int)pfl[0];
    xa = x1 / pfl[1];
    xb = x2 / pfl[1];
    d1thxv = (real_t)0.0;
    if (xb - xa < (real_t)2.0)
        return d1thxv;
    ka = (int)((real_t)0.1 * (xb - xa + (real_t)8.0));
    ka = imin(imax(4, ka), 25);
    n = 10*ka - 5;
    kb = n - ka + 1;
    sn = n - 1;
    s[0] = sn;
    s[1] = (real_t)1.0;
    xb = (xb - xa) / sn;
    k = (int)(xa + (real_t)1.0);
    xa -= (real_t)k;
    for (j = 0; j < n; j++) {
        while (xa > (real_t)0.0 && k < np) {
            xa -= (real_t)1.0;
            ++k;
        }
        s[j+2] = pfl[k+2] + (pfl[k+2] - pfl[k+1]) * xa;
        xa = xa + xb;
    }
    {
        real_t za, zb;
        z1sq1(s, (real_t)0.0, sn, &za, &zb);
        xa = za; xb = zb;
    }
    xb = (xb - xa) / sn;
    for (j = 0; j < n; j++) {
        s[j+2] -= xa;
        xa = xa + xb;
    }
    d1thxv = qtile(n-1, s+2, ka-1) - qtile(n-1, s+2, kb-1);
    d1thxv /= (real_t)1.0 - (real_t)0.8 * exp(-(x2 - x1) / (real_t)50.0e3);
    return d1thxv;
}

/* ----------------------------------------------------------------------- */
/* qlra / qlrpfl                                                           */
/* ----------------------------------------------------------------------- */

static void qlra(int kst[2], int klimx, int mdvarx,
                 private prop_type* prop, private propv_type* propv)
{
    real_t q;
    for (int j = 0; j < 2; ++j) {
        if (kst[j] <= 0) {
            prop->he[j] = prop->hg[j];
        } else {
            q = (real_t)4.0;
            if (kst[j] != 1) q = (real_t)9.0;
            if (prop->hg[j] < (real_t)5.0)
                q *= sin((real_t)0.3141593 * prop->hg[j]);
            prop->he[j] = prop->hg[j] + ((real_t)1.0 + q) * exp(-rmin((real_t)20.0, (real_t)2.0 * prop->hg[j] / rmax((real_t)1e-3, prop->dh)));
        }
        q = sqrt((real_t)2.0 * prop->he[j] / prop->gme);
        prop->dl[j] = q * exp((real_t)-0.07 * sqrt(prop->dh / rmax(prop->he[j], (real_t)5.0)));
        prop->the[j] = ((real_t)0.65 * prop->dh * (q / prop->dl[j] - (real_t)1.0) - (real_t)2.0 * prop->he[j]) / q;
    }
    prop->mdp = 1;
    propv->lvar = imax(propv->lvar, 3);
    if (mdvarx >= 0) {
        propv->mdvar = mdvarx;
        propv->lvar = imax(propv->lvar, 4);
    }
    if (klimx > 0) {
        propv->klim = klimx;
        propv->lvar = 5;
    }
}

static void qlrpfl(__global real_t* pfl, int klimx, int mdvarx,
                   private prop_type* prop, private propa_type* propa,
                   private propv_type* propv, private itm_state* st,
                   __global real_t* scratch)
{
    int np, j;
    real_t xl[2], q, za, zb, temp;
    prop->dist = pfl[0] * pfl[1];
    np = (int)pfl[0];
    hzns(pfl, prop);
    for (j = 0; j < 2; j++)
        xl[j] = rmin((real_t)15.0 * prop->hg[j], (real_t)0.1 * prop->dl[j]);
    xl[1] = prop->dist - xl[1];
    prop->dh = d1thx(pfl, xl[0], xl[1], scratch);
    if (prop->dl[0] + prop->dl[1] > (real_t)1.5 * prop->dist) {
        z1sq1(pfl, xl[0], xl[1], &za, &zb);
        prop->he[0] = prop->hg[0] + FDIM(pfl[2], za);
        prop->he[1] = prop->hg[1] + FDIM(pfl[np+2], zb);
        for (j = 0; j < 2; j++)
            prop->dl[j] = sqrt((real_t)2.0 * prop->he[j] / prop->gme) * exp((real_t)-0.07 * sqrt(prop->dh / rmax(prop->he[j], (real_t)5.0)));
        q = prop->dl[0] + prop->dl[1];
        if (q <= prop->dist) {
            temp = prop->dist / q;
            q = temp * temp;
            for (j = 0; j < 2; j++) {
                prop->he[j] *= q;
                prop->dl[j] = sqrt((real_t)2.0 * prop->he[j] / prop->gme) * exp((real_t)-0.07 * sqrt(prop->dh / rmax(prop->he[j], (real_t)5.0)));
            }
        }
        for (j = 0; j < 2; j++) {
            q = sqrt((real_t)2.0 * prop->he[j] / prop->gme);
            prop->the[j] = ((real_t)0.65 * prop->dh * (q / prop->dl[j] - (real_t)1.0) - (real_t)2.0 * prop->he[j]) / q;
        }
    } else {
        z1sq1(pfl, xl[0], (real_t)0.9 * prop->dl[0], &za, &q);
        z1sq1(pfl, prop->dist - (real_t)0.9 * prop->dl[1], xl[1], &q, &zb);
        prop->he[0] = prop->hg[0] + FDIM(pfl[2], za);
        prop->he[1] = prop->hg[1] + FDIM(pfl[np+2], zb);
    }
    prop->mdp = -1;
    propv->lvar = imax(propv->lvar, 3);
    if (mdvarx >= 0) {
        propv->mdvar = mdvarx;
        propv->lvar = imax(propv->lvar, 4);
    }
    if (klimx > 0) {
        propv->klim = klimx;
        propv->lvar = 5;
    }
    lrprop((real_t)0.0, prop, propa, st);
}

/* ----------------------------------------------------------------------- */
/* avar (single call per point_to_point_ITM; statics -> locals)            */
/* ----------------------------------------------------------------------- */

static real_t avar(real_t zzt, real_t zzl, real_t zzc,
                   private prop_type* prop, private propv_type* propv)
{
    /* formerly `static` -> locals (only called once per ITM call, lvar=5) */
    int kdv = 0;
    int ws = 0, w1 = 0;
    real_t dexa=(real_t)0.0, de=(real_t)0.0, vmd=(real_t)0.0, vs0=(real_t)0.0;
    real_t sgl=(real_t)0.0, sgtm=(real_t)0.0, sgtp=(real_t)0.0, sgtd=(real_t)0.0, tgtd=(real_t)0.0;
    real_t gm=(real_t)0.0, gp=(real_t)0.0;
    real_t cv1=(real_t)0.0, cv2=(real_t)0.0, yv1=(real_t)0.0, yv2=(real_t)0.0, yv3=(real_t)0.0;
    real_t csm1=(real_t)0.0, csm2=(real_t)0.0, ysm1=(real_t)0.0, ysm2=(real_t)0.0, ysm3=(real_t)0.0;
    real_t csp1=(real_t)0.0, csp2=(real_t)0.0, ysp1=(real_t)0.0, ysp2=(real_t)0.0, ysp3=(real_t)0.0;
    real_t csd1=(real_t)0.0, zd=(real_t)0.0;
    real_t cfm1=(real_t)0.0, cfm2=(real_t)0.0, cfm3=(real_t)0.0;
    real_t cfp1=(real_t)0.0, cfp2=(real_t)0.0, cfp3=(real_t)0.0;

    real_t bv1[7]={(real_t)-9.67,(real_t)-0.62,(real_t)1.26,(real_t)-9.21,(real_t)-0.62,(real_t)-0.39,(real_t)3.15};
    real_t bv2[7]={(real_t)12.7,(real_t)9.19,(real_t)15.5,(real_t)9.05,(real_t)9.19,(real_t)2.86,(real_t)857.9};
    real_t xv1[7]={(real_t)144.9e3,(real_t)228.9e3,(real_t)262.6e3,(real_t)84.1e3,(real_t)228.9e3,(real_t)141.7e3,(real_t)2222.e3};
    real_t xv2[7]={(real_t)190.3e3,(real_t)205.2e3,(real_t)185.2e3,(real_t)101.1e3,(real_t)205.2e3,(real_t)315.9e3,(real_t)164.8e3};
    real_t xv3[7]={(real_t)133.8e3,(real_t)143.6e3,(real_t)99.8e3,(real_t)98.6e3,(real_t)143.6e3,(real_t)167.4e3,(real_t)116.3e3};
    real_t bsm1[7]={(real_t)2.13,(real_t)2.66,(real_t)6.11,(real_t)1.98,(real_t)2.68,(real_t)6.86,(real_t)8.51};
    real_t bsm2[7]={(real_t)159.5,(real_t)7.67,(real_t)6.65,(real_t)13.11,(real_t)7.16,(real_t)10.38,(real_t)169.8};
    real_t xsm1[7]={(real_t)762.2e3,(real_t)100.4e3,(real_t)138.2e3,(real_t)139.1e3,(real_t)93.7e3,(real_t)187.8e3,(real_t)609.8e3};
    real_t xsm2[7]={(real_t)123.6e3,(real_t)172.5e3,(real_t)242.2e3,(real_t)132.7e3,(real_t)186.8e3,(real_t)169.6e3,(real_t)119.9e3};
    real_t xsm3[7]={(real_t)94.5e3,(real_t)136.4e3,(real_t)178.6e3,(real_t)193.5e3,(real_t)133.5e3,(real_t)108.9e3,(real_t)106.6e3};
    real_t bsp1[7]={(real_t)2.11,(real_t)6.87,(real_t)10.08,(real_t)3.68,(real_t)4.75,(real_t)8.58,(real_t)8.43};
    real_t bsp2[7]={(real_t)102.3,(real_t)15.53,(real_t)9.60,(real_t)159.3,(real_t)8.12,(real_t)13.97,(real_t)8.19};
    real_t xsp1[7]={(real_t)636.9e3,(real_t)138.7e3,(real_t)165.3e3,(real_t)464.4e3,(real_t)93.2e3,(real_t)216.0e3,(real_t)136.2e3};
    real_t xsp2[7]={(real_t)134.8e3,(real_t)143.7e3,(real_t)225.7e3,(real_t)93.1e3,(real_t)135.9e3,(real_t)152.0e3,(real_t)188.5e3};
    real_t xsp3[7]={(real_t)95.6e3,(real_t)98.6e3,(real_t)129.7e3,(real_t)94.2e3,(real_t)113.4e3,(real_t)122.7e3,(real_t)122.9e3};
    real_t bsd1[7]={(real_t)1.224,(real_t)0.801,(real_t)1.380,(real_t)1.000,(real_t)1.224,(real_t)1.518,(real_t)1.518};
    real_t bzd1[7]={(real_t)1.282,(real_t)2.161,(real_t)1.282,(real_t)20.,(real_t)1.282,(real_t)1.282,(real_t)1.282};
    real_t bfm1[7]={(real_t)1.0,(real_t)1.0,(real_t)1.0,(real_t)1.0,(real_t)0.92,(real_t)1.0,(real_t)1.0};
    real_t bfm2[7]={(real_t)0.0,(real_t)0.0,(real_t)0.0,(real_t)0.0,(real_t)0.25,(real_t)0.0,(real_t)0.0};
    real_t bfm3[7]={(real_t)0.0,(real_t)0.0,(real_t)0.0,(real_t)0.0,(real_t)1.77,(real_t)0.0,(real_t)0.0};
    real_t bfp1[7]={(real_t)1.0,(real_t)0.93,(real_t)1.0,(real_t)0.93,(real_t)0.93,(real_t)1.0,(real_t)1.0};
    real_t bfp2[7]={(real_t)0.0,(real_t)0.31,(real_t)0.0,(real_t)0.19,(real_t)0.31,(real_t)0.0,(real_t)0.0};
    real_t bfp3[7]={(real_t)0.0,(real_t)2.00,(real_t)0.0,(real_t)1.79,(real_t)2.00,(real_t)0.0,(real_t)0.0};

    real_t rt=(real_t)7.8, rl=(real_t)24.0, avarv, q, vs, zt, zl, zc;
    real_t sgt, yr, temp1, temp2;
    int temp_klim = propv->klim - 1;

    if (propv->lvar > 0) {
        switch (propv->lvar) {
            default:
            if (propv->klim <= 0 || propv->klim > 7) {
                propv->klim = 5;
                temp_klim = 4;
                prop->kwx = imax(prop->kwx, 2);
            }
            cv1=bv1[temp_klim]; cv2=bv2[temp_klim];
            yv1=xv1[temp_klim]; yv2=xv2[temp_klim]; yv3=xv3[temp_klim];
            csm1=bsm1[temp_klim]; csm2=bsm2[temp_klim];
            ysm1=xsm1[temp_klim]; ysm2=xsm2[temp_klim]; ysm3=xsm3[temp_klim];
            csp1=bsp1[temp_klim]; csp2=bsp2[temp_klim];
            ysp1=xsp1[temp_klim]; ysp2=xsp2[temp_klim]; ysp3=xsp3[temp_klim];
            csd1=bsd1[temp_klim]; zd=bzd1[temp_klim];
            cfm1=bfm1[temp_klim]; cfm2=bfm2[temp_klim]; cfm3=bfm3[temp_klim];
            cfp1=bfp1[temp_klim]; cfp2=bfp2[temp_klim]; cfp3=bfp3[temp_klim];
            case 4:
            kdv = propv->mdvar;
            ws = (kdv >= 20);
            if (ws) kdv -= 20;
            w1 = (kdv >= 10);
            if (w1) kdv -= 10;
            if (kdv < 0 || kdv > 3) { kdv = 0; prop->kwx = imax(prop->kwx, 2); }
            case 3:
            q = log((real_t)0.133 * prop->wn);
            gm = cfm1 + cfm2 / ((cfm3*q*cfm3*q) + (real_t)1.0);
            gp = cfp1 + cfp2 / ((cfp3*q*cfp3*q) + (real_t)1.0);
            case 2:
            dexa = sqrt((real_t)18e6 * prop->he[0]) + sqrt((real_t)18e6 * prop->he[1]) + pow(((real_t)575.7e12 / prop->wn), THIRD);
            case 1:
            if (prop->dist < dexa)
                de = (real_t)130e3 * prop->dist / dexa;
            else
                de = (real_t)130e3 + prop->dist - dexa;
        }
        vmd = curve(cv1, cv2, yv1, yv2, yv3, de);
        sgtm = curve(csm1, csm2, ysm1, ysm2, ysm3, de) * gm;
        sgtp = curve(csp1, csp2, ysp1, ysp2, ysp3, de) * gp;
        sgtd = sgtp * csd1;
        tgtd = (sgtp - sgtd) * zd;
        if (w1) {
            sgl = (real_t)0.0;
        } else {
            q = ((real_t)1.0 - (real_t)0.8 * exp(-prop->dist / (real_t)50e3)) * prop->dh * prop->wn;
            sgl = (real_t)10.0 * q / (q + (real_t)13.0);
        }
        if (ws) {
            vs0 = (real_t)0.0;
        } else {
            temp1 = ((real_t)5.0 + (real_t)3.0 * exp(-de / (real_t)100e3));
            vs0 = temp1 * temp1;
        }
        propv->lvar = 0;
    }

    zt = zzt; zl = zzl; zc = zzc;
    switch (kdv) {
        case 0: zt = zc; zl = zc; break;
        case 1: zl = zc; break;
        case 2: zl = zt; break;
    }
    if (fabs(zt) > (real_t)3.1 || fabs(zl) > (real_t)3.1 || fabs(zc) > (real_t)3.1)
        prop->kwx = imax(prop->kwx, 1);
    if (zt < (real_t)0.0)
        sgt = sgtm;
    else if (zt <= zd)
        sgt = sgtp;
    else
        sgt = sgtd + tgtd / zt;
    temp1 = sgt * zt;
    temp2 = sgl * zl;
    vs = vs0 + (temp1*temp1) / (rt + zc*zc) + (temp2*temp2) / (rl + zc*zc);
    if (kdv == 0) {
        yr = (real_t)0.0;
        propv->sgc = sqrt(sgt*sgt + sgl*sgl + vs);
    } else if (kdv == 1) {
        yr = sgt * zt;
        propv->sgc = sqrt(sgl*sgl + vs);
    } else if (kdv == 2) {
        yr = sqrt(sgt*sgt + sgl*sgl) * zt;
        propv->sgc = sqrt(vs);
    } else {
        yr = sgt * zt + sgl * zl;
        propv->sgc = sqrt(vs);
    }
    avarv = prop->aref - vmd - yr - propv->sgc * zc;
    if (avarv < (real_t)0.0)
        avarv = avarv * ((real_t)29.0 - avarv) / ((real_t)29.0 - (real_t)10.0 * avarv);
    return avarv;
}

/* ----------------------------------------------------------------------- */
/* point_to_point_ITM device function                                      */
/* ----------------------------------------------------------------------- */

static void point_to_point_ITM_cl(__global real_t* elev,
                                  real_t tht_m, real_t rht_m,
                                  real_t eps_dielect, real_t sgm_conductivity,
                                  real_t eno_ns_surfref, real_t frq_mhz,
                                  int radio_climate, int pol, real_t conf, real_t rel,
                                  private itm_state* st, __global real_t* scratch,
                                  private real_t* dbloss, private int* errnum)
{
    prop_type prop;
    propv_type propv;
    propa_type propa;
    real_t zsys = (real_t)0.0;
    real_t zc, zr, eno, enso, q, fs;
    int ja, jb, i, np;

    /* zero the structs */
    prop.aref=(real_t)0.0; prop.dist=(real_t)0.0;
    prop.hg[0]=(real_t)0.0; prop.hg[1]=(real_t)0.0;
    prop.rch[0]=(real_t)0.0; prop.rch[1]=(real_t)0.0;
    prop.wn=(real_t)0.0; prop.dh=(real_t)0.0; prop.dhd=(real_t)0.0;
    prop.ens=(real_t)0.0; prop.encc=(real_t)0.0; prop.cch=(real_t)0.0; prop.cd=(real_t)0.0;
    prop.gme=(real_t)0.0; prop.zgndreal=(real_t)0.0; prop.zgndimag=(real_t)0.0;
    prop.he[0]=(real_t)0.0; prop.he[1]=(real_t)0.0;
    prop.dl[0]=(real_t)0.0; prop.dl[1]=(real_t)0.0;
    prop.the[0]=(real_t)0.0; prop.the[1]=(real_t)0.0;
    prop.tiw=(real_t)0.0; prop.ght=(real_t)0.0; prop.ghr=(real_t)0.0; prop.rph=(real_t)0.0;
    prop.hht=(real_t)0.0; prop.hhr=(real_t)0.0; prop.tgh=(real_t)0.0; prop.tsgh=(real_t)0.0;
    prop.thera=(real_t)0.0; prop.thenr=(real_t)0.0;
    prop.rpl=0; prop.kwx=0; prop.mdp=0; prop.ptx=0; prop.los=0;
    propv.sgc=(real_t)0.0; propv.lvar=0; propv.mdvar=0; propv.klim=0;
    propa.dlsa=(real_t)0.0; propa.dx=(real_t)0.0; propa.ael=(real_t)0.0;
    propa.ak1=(real_t)0.0; propa.ak2=(real_t)0.0; propa.aed=(real_t)0.0;
    propa.emd=(real_t)0.0; propa.aes=(real_t)0.0; propa.ems=(real_t)0.0;
    propa.dls[0]=(real_t)0.0; propa.dls[1]=(real_t)0.0;
    propa.dla=(real_t)0.0; propa.tha=(real_t)0.0;
    /* itm_state already zero-initialized by the kernel */

    prop.hg[0] = tht_m;
    prop.hg[1] = rht_m;
    propv.klim = radio_climate;
    prop.kwx = 0;
    propv.lvar = 5;
    prop.mdp = -1;
    zc = qerfi(conf);
    zr = qerfi(rel);
    np = (int)elev[0];
    eno = eno_ns_surfref;
    enso = (real_t)0.0;
    q = enso;
    if (q <= (real_t)0.0) {
        ja = (int)((real_t)3.0 + (real_t)0.1 * elev[0]);
        jb = np - ja + 6;
        for (i = ja-1; i < jb; ++i)
            zsys += elev[i];
        zsys /= (jb - ja + 1);
        q = eno;
    }
    propv.mdvar = 12;
    qlrps(frq_mhz, zsys, q, pol, eps_dielect, sgm_conductivity, &prop);
    qlrpfl(elev, propv.klim, propv.mdvar, &prop, &propa, &propv, st, scratch);
    fs = (real_t)32.45 + (real_t)20.0 * log10(frq_mhz) + (real_t)20.0 * log10(prop.dist / (real_t)1000.0);
    *dbloss = avar(zr, (real_t)0.0, zc, &prop, &propv) + fs;
    *errnum = prop.kwx;
}

/* ----------------------------------------------------------------------- */
/* kernel entry (Phase 2): one work-item per pre-built profile             */
/* ----------------------------------------------------------------------- */

__kernel void itm_p2p_kernel(
    __global const real_t* profiles,   /* num_profiles x stride */
    int stride,
    __global real_t* scratch,          /* num_profiles x D1THX_SCRATCH_LEN */
    real_t tht_m, real_t rht_m,
    real_t eps_dielect, real_t sgm_conductivity, real_t eno_ns_surfref,
    real_t frq_mhz, int radio_climate, int pol,
    real_t conf, real_t rel,
    __global real_t* out_dbloss,       /* num_profiles */
    __global int* out_errnum)          /* num_profiles */
{
    int gid = get_global_id(0);
    int n = get_global_size(0);
    if (gid >= n) return;

    itm_state st;
    /* zero-initialise state */
    st.wd1=(real_t)0.0; st.xd1=(real_t)0.0; st.afo=(real_t)0.0; st.qk=(real_t)0.0; st.aht=(real_t)0.0; st.xht=(real_t)0.0;
    st.wls=(real_t)0.0;
    st.ad=(real_t)0.0; st.rr=(real_t)0.0; st.etq=(real_t)0.0; st.h0s=(real_t)0.0;
    st.wlos=0; st.wscat=0; st.dmin=(real_t)0.0; st.xae=(real_t)0.0;

    __global real_t* prof = (__global real_t*)(profiles + (size_t)gid * (size_t)stride);
    __global real_t* scr  = scratch + (size_t)gid * (size_t)D1THX_SCRATCH_LEN;

    real_t dbloss = (real_t)0.0;
    int errnum = 0;
    point_to_point_ITM_cl(prof, tht_m, rht_m, eps_dielect, sgm_conductivity,
                          eno_ns_surfref, frq_mhz, radio_climate, pol, conf, rel,
                          &st, scr, &dbloss, &errnum);
    out_dbloss[gid] = dbloss;
    out_errnum[gid] = errnum;
}

/* ----------------------------------------------------------------------- */
/* profile builder (matches ultra_main.cpp run_itm_path exactly)           */
/* ----------------------------------------------------------------------- */

static void build_profile_cl(__global const short* surface, int surf_w, int surf_h,
                             int tx_col, int tx_row, int rx_col, int rx_row,
                             real_t resolution,
                             __global real_t* elev, int stride)
{
    int dx = rx_col - tx_col;
    int dy = rx_row - tx_row;
    real_t dist_cells = sqrt((real_t)dx * (real_t)dx + (real_t)dy * (real_t)dy);
    real_t dist = rmax(resolution, dist_cells * resolution);
    int segments = imax(1, (int)ceil(dist_cells));
    elev[0] = (real_t)segments;
    elev[1] = dist / (real_t)segments;
    for (int i = 0; i <= segments; i++) {
        real_t t = (real_t)i / (real_t)segments;
        int col = imin(imax((int)round((real_t)tx_col + (real_t)dx * t), 0), surf_w - 1);
        int row = imin(imax((int)round((real_t)tx_row + (real_t)dy * t), 0), surf_h - 1);
        elev[i + 2] = (real_t)surface[(size_t)row * (size_t)surf_w + (size_t)col];
    }
    for (int i = segments + 3; i < segments + 16 && i < stride; i++)
        elev[i] = elev[segments + 2];
}

/* ----------------------------------------------------------------------- */
/* per-cell kernel: one work-item per output cell                           */
/* builds profile on-device, runs ITM, writes signal_i16 + mask_u8          */
/* ----------------------------------------------------------------------- */

__kernel void itm_cells_kernel(
    __global const short* surface,   /* int16 elevation grid */
    int surf_w, int surf_h,
    int tx_col, int tx_row,
    real_t resolution,
    real_t tht_m, real_t rht_m,
    real_t eps_dielect, real_t sgm_conductivity, real_t eno_ns_surfref,
    real_t frq_mhz, int radio_climate, int pol,
    real_t conf, real_t rel,
    real_t tx_power_w, real_t tx_gain_dbi, real_t rx_gain_dbi,
    real_t rx_sensitivity_dbm,
    int tile_x0, int tile_y0, int tile_w, int tile_h,
    int batch_start, int batch_count, int total_cells,
    __global real_t* prof_pool,     /* batch_count x stride_prof */
    int stride_prof,
    __global real_t* scratch_pool,  /* batch_count x D1THX_SCRATCH_LEN */
    __global short* out_signal,     /* total_cells, dbm_x10_i16 */
    __global uchar* out_mask)       /* total_cells, 0/1 */
{
    int gid = get_global_id(0);
    if (gid >= batch_count) return;
    int cell_idx = batch_start + gid;
    if (cell_idx >= total_cells) return;

    int local_col = cell_idx % tile_w;
    int local_row = cell_idx / tile_w;
    int col = tile_x0 + local_col;
    int row = tile_y0 + local_row;

    __global real_t* elev = prof_pool + (size_t)gid * (size_t)stride_prof;
    __global real_t* scr  = scratch_pool + (size_t)gid * (size_t)D1THX_SCRATCH_LEN;

    build_profile_cl(surface, surf_w, surf_h, tx_col, tx_row, col, row,
                     resolution, elev, stride_prof);

    /* Ensure profile writes are visible to the ITM computation. */
    mem_fence(CLK_GLOBAL_MEM_FENCE);

    itm_state st;
    st.wd1=(real_t)0.0; st.xd1=(real_t)0.0; st.afo=(real_t)0.0; st.qk=(real_t)0.0;
    st.aht=(real_t)0.0; st.xht=(real_t)0.0; st.wls=(real_t)0.0;
    st.ad=(real_t)0.0; st.rr=(real_t)0.0; st.etq=(real_t)0.0; st.h0s=(real_t)0.0;
    st.wlos=0; st.wscat=0; st.dmin=(real_t)0.0; st.xae=(real_t)0.0;

    real_t dbloss = (real_t)0.0;
    int errnum = 0;
    point_to_point_ITM_cl(elev, tht_m, rht_m, eps_dielect, sgm_conductivity,
                          eno_ns_surfref, frq_mhz, radio_climate, pol, conf, rel,
                          &st, scr, &dbloss, &errnum);

    real_t erp_w = tx_power_w * pow((real_t)10.0, tx_gain_dbi / (real_t)10.0);
    real_t rxp_w = erp_w / pow((real_t)10.0, (dbloss - (real_t)2.14) / (real_t)10.0);
    real_t dbm = (real_t)10.0 * log10(rxp_w * (real_t)1000.0) + rx_gain_dbi;

    int sig_i = (int)round(dbm * (real_t)10.0);
    sig_i = imin(imax(sig_i, -32768), 32767);
    out_signal[cell_idx] = (short)sig_i;
    out_mask[cell_idx] = (uchar)(dbm >= rx_sensitivity_dbm ? 1 : 0);
}

/* ---- debug: dump a single profile built on-device ---- */

__kernel void debug_profile_kernel(
    __global const short* surface, int surf_w, int surf_h,
    int tx_col, int tx_row, real_t resolution,
    int rx_col, int rx_row,
    __global real_t* out_profile, int stride)
{
    build_profile_cl(surface, surf_w, surf_h, tx_col, tx_row, rx_col, rx_row,
                     resolution, out_profile, stride);
}

/* ---- two-kernel pipeline: build profiles, then run ITM separately ---- */
/* Splitting into two kernels prevents the compiler from cross-optimizing
 * the profile writer and the ITM reader, which can change FP64 roundings. */

__kernel void build_profiles_kernel(
    __global const short* surface, int surf_w, int surf_h,
    int tx_col, int tx_row, real_t resolution,
    int tile_x0, int tile_y0, int tile_w, int tile_h,
    int batch_start, int batch_count, int total_cells,
    __global real_t* prof_pool, int stride_prof)
{
    int gid = get_global_id(0);
    if (gid >= batch_count) return;
    int cell_idx = batch_start + gid;
    if (cell_idx >= total_cells) return;
    int local_col = cell_idx % tile_w;
    int local_row = cell_idx / tile_w;
    int col = tile_x0 + local_col;
    int row = tile_y0 + local_row;
    __global real_t* elev = prof_pool + (size_t)gid * (size_t)stride_prof;
    build_profile_cl(surface, surf_w, surf_h, tx_col, tx_row, col, row,
                     resolution, elev, stride_prof);
}

__kernel void itm_p2p_signal_kernel(
    __global real_t* profiles, int stride,
    __global real_t* scratch,
    real_t tht_m, real_t rht_m,
    real_t eps_dielect, real_t sgm_conductivity, real_t eno_ns_surfref,
    real_t frq_mhz, int radio_climate, int pol,
    real_t conf, real_t rel,
    real_t tx_power_w, real_t tx_gain_dbi, real_t rx_gain_dbi,
    real_t rx_sensitivity_dbm,
    int batch_start, int batch_count, int total_cells,
    __global short* out_signal,
    __global uchar* out_mask)
{
    int gid = get_global_id(0);
    if (gid >= batch_count) return;
    int cell_idx = batch_start + gid;
    if (cell_idx >= total_cells) return;

    __global real_t* prof = profiles + (size_t)gid * (size_t)stride;
    __global real_t* scr  = scratch + (size_t)gid * (size_t)D1THX_SCRATCH_LEN;

    itm_state st;
    st.wd1=(real_t)0.0; st.xd1=(real_t)0.0; st.afo=(real_t)0.0; st.qk=(real_t)0.0;
    st.aht=(real_t)0.0; st.xht=(real_t)0.0; st.wls=(real_t)0.0;
    st.ad=(real_t)0.0; st.rr=(real_t)0.0; st.etq=(real_t)0.0; st.h0s=(real_t)0.0;
    st.wlos=0; st.wscat=0; st.dmin=(real_t)0.0; st.xae=(real_t)0.0;

    real_t dbloss = (real_t)0.0;
    int errnum = 0;
    point_to_point_ITM_cl(prof, tht_m, rht_m, eps_dielect, sgm_conductivity,
                          eno_ns_surfref, frq_mhz, radio_climate, pol, conf, rel,
                          &st, scr, &dbloss, &errnum);

    real_t erp_w = tx_power_w * pow((real_t)10.0, tx_gain_dbi / (real_t)10.0);
    real_t rxp_w = erp_w / pow((real_t)10.0, (dbloss - (real_t)2.14) / (real_t)10.0);
    real_t dbm = (real_t)10.0 * log10(rxp_w * (real_t)1000.0) + rx_gain_dbi;
    int sig_i = (int)round(dbm * (real_t)10.0);
    sig_i = imin(imax(sig_i, -32768), 32767);
    out_signal[cell_idx] = (short)sig_i;
    out_mask[cell_idx] = (uchar)(dbm >= rx_sensitivity_dbm ? 1 : 0);
}
