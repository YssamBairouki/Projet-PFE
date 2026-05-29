#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <set>
#include <map>
#include <deque>
#include <algorithm>
#include <cmath>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#include "DeviceFFT32Mode.h"

using namespace cuLenia;
using namespace std;

#define TITLE "Lenia"
#define SCREEN_X 1024
#define SCREEN_Y 1024
#define FPS_UPDATE 1.0

#define DEVICE_FFT_MODE 3
#define TESSELLATIUM 2

char mode_str[32] = "";
int MODE = DEVICE_FFT_MODE;
int GRID_SIZE = 1024;
int SCENE = TESSELLATIUM;
int RENDER_UPDATE = 25;
int DETECTION_UPDATE = 25;
bool batchFFT = true;

Mode* mode = nullptr;

// ===========================
// DETECTION PARAMS
// TESSELLATIUM + FFT32 ONLY
// ===========================

static const float DETECT_BINARY_THRESHOLD = 0.45f;
static const float DETECT_PEAK_REL = 0.40f;
static const int DETECT_MASK_MORPH_K = 3;
static const int DETECT_MIN_AREA = 100;


// ===========================
// SIMPLE POPULATION STATS
// ===========================

static const int HOMOGENEITY_GRID = 4;

// dynamique / capacité d'accueil
static const int HISTORY_MAX = 4096;
static const int EQUIL_WINDOW = 64;
static const double EQUIL_EPS = 0.02;

struct PopulationStats
{
    int entities = 0;
    double biomass = 0.0;
    double meanDensity = 0.0;
    double spatialCV = 0.0;

    double homogeneity = 0.0;

    double entityDelta = 0.0;
    double biomassDelta = 0.0;

    double carryingCountMean = 0.0;
    double carryingBiomassMean = 0.0;
    bool stableEquilibrium = false;
};

struct PopulationSnapshot
{
    double time = 0.0;
    PopulationStats stats;
};

static PopulationStats g_lastStats;
static std::deque<PopulationSnapshot> g_history;

// ===========================
// WINDOW TITLE / LOG
// ===========================

static void updateWindowTitle(GLFWwindow* window, double fps = -1.0)
{
    char t[768];

    if (fps >= 0.0)
    {
        int gsize = mode->lw.size * mode->lw.size;

        sprintf_s(
            t,
            "%s %s entities=%d biomass=%.1f CV=%.3f H=%.3f dN=%.2f dB=%.2f K_N=%.2f stable=%s %dx%d T=%d render=%d detect=%d batchFFT=%s %.2f FPS (TP=%.2f Mpx/s)",
            TITLE,
            mode_str,
            g_lastStats.entities,
            g_lastStats.biomass,
            g_lastStats.spatialCV,
            g_lastStats.homogeneity,
            g_lastStats.entityDelta,
            g_lastStats.biomassDelta,
            g_lastStats.carryingCountMean,
            g_lastStats.stableEquilibrium ? "yes" : "no",
            mode->lw.size,
            mode->lw.size,
            mode->lw.T,
            RENDER_UPDATE,
            DETECTION_UPDATE,
            batchFFT ? "yes" : "no",
            fps,
            fps * gsize / 1000000.f
        );
    }
    else
    {
        sprintf_s(
            t,
            "%s %s entities=%d biomass=%.1f CV=%.3f H=%.3f dN=%.2f dB=%.2f K_N=%.2f stable=%s %dx%d T=%d",
            TITLE,
            mode_str,
            g_lastStats.entities,
            g_lastStats.biomass,
            g_lastStats.spatialCV,
            g_lastStats.homogeneity,
            g_lastStats.entityDelta,
            g_lastStats.biomassDelta,
            g_lastStats.carryingCountMean,
            g_lastStats.stableEquilibrium ? "yes" : "no",
            mode->lw.size,
            mode->lw.size,
            mode->lw.T
        );
    }

    glfwSetWindowTitle(window, t);
}

static void logStats(GLFWwindow* window, const PopulationStats& stats)
{
    char msg[768];
    sprintf_s(
        msg,
        "[POP] entities=%d biomass=%.6f meanDensity=%.8f spatialCV=%.6f homogeneity=%.6f dN=%.6f dB=%.6f Kcount=%.6f Kbiomass=%.6f stable=%d\n",
        stats.entities,
        stats.biomass,
        stats.meanDensity,
        stats.spatialCV,
        stats.homogeneity,
        stats.entityDelta,
        stats.biomassDelta,
        stats.carryingCountMean,
        stats.carryingBiomassMean,
        stats.stableEquilibrium ? 1 : 0
    );

    OutputDebugStringA(msg);
    printf("%s", msg);

    g_lastStats = stats;
    updateWindowTitle(window);
}

// ===========================
// FFT32 HELPERS
// ===========================

// En DeviceFFT32Mode batché, tous les canaux sont stockés à la suite dans gridA[0]/gridB[0].
static float* currentGridFFT32Channel(Mode* mode, int ch)
{
    const int S = mode->lw.size;
    float* base = mode->lw.toGridB ? mode->lw.gridB[0] : mode->lw.gridA[0];
    return base + (size_t)ch * (size_t)S * (size_t)S;
}

static cv::Mat downloadChannelFFT32(Mode* mode, int channel)
{
    const int S = mode->lw.size;
    const size_t count = (size_t)S * (size_t)S;
    const size_t bytes = count * sizeof(float);

    vector<float> host(count);

    cudaError_t err = cudaMemcpy(
        host.data(),
        currentGridFFT32Channel(mode, channel),
        bytes,
        cudaMemcpyDeviceToHost
    );

    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed in detection: %s\n", cudaGetErrorString(err));
        return cv::Mat();
    }

    cv::Mat tmp(S, S, CV_32F, host.data());
    return tmp.clone();
}

static cv::Mat buildCombinedTesselatiumFieldFFT32(Mode* mode)
{
    cv::Mat ch0 = downloadChannelFFT32(mode, 0);
    cv::Mat ch1 = downloadChannelFFT32(mode, 1);
    cv::Mat ch2 = downloadChannelFFT32(mode, 2);

    if (ch0.empty() || ch1.empty() || ch2.empty())
        return cv::Mat();

    // combinaison maxRGB
    cv::Mat m01, combined;
    cv::max(ch0, ch1, m01);
    cv::max(m01, ch2, combined);

    return combined;
}

static cv::Mat tile3x3Toroidal(const cv::Mat& src)
{
    cv::Mat tiled;
    cv::repeat(src, 3, 3, tiled);
    return tiled;
}

// ===========================
// WATERSHED COUNTING
// TESSELLATIUM + FFT32 ONLY
// ===========================

static int countEntitiesWatershedFromCombined(
    const cv::Mat& density,
    float binaryThreshold,
    float peakRel,
    int morphK,
    int minArea)
{
    if (density.empty())
        return 0;

    const int S = density.rows;
    const int k = std::max(3, (morphK | 1));

    // Monde toroïdal : pavage 3x3 pour éviter de casser les individus aux bords
    cv::Mat tiled = tile3x3Toroidal(density);

    // Lissage léger
    cv::Mat smooth;
    cv::GaussianBlur(tiled, smooth, cv::Size(5, 5), 0.0, 0.0);

    // Masque binaire de présence
    cv::Mat mask;
    cv::threshold(smooth, mask, binaryThreshold, 1.0, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U, 255.0);

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(k, k)
    );

    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    if (cv::countNonZero(mask) == 0)
        return 0;

    // Distance transform -> graines
    cv::Mat dist;
    cv::distanceTransform(mask, dist, cv::DIST_L2, 5);

    double distMax = 0.0;
    cv::minMaxLoc(dist, nullptr, &distMax);
    if (distMax <= 0.0)
        return 0;

    cv::Mat sureFg;
    cv::threshold(dist, sureFg, peakRel * distMax, 1.0, cv::THRESH_BINARY);
    sureFg.convertTo(sureFg, CV_8U, 255.0);

    cv::morphologyEx(sureFg, sureFg, cv::MORPH_OPEN, kernel);

    if (cv::countNonZero(sureFg) == 0)
        return 0;

    cv::Mat markers;
    int nMarkers = cv::connectedComponents(sureFg, markers, 8, CV_32S);
    if (nMarkers <= 1)
        return 0;

    markers += 1;

    for (int y = 0; y < mask.rows; ++y)
    {
        const uchar* m = mask.ptr<uchar>(y);
        int* mk = markers.ptr<int>(y);
        for (int x = 0; x < mask.cols; ++x)
        {
            if (m[x] == 0)
                mk[x] = 0;
        }
    }

    cv::Mat gray8;
    cv::Mat norm01;
    cv::normalize(smooth, norm01, 0.0, 255.0, cv::NORM_MINMAX);
    norm01.convertTo(gray8, CV_8U);

    cv::Mat rgb;
    cv::cvtColor(gray8, rgb, cv::COLOR_GRAY2BGR);

    cv::watershed(rgb, markers);

    // On ne garde que les labels présents dans la tuile centrale
    const cv::Rect centralROI(S, S, S, S);
    cv::Mat centralLabels = markers(centralROI);

    std::set<int> labelsInCentral;
    for (int y = 0; y < centralLabels.rows; ++y)
    {
        const int* row = centralLabels.ptr<int>(y);
        for (int x = 0; x < centralLabels.cols; ++x)
        {
            const int lab = row[x];
            if (lab > 1)
                labelsInCentral.insert(lab);
        }
    }

    if (labelsInCentral.empty())
        return 0;

    std::map<int, int> totalArea;
    for (int lab : labelsInCentral)
        totalArea[lab] = 0;

    for (int y = 0; y < markers.rows; ++y)
    {
        const int* row = markers.ptr<int>(y);
        for (int x = 0; x < markers.cols; ++x)
        {
            const int lab = row[x];
            auto it = totalArea.find(lab);
            if (it != totalArea.end())
                it->second++;
        }
    }

    int count = 0;
    for (const auto& kv : totalArea)
    {
        if (kv.second >= minArea)
            count++;
    }

    return count;
}

// ===========================
// BASIC POPULATION STATS
// ===========================

static double computeBiomass(const cv::Mat& density)
{
    if (density.empty())
        return 0.0;

    return cv::sum(density)[0];
}

static double computeMeanDensity(const cv::Mat& density)
{
    if (density.empty())
        return 0.0;

    const double biomass = computeBiomass(density);
    return biomass / double(density.rows * density.cols);
}

static double computeSpatialCV(const cv::Mat& density, int gridN)
{
    if (density.empty() || gridN <= 0)
        return 0.0;

    const int H = density.rows;
    const int W = density.cols;

    vector<double> cellMasses;
    cellMasses.reserve(gridN * gridN);

    for (int gy = 0; gy < gridN; ++gy)
    {
        for (int gx = 0; gx < gridN; ++gx)
        {
            int x0 = (gx * W) / gridN;
            int x1 = ((gx + 1) * W) / gridN;
            int y0 = (gy * H) / gridN;
            int y1 = ((gy + 1) * H) / gridN;

            cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
            double m = cv::sum(density(roi))[0];
            cellMasses.push_back(m);
        }
    }

    if (cellMasses.empty())
        return 0.0;

    double mean = 0.0;
    for (double v : cellMasses) mean += v;
    mean /= double(cellMasses.size());

    if (mean <= 1e-12)
        return 0.0;

    double var = 0.0;
    for (double v : cellMasses)
    {
        double d = v - mean;
        var += d * d;
    }
    var /= double(cellMasses.size());

    double sd = std::sqrt(var);
    return sd / mean;
}

static double computeHomogeneityFromCV(double spatialCV)
{
    return 1.0 / (1.0 + std::max(0.0, spatialCV));
}

// ===========================
// HISTORY / DYNAMICS / K
// ===========================

static double meanHistoryCount(int window)
{
    if ((int)g_history.size() < window || window <= 0)
        return 0.0;

    double s = 0.0;
    for (int i = (int)g_history.size() - window; i < (int)g_history.size(); ++i)
        s += g_history[i].stats.entities;

    return s / double(window);
}

static double meanHistoryBiomass(int window)
{
    if ((int)g_history.size() < window || window <= 0)
        return 0.0;

    double s = 0.0;
    for (int i = (int)g_history.size() - window; i < (int)g_history.size(); ++i)
        s += g_history[i].stats.biomass;

    return s / double(window);
}

static bool detectStableEquilibrium(int window, double epsRel)
{
    if ((int)g_history.size() < 2 * window || window <= 0)
        return false;

    int n = (int)g_history.size();

    double aCount = 0.0, bCount = 0.0;
    double aBio = 0.0, bBio = 0.0;

    for (int i = n - 2 * window; i < n - window; ++i)
    {
        aCount += g_history[i].stats.entities;
        aBio += g_history[i].stats.biomass;
    }

    for (int i = n - window; i < n; ++i)
    {
        bCount += g_history[i].stats.entities;
        bBio += g_history[i].stats.biomass;
    }

    aCount /= window; bCount /= window;
    aBio /= window;   bBio /= window;

    double relCount = (fabs(aCount) > 1e-9) ? fabs(bCount - aCount) / fabs(aCount) : fabs(bCount - aCount);
    double relBio = (fabs(aBio) > 1e-9) ? fabs(bBio - aBio) / fabs(aBio) : fabs(bBio - aBio);

    return relCount < epsRel && relBio < epsRel;
}

static PopulationStats computePopulationStats(Mode* mode)
{
    PopulationStats stats;

    cv::Mat combined = buildCombinedTesselatiumFieldFFT32(mode);
    if (combined.empty())
        return stats;

    stats.entities = countEntitiesWatershedFromCombined(
        combined,
        DETECT_BINARY_THRESHOLD,
        DETECT_PEAK_REL,
        DETECT_MASK_MORPH_K,
        DETECT_MIN_AREA
    );

    stats.biomass = computeBiomass(combined);
    stats.meanDensity = computeMeanDensity(combined);
    stats.spatialCV = computeSpatialCV(combined, HOMOGENEITY_GRID);
    stats.homogeneity = computeHomogeneityFromCV(stats.spatialCV);

    if (!g_history.empty())
    {
        stats.entityDelta = stats.entities - g_history.back().stats.entities;
        stats.biomassDelta = stats.biomass - g_history.back().stats.biomass;
    }

    PopulationSnapshot snap;
    snap.time = glfwGetTime();
    snap.stats = stats;
    g_history.push_back(snap);

    while ((int)g_history.size() > HISTORY_MAX)
        g_history.pop_front();

    stats.carryingCountMean = meanHistoryCount(EQUIL_WINDOW);
    stats.carryingBiomassMean = meanHistoryBiomass(EQUIL_WINDOW);
    stats.stableEquilibrium = detectStableEquilibrium(EQUIL_WINDOW, EQUIL_EPS);

    return stats;
}

// ===========================
// SCENE
// TESSELLATIUM ONLY
// ===========================

void initTesselatium()
{
    float c[3][4] = {
        { 1.f, 0.f, 0.f, 0.5f },
        { 0.f, 1.f, 0.f, 0.5f },
        { 0.f, 0.f, 1.f, 0.5f }
    };

    float r[15][11] = {
        { 0,0,12,0.91f,1,4.f,1.f,0.f,0.272f,0.0595f,0.19f },
        { 0,0,12,0.62f,1,4.f,1.f,0.f,0.349f,0.1585f,0.66f },
        { 0,0,12,0.5f,2,4.f,1.f,1.f / 4.f,0.2f,0.0332f,0.39f },
        { 1,1,12,0.97f,2,4.f,0.f,1.f,0.114f,0.0528f,0.38f },
        { 1,1,12,0.72f,1,4.f,1.f,0.f,0.447f,0.0777f,0.74f },
        { 1,1,12,0.8f,2,4.f,5.f / 6.f,1.f,0.247f,0.0342f,0.92f },
        { 2,2,12,0.96f,1,4.f,1.f,0.f,0.21f,0.0617f,0.59f },
        { 2,2,12,0.56f,1,4.f,1.f,0.f,0.462f,0.1192f,0.37f },
        { 2,2,12,0.78f,1,4.f,1.f,0.f,0.446f,0.1793f,0.94f },
        { 0,1,12,0.79f,2,4.f,11.f / 12.f,1.f,0.327f,0.1408f,0.51f },
        { 0,2,12,0.5f,2,4.f,3.f / 4.f,1.f,0.476f,0.0995f,0.77f },
        { 1,0,12,0.72f,2,4.f,11.f / 12.f,1.f,0.379f,0.0697f,0.92f },
        { 1,2,12,0.68f,1,4.f,1.f,0.f,0.262f,0.0877f,0.71f },
        { 2,0,12,0.55f,2,4.f,1.f / 6.f,1.f,0.412f,0.1101f,0.59f },
        { 2,1,12,0.82f,1,4.f,1.f,0.f,0.201f,0.0786f,0.41f }
    };

    mode = new DeviceFFT32Mode(GRID_SIZE, 8, 3, 15, c, r);
}

void initLenia(GLFWwindow* window)
{
    MODE = DEVICE_FFT_MODE;
    SCENE = TESSELLATIUM;

    initTesselatium();
    sprintf_s(mode_str, "GPU-FFT32");

    g_lastStats = PopulationStats{};
    g_history.clear();

    updateWindowTitle(window);
}

void toggle(GLFWwindow* window)
{
    if (mode)
    {
        delete mode;
        mode = nullptr;
    }

    srand((unsigned)time(NULL));
    initLenia(window);
}

void checkFPS(GLFWwindow* window, double* lastTime, int* fpsCount)
{
    double currentTime = glfwGetTime();
    double delta = currentTime - *lastTime;

    if (delta >= FPS_UPDATE)
    {
        double fps = double(*fpsCount) / delta;
        updateWindowTitle(window, fps);
        *fpsCount = 0;
        *lastTime = currentTime;
    }
}

void checkRender(GLFWwindow* window, int* renderCount)
{
    int w, h;
    if (*renderCount == RENDER_UPDATE)
    {
        glfwGetFramebufferSize(window, &w, &h);
        mode->render(w, h);
        glfwSwapBuffers(window);
        *renderCount = 0;
    }
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, GLFW_TRUE);

        // On reste en FFT32 / Tesselatium uniquement
        if (key == '3') toggle(window);
        if (key == 'T') toggle(window);
        if (key == GLFW_KEY_ENTER) toggle(window);

        if (key == 'B')
            batchFFT = !batchFFT;

        if (key == 'R')
        {
            if (RENDER_UPDATE < 5) RENDER_UPDATE += 1;
            else RENDER_UPDATE += 5;
        }

        if (key == 'F')
        {
            if (RENDER_UPDATE > 1)
            {
                if (RENDER_UPDATE < 10) RENDER_UPDATE -= 1;
                else RENDER_UPDATE -= 5;
            }
        }

        if (key == 'Y')
        {
            if (DETECTION_UPDATE < 5) DETECTION_UPDATE += 1;
            else DETECTION_UPDATE += 5;
        }

        if (key == 'H')
        {
            if (DETECTION_UPDATE > 1)
            {
                if (DETECTION_UPDATE < 10) DETECTION_UPDATE -= 1;
                else DETECTION_UPDATE -= 5;
            }
        }

        if (key == GLFW_KEY_UP)
        {
            GRID_SIZE *= 2;
            toggle(window);
        }

        if (key == GLFW_KEY_DOWN)
        {
            if (GRID_SIZE > 64)
            {
                GRID_SIZE /= 2;
                toggle(window);
            }
        }

        if (key == GLFW_KEY_RIGHT)
        {
            if (mode->lw.T < 4) mode->lw.T += 1;
            else mode->lw.T *= 2;
        }

        if (key == GLFW_KEY_LEFT)
        {
            if (mode->lw.T >= 2)
            {
                if (mode->lw.T <= 4) mode->lw.T -= 1;
                else mode->lw.T /= 2;
            }
        }
    }
}

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW error: %s\n", description);
}

int main(void)
{
    srand((unsigned)time(NULL));

    if (!glfwInit())
        return -1;

    GLFWwindow* window = glfwCreateWindow(SCREEN_X, SCREEN_Y, TITLE, NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    GLint GlewInitResult = glewInit();
    if (GlewInitResult != GLEW_OK)
    {
        printf("ERROR: %s\n", glewGetErrorString(GlewInitResult));
    }

    glfwSetKeyCallback(window, key_callback);
    glfwSetErrorCallback(error_callback);

    toggle(window);

    double lastTime = glfwGetTime();
    int fpsCount = 0;
    int renderCount = 0;
    int detectionCount = 0;

    while (!glfwWindowShouldClose(window))
    {
        fpsCount++;
        renderCount++;
        detectionCount++;

        const bool doRender = (renderCount == RENDER_UPDATE);
        mode->compute(doRender, batchFFT);

        if (detectionCount >= DETECTION_UPDATE)
        {
            PopulationStats stats = computePopulationStats(mode);
            logStats(window, stats);
            detectionCount = 0;
        }

        checkFPS(window, &lastTime, &fpsCount);
        checkRender(window, &renderCount);
        glfwPollEvents();
    }

    if (mode)
    {
        delete mode;
        mode = nullptr;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}