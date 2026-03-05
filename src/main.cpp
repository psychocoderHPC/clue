#include "CLUEAlgo.h"

#include <stdlib.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <regex>
#include <string>
#if defined(USE_ALPAKA)
#    include "CLUEAlgoAlpaka.h"

#    include <alpaka/onHost/example/executors.hpp>
#    include <alpaka/onHost/executeForEach.hpp>
#else
#    include "CLUEAlgoGPU.h"
#endif

#define NLAYERS 100

using namespace std;

void exclude_stats_outliers(std::vector<float>& v)
{
    if(v.size() == 1)
        return;
    float mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    float sum_sq_diff = std::accumulate(
        v.begin(),
        v.end(),
        0.0,
        [mean](float acc, float x) { return acc + (x - mean) * (x - mean); });
    float stddev = std::sqrt(sum_sq_diff / (v.size() - 1));
    std::cout << "Sigma cut outliers: " << stddev << std::endl;
    float z_score_threshold = 3.0;
    v.erase(
        std::remove_if(
            v.begin(),
            v.end(),
            [mean, stddev, z_score_threshold](float x)
            {
                float z_score = std::abs(x - mean) / stddev;
                return z_score > z_score_threshold;
            }),
        v.end());
}

pair<float, float> stats(std::vector<float> const& v)
{
    float m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    float sum = std::accumulate(v.begin(), v.end(), 0.0, [m](float acc, float x) { return acc + (x - m) * (x - m); });
    auto den = v.size() > 1 ? (v.size() - 1) : v.size();
    return {m, std::sqrt(sum / den)};
}

void printTimingReport(std::vector<float>& vals, int repeats, std::string const label = "SUMMARY ")
{
    int precision = 2;
    float mean = 0.f;
    float sigma = 0.f;
    exclude_stats_outliers(vals);
    tie(mean, sigma) = stats(vals);
    std::cout << label << " 1 outliers(" << repeats << "/" << vals.size() << ") " << std::fixed
              << std::setprecision(precision) << mean << " +/- " << sigma << " [ms]" << std::endl;
    exclude_stats_outliers(vals);
    tie(mean, sigma) = stats(vals);
    std::cout << label << " 2 outliers(" << repeats << "/" << vals.size() << ") " << std::fixed
              << std::setprecision(precision) << mean << " +/- " << sigma << " [ms]" << std::endl;
}

void readDataFromFile(
    std::string const& inputFileName,
    std::vector<float>& x,
    std::vector<float>& y,
    std::vector<int>& layer,
    std::vector<float>& weight)
{
    // make dummy layers
    for(int l = 0; l < NLAYERS; l++)
    {
        // open csv file
        std::ifstream iFile(inputFileName);
        std::string value = "";
        // Iterate through each line and split the content using delimeter
        while(getline(iFile, value, ','))
        {
            x.push_back(std::stof(value));
            getline(iFile, value, ',');
            y.push_back(std::stof(value));
            getline(iFile, value, ',');
            layer.push_back(std::stoi(value) + l);
            getline(iFile, value);
            weight.push_back(std::stof(value));
        }
        iFile.close();
    }
}

std::string create_outputfileName(
    std::string const& inputFileName,
    float const dc,
    float const rhoc,
    float const outlierDeltaFactor)
{
    //  C++20
    //  auto suffix = std::format("_{:.2f}_{:.2f}_{:.2f}.csv", dc, rhoc,
    //  outlierDeltaFactor);
    char suffix[100];
    snprintf(suffix, 100, "_dc_%.2f_rho_%.2f_outl_%.2f.csv", dc, rhoc, outlierDeltaFactor);

    std::string tmpFileName;
    std::regex regexp("input");
    std::regex_replace(back_inserter(tmpFileName), inputFileName.begin(), inputFileName.end(), regexp, "output");

    std::string outputFileName;
    std::regex regexp2(".csv");
    std::regex_replace(back_inserter(outputFileName), tmpFileName.begin(), tmpFileName.end(), regexp2, suffix);

    return outputFileName;
}

void mainRun(
    std::string const& inputFileName,
    std::string const& outputFileName,
    float const dc,
    float const rhoc,
    float const outlierDeltaFactor,
    bool const use_accelerator,
    auto alpakaCfg,
    int const repeats,
    bool const verbose)
{
    //////////////////////////////
    // read toy data from csv file
    //////////////////////////////
    std::cout << "Start to load input points" << std::endl;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<int> layer;
    std::vector<float> weight;

    readDataFromFile(inputFileName, x, y, layer, weight);
    std::cout << "Finished loading input points" << std::endl;
    // Vector to perform some bread and butter analysis on the timing
    vector<float> vals;

    //////////////////////////////
    // run CLUE algorithm
    //////////////////////////////
    std::cout << "Start to run CLUE algorithm" << std::endl;
    if(use_accelerator)
    {
#if !defined(USE_ALPAKA)
        std::cout << "Native CUDA Backend selected" << std::endl;
        CLUEAlgoGPU<TilesConstants, NLAYERS> clueAlgo(dc, rhoc, outlierDeltaFactor, verbose);
        vals.clear();
        for(unsigned r = 0; r < repeats; r++)
        {
            if(!clueAlgo.setPoints(x.size(), &x[0], &y[0], &layer[0], &weight[0]))
                exit(EXIT_FAILURE);
            // measure excution time of makeClusters
            auto start = std::chrono::high_resolution_clock::now();
            clueAlgo.makeClusters();
            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;
            std::cout << "Iteration " << r;
            std::cout << " | Elapsed time: " << elapsed.count() * 1000 << " ms\n";
            // Skip first event
            if(r != 0 or repeats == 1)
            {
                vals.push_back(elapsed.count() * 1000);
            }
        }

        printTimingReport(vals, repeats, "SUMMARY WorkDivByPoints:");

        // output result to outputFileName. -1 means all points.
        clueAlgo.verboseResults(outputFileName, -1);

        std::cout << "Native CUDA Backend selected WorkDivByTile" << std::endl;
        CLUEAlgoGPU<TilesConstants, NLAYERS, WorkDivByTile> clueAlgoByTile(dc, rhoc, outlierDeltaFactor, verbose);
        vals.clear();
        for(unsigned r = 0; r < repeats; r++)
        {
            if(!clueAlgoByTile.setPoints(x.size(), &x[0], &y[0], &layer[0], &weight[0]))
                exit(EXIT_FAILURE);
            // measure excution time of makeClusters
            auto start = std::chrono::high_resolution_clock::now();
            clueAlgoByTile.makeClusters();
            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;
            std::cout << "Iteration " << r;
            std::cout << " | Elapsed time: " << elapsed.count() * 1000 << " ms\n";
            // Skip first event
            if(r != 0 or repeats == 1)
            {
                vals.push_back(elapsed.count() * 1000);
            }
        }

        printTimingReport(vals, repeats, "SUMMARY WorkDivByTile:");

        // output result to outputFileName. -1 means all points.
        clueAlgoByTile.verboseResults(outputFileName, -1);
#elif defined(USE_ALPAKA)
        std::cout << "ALPAKA 'Backend' selected" << std::endl;
        using namespace alpaka;
        using namespace alpaka::onHost;
        // Define the index domain

        auto deviceSpec = alpakaCfg[object::deviceSpec];
        auto exec = alpakaCfg[object::exec];

        std::cout << deviceSpec.getApi().getName() << std::endl;
        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        onHost::Device computeDevice = devSelector.makeDevice(0);
        std::cout << "Using alpaka accelerator: " << onHost::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << std::endl;
        Queue queue = computeDevice.makeQueue();

        Device cpuDevice = makeHostDevice();

        CLUEAlgoAlpaka<
            ALPAKA_TYPEOF(exec),
            ALPAKA_TYPEOF(computeDevice),
            ALPAKA_TYPEOF(queue),
            ALPAKA_TYPEOF(cpuDevice),
            TilesConstants,
            NLAYERS>
            clueAlgo(computeDevice, queue, cpuDevice, dc, rhoc, outlierDeltaFactor, verbose);
        vals.clear();
        for(unsigned r = 0; r < repeats; r++)
        {
            if(!clueAlgo.setPoints(x.size(), &x[0], &y[0], &layer[0], &weight[0]))
                exit(EXIT_FAILURE);
            // measure excution time of makeClusters
            auto start = std::chrono::high_resolution_clock::now();
            clueAlgo.makeClusters();
            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;
            std::cout << "Iteration " << r;
            std::cout << " | Elapsed time: " << elapsed.count() * 1000 << " ms\n";
            // Skip first event
            if(r != 0 or repeats == 1)
            {
                vals.push_back(elapsed.count() * 1000);
            }
        }

        printTimingReport(vals, repeats, "SUMMARY Alpaka Backend:");

        // output result to outputFileName. -1 means all points.
        clueAlgo.verboseResults(outputFileName, -1);
#endif
    }
    else
    {
        std::cout << "Native CPU(serial) Backend selected" << std::endl;
        CLUEAlgo<TilesConstants, NLAYERS> clueAlgo(dc, rhoc, outlierDeltaFactor, verbose);
        vals.clear();
        for(unsigned r = 0; r < repeats; r++)
        {
            if(!clueAlgo.setPoints(x.size(), &x[0], &y[0], &layer[0], &weight[0]))
                exit(EXIT_FAILURE);
            // measure excution time of makeClusters
            auto start = std::chrono::high_resolution_clock::now();
            clueAlgo.makeClusters();
            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;
            std::cout << "Iteration " << r;
            std::cout << " | Elapsed time: " << elapsed.count() * 1000 << " ms\n";
            // Skip first event
            if(r != 0 or repeats == 1)
            {
                vals.push_back(elapsed.count() * 1000);
            }
        }

        printTimingReport(vals, repeats, "SUMMARY Native CPU:");
        // output result to outputFileName. -1 means all points.
        if(verbose)
            clueAlgo.verboseResults(outputFileName, -1);
    }

    std::cout << "Finished running CLUE algorithm" << std::endl;
} // end of testRun()

int main(int argc, char* argv[])
{
    //////////////////////////////
    // MARK -- set algorithm parameters
    //////////////////////////////

    extern char* optarg;

    bool use_accelerator = false;
    bool verbose = false;
    float dc = 20.f, rhoc = 80.f, outlierDeltaFactor = 2.f;
    int repeats = 10;
    int opt;
    bool has_outFileName = false;
    std::string inputFileName;
    std::string outputFileName;
    std::string alpakaExecutor;
    bool list_alpaka_executors = false;

    while((opt = getopt(argc, argv, "i:d:r:o:O:e:u:Uv")) != -1)
    {
        switch(opt)
        {
        case 'i': /* input filename */
            inputFileName = string(optarg);
            break;
        case 'O': /* output filename */
            std::cout << " reached that" << std::endl;
            has_outFileName = true;
            outputFileName = string(optarg);
            break;
        case 'd': /* delta_c */
            dc = stof(string(optarg));
            break;
        case 'r': /* critical density */
            rhoc = stof(string(optarg));
            break;
        case 'o': /* outlier factor */
            outlierDeltaFactor = stof(string(optarg));
            break;
        case 'e': /* number of repeated session(s) a the selected input file */
            repeats = stoi(string(optarg));
            break;
        case 'u': /* Use accelerator */
            use_accelerator = true;
            alpakaExecutor = string(optarg);
            break;
        case 'U': /* Use accelerator */
            list_alpaka_executors = true;
            break;
        case 'v': /* Verbose output */
            verbose = true;
            break;
        default:
            std::cout << "bin/main -i [fileName] -d [dc] -r [rhoc] -o "
                         "[outlierDeltaFactor] -e [repeats] -u [executor] -v"
                      << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    //////////////////////////////
    // MARK -- set input and output files
    //////////////////////////////
    std::cout << "Input file: " << inputFileName << std::endl;
    if(has_outFileName)
    {
        outputFileName = create_outputfileName(outputFileName, dc, rhoc, outlierDeltaFactor);
    }
    else
    {
        outputFileName = create_outputfileName(inputFileName, dc, rhoc, outlierDeltaFactor);
    }
    std::cout << "Output file: " << outputFileName << std::endl;

    //////////////////////////////
    // MARK -- test run
    //////////////////////////////
    if(use_accelerator)
    {
#if defined(USE_ALPAKA)

        if(list_alpaka_executors)
        {
            std::cout << "alpaka executors" << std::endl;
            alpaka::onHost::executeForEach(
                [&](auto const& cfg)
                {
                    std::cout << "  " << alpaka::onHost::getStaticName(cfg[alpaka::object::exec]) << std::endl;
                    return 0;
                },
                alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));
            return 0;
        }

        return alpaka::onHost::executeForEach(
            [&](auto const& cfg)
            {
                if(alpakaExecutor == alpaka::onHost::getStaticName(cfg[alpaka::object::exec]))
                    mainRun(
                        inputFileName,
                        outputFileName,
                        dc,
                        rhoc,
                        outlierDeltaFactor,
                        use_accelerator,
                        cfg,
                        repeats,
                        verbose);

                return 0;
            },
            alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));
#else
        mainRun(
            inputFileName,
            outputFileName,
            dc,
            rhoc,
            outlierDeltaFactor,
            use_accelerator,
            // dummy, not used if alpaka is disabled
            std::make_tuple(1, 1),
            repeats,
            verbose);
#endif
    }
    else
    {
        mainRun(
            inputFileName,
            outputFileName,
            dc,
            rhoc,
            outlierDeltaFactor,
            use_accelerator,
        // dummy, not used if alpaka is disabled
#if defined(USE_ALPAKA)

            // select the first valid accelerator, -u is not set therefor we need only a valid configuration
            std::get<0>(
                alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors)),
#else
            std::make_tuple(1, 1),
#endif
            repeats,
            verbose);
    }
    return 0;
}
