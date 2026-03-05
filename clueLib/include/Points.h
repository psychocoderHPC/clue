#pragma once

struct Points
{
    // For internally managed input
    std::vector<float> x;
    std::vector<float> y;
    std::vector<int> layer;
    std::vector<float> weight;
    std::vector<float> sigmaNoise;

    // For externally managed input
    float const* p_x;
    float const* p_y;
    int const* p_layer;
    float const* p_weight;
    float const* p_sigmaNoise;

    std::vector<float> rho;
    std::vector<float> delta;
    std::vector<unsigned int> nearestHigher;
    std::vector<int> clusterIndex;
    std::vector<std::vector<int>> followers;
    std::vector<uint8_t> isSeed;

    int n;

    void clear()
    {
        x.clear();
        y.clear();
        layer.clear();
        weight.clear();

        p_x = nullptr;
        p_y = nullptr;
        p_layer = nullptr;
        p_weight = nullptr;

        rho.clear();
        delta.clear();
        nearestHigher.clear();
        clusterIndex.clear();
        followers.clear();
        isSeed.clear();

        n = 0;
    }
};
