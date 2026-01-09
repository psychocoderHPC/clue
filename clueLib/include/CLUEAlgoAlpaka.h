#pragma once
// clang-format off
#include <alpaka/alpaka.hpp>

#include "CLUEAlgo.h"
#include "TilesAlpaka.h"

// clang-format on

#include <cfloat>
#include <chrono>
#include <optional>

#define ORDER_TILE 0

#define DECLARE_TASKTYPE_AND_KERNEL(ACC, NAME, ...)                                                                   \
    struct Kernel##NAME                                                                                               \
    {                                                                                                                 \
    };                                                                                                                \
    ALPAKA_FN_ACC void operator()(ACC const& acc, Kernel##NAME dummy, ##__VA_ARGS__) const

// Maximum number of uniques seeds that could be handled. A higher number of
// potential seed will trigger an exception.
static int const maxNSeeds = 262144;

// Maximum number of followers that could be handled. A higher number of
// followers will trigger an exception.
static int const maxNFollowers = 128;

// Maximum size of the local stack used to assign clusters to seeds and
// followers. It should be at least as big as the maximum allowed number of
// followers. Adding more elements with respect to the reserved size will
// trigger an exception.
static int const localStackSizePerSeed = 128;

// The type T is used to pass the number of bins in each dimension and the
// allowed ranges spanned. Anchillary quantitied, like the inverse of the bin
// width should also be provided. Code will not compile if any such information
// is missing.
template<typename TExecutor, typename TComputeDevice, typename TQueue, typename THostDevice, typename T, int NLAYERS>
class CLUEAlgoAlpaka : public CLUEAlgo<T, NLAYERS>
{
public:
    static constexpr uint32_t dim = 1u;
    using Idx = uint32_t;

    template<typename TT>
    using BufAccT = ALPAKA_TYPEOF(
        alpaka::onHost::alloc<TT>(std::declval<TComputeDevice>(), std::declval<alpaka::Vec<Idx, dim>>()));

    template<typename TT>
    using ViewHostT
        = ALPAKA_TYPEOF(alpaka::onHost::alloc<TT>(std::declval<THostDevice>(), std::declval<alpaka::Vec<Idx, dim>>()));

    using LayerTilesAcc = TilesAlpaka<T>;
    using BufLayerTiles = BufAccT<LayerTilesAcc>;

    using BufVecArrSeeds = BufAccT<GPUAlpaka::VecArray<int, maxNSeeds>>;
    using BufVecArrFollowers = BufAccT<GPUAlpaka::VecArray<int, maxNFollowers>>;

    //
    // Bring base-class public variables into the scope of this template derived
    // class
    using CLUEAlgo<T, NLAYERS>::dc_;
    using CLUEAlgo<T, NLAYERS>::rhoc_;
    using CLUEAlgo<T, NLAYERS>::kappa_;
    using CLUEAlgo<T, NLAYERS>::outlierDeltaFactor_;
    using CLUEAlgo<T, NLAYERS>::verbose_;
    using CLUEAlgo<T, NLAYERS>::useAbsoluteSigma_;
    using CLUEAlgo<T, NLAYERS>::points_;

    struct PointsBuf
    {
        // Input Buffers
        std::optional<BufAccT<float>> x;
        std::optional<BufAccT<float>> y;
        std::optional<BufAccT<int>> layer;
        std::optional<BufAccT<float>> weight;
        std::optional<BufAccT<float>> sigmaNoise;

        // Output Buffers
        std::optional<BufAccT<float>> rho;
        std::optional<BufAccT<float>> delta;
        std::optional<BufAccT<unsigned int>> nearestHigher;
        std::optional<BufAccT<int>> clusterIndex;
        std::optional<BufAccT<uint8_t>> isSeed;
    };

    class DeviceRunner
    {
    public:
        struct DeviceRawPointers
        {
            // Pointers to Input buffers on device
            float* x;
            float* y;
            int* layer;
            float* weight;
            float* sigmaNoise;
            uint32_t* detid;

            // Pointers to output buffers on device
            float* rho;
            float* delta;
            unsigned int* nearestHigher;
            int* clusterIndex;
            uint8_t* isSeed;

            // LayerTiles and utility data structures
            TilesAlpaka<T>* hist_;
            GPUAlpaka::VecArray<int, maxNSeeds>* seeds_;
            GPUAlpaka::VecArray<int, maxNFollowers>* followers_;
        };

        // Kernel and KernelTask definitions
        DECLARE_TASKTYPE_AND_KERNEL(auto, ComputeHistogram, unsigned int const num_elements);
        DECLARE_TASKTYPE_AND_KERNEL(auto, SortHistogram);
        DECLARE_TASKTYPE_AND_KERNEL(auto, ComputeLocalDensity, float dc, unsigned int const num_elements);
        DECLARE_TASKTYPE_AND_KERNEL(
            auto,
            ComputeDistanceToHigherNoDetId,
            float outlierDeltaFactor,
            float dc,
            unsigned int const num_elements);
        DECLARE_TASKTYPE_AND_KERNEL(
            auto,
            ComputeDistanceToHigher,
            float outlierDeltaFactor,
            float dc,
            unsigned int const num_elements);
        DECLARE_TASKTYPE_AND_KERNEL(
            auto,
            FindClusters,
            float outlierDeltaFactor,
            float dc,
            float rhoc,
            unsigned int const num_elements);
        DECLARE_TASKTYPE_AND_KERNEL(
            auto,
            FindClustersKappa,
            float outlierDeltaFactor,
            float dc,
            float kappa,
            unsigned int const num_elements);
        DECLARE_TASKTYPE_AND_KERNEL(
            auto,
            AssignClusters,
            unsigned int* numberOfClustersScalar,
            bool writeOutNumClusters = true);
        DeviceRawPointers ptrs_;
    };

    CLUEAlgoAlpaka(
        TComputeDevice& computeDevice,
        TQueue& queue,
        THostDevice& hostDevice,
        float dc,
        float kappa,
        float outlierDeltaFactor,
        bool verbose,
        bool useAbsoluteSigma = false)
        : CLUEAlgo<T, NLAYERS>(dc, kappa, outlierDeltaFactor, verbose, useAbsoluteSigma)
        , device_(computeDevice)
        , queue_(queue)
        , host_(hostDevice)
    {
        init_device();
    }

    ~CLUEAlgoAlpaka()
    {
        free_device();
    }

    void makeClusters();
    void makeClustersCMSSW(
        unsigned int const points,
        float const* x,
        float const* y,
        int const* layer,
        float const* weight,
        float const* sigmaNoise,
        uint32_t const* detid,
        float* rho,
        float* delta,
        unsigned int* nearestHigher,
        int* clusterIndex,
        uint8_t* isSeed,
        unsigned int* numberOfClustersScalar);

    // Device runner to submit kernels
    DeviceRunner device_runner_;

private:
    TComputeDevice device_;
    // choose between Blocking and NonBlocking
    TQueue queue_;
    THostDevice host_;

    // Memory management variables
    PointsBuf device_bufs_;
    std::optional<BufLayerTiles> device_hist_;
    std::optional<BufVecArrSeeds> device_seeds_;
    std::optional<BufVecArrFollowers> device_followers_;

    void init_device()
    {
        Idx const reserve = 1'000'000;
        // If Dim is not 1, fail compilation. This is assumed to be a
        // mono-dimensional problem
        static_assert(dim == 1u);
        alpaka::Vec<Idx, dim> const extents(reserve);

        // INPUT VARIABLES
        // Allocate device memory
        device_bufs_.x = std::make_optional(alpaka::onHost::alloc<float>(device_, extents));
        device_bufs_.y = std::make_optional(alpaka::onHost::alloc<float>(device_, extents));
        device_bufs_.layer = std::make_optional(alpaka::onHost::alloc<int>(device_, extents));
        device_bufs_.weight = std::make_optional(alpaka::onHost::alloc<float>(device_, extents));
        device_bufs_.sigmaNoise = std::make_optional(alpaka::onHost::alloc<float>(device_, extents));

        // RESULT VARIABLES
        device_bufs_.rho = std::make_optional(alpaka::onHost::alloc<float>(device_, extents));
        device_bufs_.delta = std::make_optional(alpaka::onHost::alloc<float>(device_, extents));
        device_bufs_.nearestHigher = std::make_optional(alpaka::onHost::alloc<unsigned int>(device_, extents));
        device_bufs_.clusterIndex = std::make_optional(alpaka::onHost::alloc<int>(device_, extents));
        device_bufs_.isSeed = std::make_optional(alpaka::onHost::alloc<uint8_t>(device_, extents));

        // INTERNAL VARIABLES
        alpaka::Vec<Idx, dim> const layerTilesExtents(static_cast<Idx>(NLAYERS));
        device_hist_ = std::make_optional(alpaka::onHost::alloc<LayerTilesAcc>(device_, layerTilesExtents));

        alpaka::Vec<Idx, dim> const seedsExtents(1u);
        device_seeds_
            = std::make_optional(alpaka::onHost::alloc<GPUAlpaka::VecArray<int, maxNSeeds>>(device_, seedsExtents));

        device_followers_
            = std::make_optional(alpaka::onHost::alloc<GPUAlpaka::VecArray<int, maxNFollowers>>(device_, extents));

        // Update RAW device pointers, grouped in a struct for convenience
        device_runner_.ptrs_.x = alpaka::onHost::data(device_bufs_.x.value());
        device_runner_.ptrs_.y = alpaka::onHost::data(device_bufs_.y.value());
        device_runner_.ptrs_.layer = alpaka::onHost::data(device_bufs_.layer.value());
        device_runner_.ptrs_.weight = alpaka::onHost::data(device_bufs_.weight.value());
        if(useAbsoluteSigma_)
            device_runner_.ptrs_.sigmaNoise = alpaka::onHost::data(device_bufs_.sigmaNoise.value());

        // RESULT VARIABLES
        device_runner_.ptrs_.rho = alpaka::onHost::data(device_bufs_.rho.value());
        device_runner_.ptrs_.delta = alpaka::onHost::data(device_bufs_.delta.value());
        device_runner_.ptrs_.nearestHigher = alpaka::onHost::data(device_bufs_.nearestHigher.value());
        device_runner_.ptrs_.clusterIndex = alpaka::onHost::data(device_bufs_.clusterIndex.value());
        device_runner_.ptrs_.isSeed = alpaka::onHost::data(device_bufs_.isSeed.value());

        // UPDATE RAW POINTERS FOR INTERNATL DATA STRUCTURES
        device_runner_.ptrs_.hist_ = alpaka::onHost::data(device_hist_.value());
        device_runner_.ptrs_.seeds_ = alpaka::onHost::data(device_seeds_.value());
        device_runner_.ptrs_.followers_ = alpaka::onHost::data(device_followers_.value());
    }

    void free_device()
    {
        // Nothing really to be done here, since Alpaka memory buffers are
        // reference counted and will be automatically deleted when needed.
    }

    // Returns a view host using the memory allocated by the passed in
    // std::vector. The size of the view is inferred from the size of the vector.
    // This means the view will, possibly, become invalid if, in the meantime,
    // the vector re-allocated its underlying storage.
    template<typename TT>
    auto getViewHost(std::vector<TT>& t)
    {
        using type = typename std::vector<TT>::value_type;
        using ExtentType = alpaka::Vec<Idx, 1u>;
        ExtentType vectorSize(static_cast<Idx>(t.size()));

        auto deleter = [](type* ptr) {};
        auto pitches = ExtentType{sizeof(type)};
        /*
        alpaka::onHost::data()
        alpaka::makeMdSpan(t.data(),)
        auto data = std::make_shared<alpaka::onHost::internal::Data<ALPAKA_TYPEOF(host_), type, ExtentType,
        ExtentType>>( host_, t.data(), vectorSize, pitches, std::move(deleter));*/
        return alpaka::View(alpaka::api::host, t.data(), vectorSize, pitches);
    }

    template<typename TT>
    auto getViewHost(TT* t, int size)
    {
        using ExtentType = alpaka::Vec<Idx, 1u>;
        ExtentType vectorSize(static_cast<Idx>(size));

        auto deleter = [](TT* ptr) {};
        auto pitches = ExtentType{sizeof(TT)};
        return alpaka::View(alpaka::api::host, t, vectorSize, pitches);
    }

    void copy_todevice()
    {
        // input variables
        alpaka::Vec<Idx, 1u> const extentToTransfer(static_cast<Idx>(points_.n));
        alpaka::onHost::memcpy(queue_, device_bufs_.x.value(), getViewHost(points_.p_x, points_.n), extentToTransfer);
        alpaka::onHost::memcpy(queue_, device_bufs_.y.value(), getViewHost(points_.p_y, points_.n), extentToTransfer);
        alpaka::onHost::memcpy(
            queue_,
            device_bufs_.layer.value(),
            getViewHost(points_.p_layer, points_.n),
            extentToTransfer);
        alpaka::onHost::memcpy(
            queue_,
            device_bufs_.weight.value(),
            getViewHost(points_.p_weight, points_.n),
            extentToTransfer);
        if(useAbsoluteSigma_)
            alpaka::onHost::memcpy(
                queue_,
                device_bufs_.sigmaNoise.value(),
                getViewHost(points_.p_sigmaNoise, points_.n),
                extentToTransfer);
        alpaka::onHost::wait(queue_);
    }

    void clear_internal_buffers()
    {
        // result variables
        alpaka::Vec<Idx, 1u> extents(static_cast<Idx>(points_.n));
        alpaka::onHost::memset(queue_, device_bufs_.rho.value(), 0x0, extents);
        alpaka::onHost::memset(queue_, device_bufs_.delta.value(), 0x0, extents);
        alpaka::onHost::memset(queue_, device_bufs_.nearestHigher.value(), 0x0, extents);
        alpaka::onHost::memset(queue_, device_bufs_.clusterIndex.value(), 0x0, extents);
        alpaka::onHost::memset(queue_, device_bufs_.isSeed.value(), 0x0, extents);
        // algorithm internal variables
        // INTERNAL VARIABLES
        alpaka::Vec<Idx, 1u> const layerTilesExtents(static_cast<Idx>(NLAYERS));
        alpaka::onHost::memset(queue_, device_hist_.value(), 0x0, layerTilesExtents);

        alpaka::Vec<Idx, 1u> const seedsExtents(1u);
        alpaka::onHost::memset(queue_, device_seeds_.value(), 0x0, seedsExtents);

        alpaka::onHost::memset(queue_, device_followers_.value(), 0x0, extents);
        alpaka::onHost::wait(queue_);
    }

    void copy_tohost()
    {
        // result variables
        alpaka::Vec<Idx, 1u> extents(static_cast<Idx>(points_.n));

        auto clusterHV = getViewHost(points_.clusterIndex);
        alpaka::onHost::memcpy(queue_, clusterHV, device_bufs_.clusterIndex.value(), extents);
        if(verbose_)
        {
            // other variables, copy only when verbose_==True
            auto rhoHV = getViewHost(points_.rho);
            alpaka::onHost::memcpy(queue_, rhoHV, device_bufs_.rho.value(), extents);

            auto deltaHV = getViewHost(points_.delta);
            alpaka::onHost::memcpy(queue_, deltaHV, device_bufs_.delta.value(), extents);

            auto nearestHV = getViewHost(points_.nearestHigher);
            alpaka::onHost::memcpy(queue_, nearestHV, device_bufs_.nearestHigher.value(), extents);

            auto isSeedHV = getViewHost(points_.isSeed);
            alpaka::onHost::memcpy(queue_, isSeedHV, device_bufs_.isSeed.value(), extents);
        }
        alpaka::onHost::wait(queue_);
    }
};

template<typename TExecutor, typename TComputeDevice, typename TQueue, typename THostDevice, typename T, int NLAYERS>
ALPAKA_FN_ACC auto CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
operator()(
    auto const& acc,
    CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::KernelComputeHistogram
        dummy,
    unsigned int const numberOfPoints) const -> void
{
    for(auto [i] :
        alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{numberOfPoints}))
    {
        // push index of points into tiles
        ptrs_.hist_[ptrs_.layer[i]].fill(acc, ptrs_.x[i], ptrs_.y[i], i);
    }
}

template<typename TExecutor, typename TComputeDevice, typename TQueue, typename THostDevice, typename T, int NLAYERS>
ALPAKA_FN_ACC auto CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
operator()(
    auto const& acc,
    CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::KernelSortHistogram
        dummy) const -> void
{
    for(auto [i] :
        alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{NLAYERS * T::nTiles}))
    {
        int layer = i / T::nTiles;
        int bin = i - layer * T::nTiles;
        ptrs_.hist_[layer].sort_unsafe(acc, bin);
    }
}

template<typename TExecutor, typename TComputeDevice, typename TQueue, typename THostDevice, typename T, int NLAYERS>
ALPAKA_FN_ACC auto CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
operator()(
    auto const& acc,
    CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::KernelComputeLocalDensity
        dummy,
    float dc,
    unsigned int const numberOfPoints) const -> void
{
    for(auto [i] :
        alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{numberOfPoints}))
    {
        float rhoi{0.};
        int layeri = ptrs_.layer[i];
        float xi = ptrs_.x[i];
        float yi = ptrs_.y[i];

        // get search box
        int4 search_box = ptrs_.hist_[layeri].searchBox(xi - dc, xi + dc, yi - dc, yi + dc);

        // loop over bins in the search box
        for(int xBin = search_box.x; xBin < search_box.y + 1; ++xBin)
        {
            for(int yBin = search_box.z; yBin < search_box.w + 1; ++yBin)
            {
                // get the id of this bin
                int binId = ptrs_.hist_[layeri].getGlobalBinByBin(xBin, yBin);
                // get the size of this bin
                int binSize = ptrs_.hist_[layeri][binId].size();

                // interate inside this bin
                for(int binIter = 0; binIter < binSize; binIter++)
                {
                    unsigned int j = ptrs_.hist_[layeri][binId][binIter];
                    // query N_{dc_}(i)
                    float xj = ptrs_.x[j];
                    float yj = ptrs_.y[j];
                    float dist_ij = ((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj));
                    if(dist_ij < dc * dc)
                    {
                        // sum weights within N_{dc_}(i)
                        rhoi += (i == j ? 1.f : 0.5f) * ptrs_.weight[j];
                    }
                } // end of interate inside this bin
            }
        } // end of loop over bins in search box
        ptrs_.rho[i] = rhoi;
    }
}

template<typename TExecutor, typename TComputeDevice, typename TQueue, typename THostDevice, typename T, int NLAYERS>
ALPAKA_FN_ACC auto CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
operator()(
    auto const& acc,
    CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
        KernelComputeDistanceToHigherNoDetId dummy,
    float outlierDeltaFactor,
    float dc,
    unsigned int const numberOfPoints) const -> void
{
    float dm = outlierDeltaFactor * dc;

    for(auto [i] :
        alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{numberOfPoints}))
    {
        int layeri = ptrs_.layer[i];

        float deltai = std::numeric_limits<float>::max();
        unsigned int nearestHigheri = std::numeric_limits<unsigned int>::max();
        float xi = ptrs_.x[i];
        float yi = ptrs_.y[i];
        float rhoi = ptrs_.rho[i];
        float rho_max = 0.f;

        // get search box
        int4 search_box = ptrs_.hist_[layeri].searchBox(xi - dm, xi + dm, yi - dm, yi + dm);

        // loop over all bins in the search box
        for(int xBin = search_box.x; xBin < search_box.y + 1; ++xBin)
        {
            for(int yBin = search_box.z; yBin < search_box.w + 1; ++yBin)
            {
                // get the id of this bin
                int binId = ptrs_.hist_[layeri].getGlobalBinByBin(xBin, yBin);
                // get the size of this bin
                int binSize = ptrs_.hist_[layeri][binId].size();

                // iterate inside this bin
#if ORDER_TILE
                unsigned int old_j = 0;
#endif
                for(int binIter = 0; binIter < binSize; binIter++)
                {
                    unsigned int j = ptrs_.hist_[layeri][binId][binIter];
#if ORDER_TILE
                    assert(j >= old_j);
                    old_j = j;
#endif
                    // query N'_{dm}(i)
                    float xj = ptrs_.x[j];
                    float yj = ptrs_.y[j];
                    float dist_ij =
                        // std::sqrt((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj));
                        ((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj));
                    bool foundHigher = (ptrs_.rho[j] > rhoi);
                    // in the rare case where rho is the same, use indices
                    foundHigher = foundHigher || ((ptrs_.rho[j] == rhoi) && (j > i));

                    // combine the three conditions to avoid branching into different code segments to speedup the
                    // memory write a little bit
                    bool condition
                        = foundHigher && dist_ij < deltai
                          || dist_ij == deltai && ((ptrs_.rho[j] > rho_max) || (ptrs_.rho[j] == rho_max && j > i));
                    if(condition)
                    {
                        rho_max = ptrs_.rho[j];
                        deltai = dist_ij;
                        nearestHigheri = j;
                    }
                }
            }
        } // end of loop over bins in search box
        ptrs_.delta[i] = std::sqrt(deltai);
        ptrs_.nearestHigher[i] = nearestHigheri;
    }
}

template<typename TExecutor, typename TComputeDevice, typename TQueue, typename THostDevice, typename T, int NLAYERS>
ALPAKA_FN_ACC auto CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
operator()(
    auto const& acc,
    CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
        KernelComputeDistanceToHigher dummy,
    float outlierDeltaFactor,
    float dc,
    unsigned int const numberOfPoints) const -> void
{
    float dm = outlierDeltaFactor * dc;

    for(auto [i] :
        alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{numberOfPoints}))
    {
        int layeri = ptrs_.layer[i];

        float deltai = std::numeric_limits<float>::max();
        unsigned int nearestHigheri = std::numeric_limits<unsigned int>::max();
        float xi = ptrs_.x[i];
        float yi = ptrs_.y[i];
        float rhoi = ptrs_.rho[i];
        float rho_max = 0.f;

        // get search box
        int4 search_box = ptrs_.hist_[layeri].searchBox(xi - dm, xi + dm, yi - dm, yi + dm);

        // loop over all bins in the search box
        for(int xBin = search_box.x; xBin < search_box.y + 1; ++xBin)
        {
            for(int yBin = search_box.z; yBin < search_box.w + 1; ++yBin)
            {
                // get the id of this bin
                int binId = ptrs_.hist_[layeri].getGlobalBinByBin(xBin, yBin);
                // get the size of this bin
                int binSize = ptrs_.hist_[layeri][binId].size();

                // iterate inside this bin
#if ORDER_TILE
                unsigned int old_j = 0;
#endif
                for(int binIter = 0; binIter < binSize; binIter++)
                {
                    unsigned int j = ptrs_.hist_[layeri][binId][binIter];
#if ORDER_TILE
                    assert(j >= old_j);
                    old_j = j;
#endif
                    // query N'_{dm}(i)
                    float xj = ptrs_.x[j];
                    float yj = ptrs_.y[j];
                    float dist_ij =
                        // std::sqrt((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj));
                        ((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj));
                    bool foundHigher = (ptrs_.rho[j] > rhoi);
                    // in the rare case where rho is the same, use detid
                    foundHigher = foundHigher || ((ptrs_.rho[j] == rhoi) && (ptrs_.detid[j] > ptrs_.detid[i]));

                    // combine the three conditions to avoid branching into different code segments to speedup the
                    // memory write a little bit
                    bool condition = foundHigher && dist_ij < deltai
                                     || dist_ij == deltai
                                            && ((ptrs_.rho[j] > rho_max)
                                                || (ptrs_.rho[j] == rho_max && ptrs_.detid[j] > ptrs_.detid[i]));
                    if(condition)
                    {
                        rho_max = ptrs_.rho[j];
                        deltai = dist_ij;
                        nearestHigheri = j;
                    }
                }
            } // end of interate inside this bin
        } // end of loop over bins in search box
        ptrs_.delta[i] = std::sqrt(deltai);
        ptrs_.nearestHigher[i] = nearestHigheri;
    }
}

template<typename TExecutor, typename TComputeDevice, typename TQueue, typename THostDevice, typename T, int NLAYERS>
ALPAKA_FN_ACC auto CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
operator()(
    auto const& acc,
    KernelFindClustersKappa dummy,
    float outlierDeltaFactor,
    float dc,
    float kappa,
    unsigned int const numberOfPoints) const -> void
{
    /* This looks boring and is boring.
     * Normally the MdSPan object would be passed as argument but since the original code only stores pointers I
     * decided to create the MdSpan on the fly. The alignment is set to a safe value and could also be `sizeof(type) *
     * simdWidth` because alpaka is always aligning allocated memory to simdWidth. MdSpan is required for the
     * concurrent for each to generate SIMD code and/or at least code instruction level parallel code.
     */
    auto clusterIndexSpan = makeMdSpan(
        ptrs_.clusterIndex,
        alpaka::Vec{numberOfPoints},
        alpaka::Vec{sizeof(int)},
        alpaka::Alignment<sizeof(int)>{});

    auto deltaSpan = makeMdSpan(
        ptrs_.delta,
        alpaka::Vec{numberOfPoints},
        alpaka::Vec{sizeof(float)},
        alpaka::Alignment<sizeof(float)>{});

    auto rohSpan = makeMdSpan(
        ptrs_.rho,
        alpaka::Vec{numberOfPoints},
        alpaka::Vec{sizeof(float)},
        alpaka::Alignment<sizeof(float)>{});

    auto sigmaNoiseSpan = makeMdSpan(
        ptrs_.sigmaNoise,
        alpaka::Vec{numberOfPoints},
        alpaka::Vec{sizeof(float)},
        alpaka::Alignment<sizeof(float)>{});

    auto simdGrid = alpaka::onAcc::SimdAlgo{alpaka::onAcc::worker::threadsInGrid};
    simdGrid.concurrent(
        acc,
        [&](auto const&, auto&& simdClusterIdx, auto&& simdDelta, auto&& simdRoh, auto&& simdSigmaNoise) constexpr
        {
            // initialize clusterIndex
            using T_SimdType = ALPAKA_TYPEOF(simdClusterIdx.load());
            simdClusterIdx = T_SimdType::fill(-1);

            // determine seed or outlier
            alpaka::concepts::Simd auto deltai = simdDelta.load();
            alpaka::concepts::Simd auto rhoi = simdRoh.load();
            alpaka::concepts::Simd auto rhoc = simdSigmaNoise.load() * kappa;
            alpaka::concepts::Simd auto isSeed = (deltai > dc) && (rhoi >= rhoc);
            alpaka::concepts::Simd auto isOutlier = (deltai > outlierDeltaFactor * dc) && (rhoi < rhoc);

            auto idxOffset = simdDelta.getIdx().x();
            for(int sIdx = 0; sIdx < alpaka::getDim(isSeed); ++sIdx)
            {
                int i = idxOffset + sIdx;
                if(isSeed[sIdx])
                {
                    // set isSeed as 1
                    ptrs_.isSeed[i] = 1;
                    // head of device_seeds_
                    ptrs_.seeds_[0].push_back(acc, i);
                }
                else
                {
                    if(!isOutlier[sIdx])
                    {
                        assert(ptrs_.nearestHigher[i] < numberOfPoints);
                        // register as follower at its nearest higher
                        ptrs_.followers_[ptrs_.nearestHigher[i]].push_back(acc, i);
                    }
                }
            }
        },
        clusterIndexSpan,
        deltaSpan,
        rohSpan,
        sigmaNoiseSpan);
}

template<typename TExecutor, typename TComputeDevice, typename TQueue, typename THostDevice, typename T, int NLAYERS>
ALPAKA_FN_ACC auto CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
operator()(
    auto const& acc,
    KernelFindClusters dummy,
    float outlierDeltaFactor,
    float dc,
    float rhoc,
    unsigned int const numberOfPoints) const -> void
{
    // see the kernel above
    auto clusterIndexSpan = makeMdSpan(
        ptrs_.clusterIndex,
        alpaka::Vec{numberOfPoints},
        alpaka::Vec{sizeof(int)},
        alpaka::Alignment<sizeof(int)>{});

    auto deltaSpan = makeMdSpan(
        ptrs_.delta,
        alpaka::Vec{numberOfPoints},
        alpaka::Vec{sizeof(float)},
        alpaka::Alignment<sizeof(float)>{});

    auto rohSpan = makeMdSpan(
        ptrs_.rho,
        alpaka::Vec{numberOfPoints},
        alpaka::Vec{sizeof(float)},
        alpaka::Alignment<sizeof(float)>{});

    auto simdGrid = alpaka::onAcc::SimdAlgo{alpaka::onAcc::worker::threadsInGrid};
    simdGrid.concurrent(
        acc,
        [&](auto const&, auto&& simdClusterIdx, auto&& simdDelta, auto&& simdRoh) constexpr
        {
            // initialize clusterIndex
            using T_SimdType = ALPAKA_TYPEOF(simdClusterIdx.load());
            simdClusterIdx = T_SimdType::fill(-1);

            // determine seed or outlier
            alpaka::concepts::Simd auto deltai = simdDelta.load();
            alpaka::concepts::Simd auto rhoi = simdRoh.load();
            alpaka::concepts::Simd auto isSeed = (deltai > dc) && (rhoi >= rhoc);
            alpaka::concepts::Simd auto isOutlier = (deltai > outlierDeltaFactor * dc) && (rhoi < rhoc);

            auto idxOffset = simdDelta.getIdx().x();
            for(int sIdx = 0; sIdx < alpaka::getDim(isSeed); ++sIdx)
            {
                int i = idxOffset + sIdx;
                if(isSeed[sIdx])
                {
                    // set isSeed as 1
                    ptrs_.isSeed[i] = 1;
                    // head of device_seeds_
                    ptrs_.seeds_[0].push_back(acc, i);
                }
                else
                {
                    if(!isOutlier[sIdx])
                    {
                        assert(ptrs_.nearestHigher[i] < numberOfPoints);
                        // register as follower at its nearest higher
                        ptrs_.followers_[ptrs_.nearestHigher[i]].push_back(acc, i);
                    }
                }
            }
        },
        clusterIndexSpan,
        deltaSpan,
        rohSpan);
}

template<typename TExecutor, typename TComputeDevice, typename TQueue, typename THostDevice, typename T, int NLAYERS>
ALPAKA_FN_ACC auto CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
operator()(
    auto const& acc,
    CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::KernelAssignClusters
        dummy,
    unsigned int* numberOfClustersScalar,
    bool writeOutNumClusters) const -> void
{
    if(writeOutNumClusters)
    {
        for(auto [idxCls] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{1u}))
        {
            *numberOfClustersScalar = ptrs_.seeds_[0].size();
        }
    }

#if ALPAKA_LANG_CUDA && __CUDA_ARCH__
    /* alpaka has currently not implemented warps. Never the less it is possible to define any kind of group therefore
     * we hard code a warp size of 32 if CLUE_USE_CUDA_WARP is defined and use later warps in a grid to iterate over
     * the seed noods and all therads within a warp to iterate of the followers of the seed node.
     * Currently the start parameters are not adjusted for warps and we use always the same frame extent.
     * Native support for warps is coming soon! */
#    define CLUE_USE_CUDA_WARP 1
#endif

    // iterate with thread blocks over the seed particles
    for(auto [idxCls] : alpaka::onAcc::makeIdxMap(
            acc,
#if CLUE_USE_CUDA_WARP
            alpaka::onAcc::WorkerGroup{
                (acc[alpaka::layer::block].idx() * acc[alpaka::layer::thread].count()
                 + acc[alpaka::layer::thread].idx())
                    / 32u,
                (acc[alpaka::layer::thread].count() * acc[alpaka::layer::block].count()) / 32u},
#else
            alpaka::onAcc::worker::blocksInGrid,
#endif

            alpaka::IdxRange{(unsigned int) ptrs_.seeds_[0].size()}))
    {
        int localStack[localStackSizePerSeed] = {-1};
        int localStackSize = 0;

        // assign cluster to seed[idxCls]
        int idxThisSeed = ptrs_.seeds_[0][idxCls];
        for(auto [idxCls] :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInBlock, alpaka::IdxRange{1u}))
        {
            ptrs_.clusterIndex[idxThisSeed] = idxCls;
        }
        //  the first level of the hierarchy will be processed by all threads in a block
        for(auto [stackIdx] : alpaka::onAcc::makeIdxMap(
                acc,
#if CLUE_USE_CUDA_WARP
                alpaka::onAcc::WorkerGroup{
                    alpaka::Vec<uint32_t, 1u>{acc[alpaka::layer::thread].idx() % 32u},
                    alpaka::Vec{32u}},
#else
                alpaka::onAcc::worker::threadsInBlock,
#endif
                alpaka::IdxRange{alpaka::Vec<uint32_t, 1u>{ptrs_.followers_[idxThisSeed].size()}}))
        {
            int rootIdx = ptrs_.followers_[idxThisSeed][stackIdx];
            ptrs_.clusterIndex[rootIdx] = idxCls;
            int currentRootIdx = rootIdx;

            if(ptrs_.followers_[currentRootIdx].size() > 0)
            {
                // process all elements in localStack
                do
                {
                    // during the first visit of this loop we do not need from the stack
                    if(currentRootIdx != rootIdx)
                    {
                        // get last element of localStack
                        assert(localStackSize - 1 < localStackSizePerSeed);
                        currentRootIdx = localStack[localStackSize - 1];

                        // pop_back last element of localStack
                        localStack[localStackSize - 1] = -1;
                        localStackSize--;
                    }

                    // loop over followers of last element of localStack
                    for(int j : ptrs_.followers_[currentRootIdx])
                    {
                        // pass id to follower
                        ptrs_.clusterIndex[j] = idxCls;
                        // push only to the stack of we have followers to avoid useless memory operations and reduce
                        // the number of elements on te stack
                        if(ptrs_.followers_[currentRootIdx].size() > 0)
                        {
                            // push_back follower to localStack
                            assert(localStackSize < localStackSizePerSeed);
                            localStack[localStackSize] = j;
                            localStackSize++;
                        }
                    }
                    // reset the current root to load next index from the stack
                    currentRootIdx = -1;
                } while(localStackSize > 0);
            }
        }
    }
}

inline auto fString(std::string s)
{
    s.resize(std::max(s.size(), size_t{30}), ' ');
    return s;
}

template<typename TExecutor, typename TComputeDevice, typename TQueue, typename THostDevice, typename T, int NLAYERS>
void CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::makeClusters()
{
    copy_todevice();
    clear_internal_buffers();

    // Dimension the grid for submission
    alpaka::Vec<Idx, dim> const threadsPerBlock(1024u);
    alpaka::Vec<Idx, dim> const blocksPerGrid(alpaka::divExZero(static_cast<Idx>(points_.n), threadsPerBlock[0]));

    auto const manualWorkDiv = alpaka::onHost::FrameSpec{blocksPerGrid, threadsPerBlock};

    // Create the kernel execution tasks.
    typename CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
        KernelComputeHistogram taskComputeHistogram;
    auto const kernelComputeHistogram
        = alpaka::KernelBundle(device_runner_, taskComputeHistogram, static_cast<int>(points_.n));

    typename CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
        KernelComputeLocalDensity taskComputeLocalDensity;
    auto const kernelComputeLocalDensity
        = alpaka::KernelBundle(device_runner_, taskComputeLocalDensity, dc_, static_cast<int>(points_.n));

    typename CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
        KernelComputeDistanceToHigherNoDetId taskComputeDistanceToHigherNoDetId;
    auto const kernelComputeDistanceToHigherNoDetId = (alpaka::KernelBundle(
        device_runner_,
        taskComputeDistanceToHigherNoDetId,
        outlierDeltaFactor_,
        dc_,
        static_cast<int>(points_.n)));

    // use int as data type since we handle indecision and float value in the kernels
    uint32_t elementsPerFrameItem = alpaka::getNumElemPerThread<int>(queue_);
    alpaka::Vec<Idx, dim> const blocksPerGridSimd(
        alpaka::divExZero(static_cast<Idx>(points_.n), (threadsPerBlock[0] * elementsPerFrameItem)));
    auto const manualWorkDivSimd = alpaka::onHost::FrameSpec{blocksPerGridSimd, threadsPerBlock};

    typename CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
        KernelFindClusters taskFindClusters;
    auto const kernelFindClusters = (alpaka::KernelBundle(
        device_runner_,
        taskFindClusters,
        outlierDeltaFactor_,
        dc_,
        rhoc_,
        static_cast<int>(points_.n)));

    typename CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
        KernelFindClustersKappa taskFindClustersKappa;
    auto const kernelFindClustersKappa = (alpaka::KernelBundle(
        device_runner_,
        taskFindClustersKappa,
        outlierDeltaFactor_,
        dc_,
        kappa_,
        static_cast<int>(points_.n)));

    // Dimension the grid for submission
    alpaka::Vec<Idx, dim> const threadsPerBlockX(64u);

    // This value is too large and should be substituted with something reasonable
    alpaka::Vec<Idx, dim> const blocksPerGridX(static_cast<Idx>(points_.n));

    auto const manualWorkDivX = alpaka::onHost::FrameSpec{blocksPerGridX, threadsPerBlockX};

    typename CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
        KernelAssignClusters taskAssignClusters;
    auto const kernelAssignClusters = (alpaka::KernelBundle(device_runner_, taskAssignClusters, nullptr, false));

    // Enqueue the kernel execution task
    auto start = std::chrono::high_resolution_clock::now();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed;

    alpaka::onHost::wait(queue_);
    start = std::chrono::high_resolution_clock::now();
    queue_.enqueue(TExecutor{}, manualWorkDiv, kernelComputeHistogram);
    alpaka::onHost::wait(queue_); // wait in case we are using an asynchronous queue to
    // time actual kernel runtime
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << fString("--- computeHistogram:") << elapsed.count() * 1000 << "ms\n";

    start = std::chrono::high_resolution_clock::now();
    queue_.enqueue(TExecutor{}, manualWorkDiv, kernelComputeLocalDensity);
    alpaka::onHost::wait(queue_); // wait in case we are using an asynchronous queue to
    // time actual kernel runtime
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << fString("--- computeLocalDensity:") << elapsed.count() * 1000 << "ms\n";

    start = std::chrono::high_resolution_clock::now();
    queue_.enqueue(TExecutor{}, manualWorkDiv, kernelComputeDistanceToHigherNoDetId);
    alpaka::onHost::wait(queue_); // wait in case we are using an asynchronous queue to
    // time actual kernel runtime
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << fString("--- computeDistanceToHigher:") << elapsed.count() * 1000 << "ms\n";

    start = std::chrono::high_resolution_clock::now();
    if(useAbsoluteSigma_)
    {
        queue_.enqueue(TExecutor{}, manualWorkDivSimd, kernelFindClustersKappa);
    }
    else
    {
        queue_.enqueue(TExecutor{}, manualWorkDivSimd, kernelFindClusters);
    }
    alpaka::onHost::wait(queue_); // wait in case we are using an asynchronous queue to
    // time actual kernel runtime
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << fString("--- findClusters:") << elapsed.count() * 1000 << "ms\n";

    start = std::chrono::high_resolution_clock::now();
    queue_.enqueue(TExecutor{}, manualWorkDivX, kernelAssignClusters);
    alpaka::onHost::wait(queue_); // wait in case we are using an asynchronous queue to
    // time actual kernel runtime
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << fString("--- assignClusters:") << elapsed.count() * 1000 << "ms\n";

    copy_tohost();
}

template<typename TExecutor, typename TComputeDevice, typename TQueue, typename THostDevice, typename T, int NLAYERS>
void CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::makeClustersCMSSW(
    unsigned int const points,
    float const* x,
    float const* y,
    int const* layer,
    float const* weight,
    float const* sigmaNoise,
    uint32_t const* detid,
    float* rho,
    float* delta,
    unsigned int* nearestHigher,
    int* clusterIndex,
    uint8_t* isSeed,
    unsigned int* numberOfClustersScalar)
{
    // std::cout << "makeClustersCMSSW received " << points << " RecHits" <<
    // std::endl;

    // INTERNAL TILES VARIABLES
    Idx const reserve = 1'000'000;
    // If Dim is not 1, fail compilation. This is assumed to be a
    // mono-dimensional problem
    static_assert(dim == 1u);
    alpaka::Vec<Idx, dim> const extents(reserve);

    alpaka::Vec<Idx, dim> const layerTilesExtents(static_cast<Idx>(NLAYERS));

    /** @todo use queue allocateor */
    device_hist_ = std::make_optional(alpaka::onHost::alloc<LayerTilesAcc>(device_, layerTilesExtents));
    alpaka::Vec<Idx, dim> const seedsExtents(1u);
    device_seeds_
        = std::make_optional(alpaka::onHost::alloc<GPUAlpaka::VecArray<int, maxNSeeds>>(device_, seedsExtents));
    device_followers_
        = std::make_optional(alpaka::onHost::alloc<GPUAlpaka::VecArray<int, maxNFollowers>>(device_, extents));
    // INTERNAL VARIABLES RESETTING
    alpaka::onHost::memset(queue_, device_hist_.value(), 0x0, layerTilesExtents);
    alpaka::onHost::memset(queue_, device_seeds_.value(), 0x0, seedsExtents);
    alpaka::onHost::memset(queue_, device_followers_.value(), 0x0, extents);

    // Set Device Raw Pointers using values from outsice and also internal buffers
    device_runner_.ptrs_.x = const_cast<float*>(x);
    device_runner_.ptrs_.y = const_cast<float*>(y);
    device_runner_.ptrs_.layer = const_cast<int*>(layer);
    device_runner_.ptrs_.weight = const_cast<float*>(weight);
    device_runner_.ptrs_.sigmaNoise = const_cast<float*>(sigmaNoise);
    device_runner_.ptrs_.detid = const_cast<uint32_t*>(detid);

    // RESULT VARIABLES
    device_runner_.ptrs_.rho = rho;
    device_runner_.ptrs_.delta = delta;
    device_runner_.ptrs_.nearestHigher = nearestHigher;
    device_runner_.ptrs_.clusterIndex = clusterIndex;
    device_runner_.ptrs_.isSeed = isSeed;

    // UPDATE RAW POINTERS FOR INTERNATL DATA STRUCTURES
    device_runner_.ptrs_.hist_ = alpaka::onHost::data(device_hist_.value());
    device_runner_.ptrs_.seeds_ = alpaka::onHost::data(device_seeds_.value());
    device_runner_.ptrs_.followers_ = alpaka::onHost::data(device_followers_.value());

    // Dimension the grid for submission
    Idx threads_per_block = 256u;

    alpaka::Vec<Idx, dim> const threadsPerBlock(threads_per_block);

    alpaka::Vec<Idx, dim> const blocksPerGrid(alpaka::divExZero(static_cast<Idx>(points), threadsPerBlock[0]));

    auto const manualWorkDiv = alpaka::onHost::FrameSpec{blocksPerGrid, threadsPerBlock};

    // Create the kernel execution tasks.
    typename CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
        KernelComputeHistogram taskComputeHistogram;
    auto const kernelComputeHistogram = alpaka::KernelBundle(device_runner_, taskComputeHistogram, points);

#if ORDER_TILE
    // printf("Sorting tile for all layers.\n");
    alpaka::Vec<Dim, Idx> const threadsPerBlockTile(512u);
    alpaka::Vec<Dim, Idx> const blocksPerGridTile(std::ceil((NLAYERS * T::nTiles) / (float) threadsPerBlockTile[0]));
    auto const manualWorkDivTile = WorkDiv{blocksPerGridTile, threadsPerBlockTile, elementsPerThread};

    typename CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
        KernelSortHistogram taskSortHistogram;
    auto const kernelSortHistogram
        = alpaka::createTaskKernel<TAcc>(manualWorkDivTile, device_runner_, taskSortHistogram);
#endif

    typename CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
        KernelComputeLocalDensity taskComputeLocalDensity;
    auto const kernelComputeLocalDensity
        = (alpaka::KernelBundle(device_runner_, taskComputeLocalDensity, dc_, static_cast<int>(points)));

    typename CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
        KernelComputeDistanceToHigher taskComputeDistanceToHigher;
    auto const kernelComputeDistanceToHigher = (alpaka::KernelBundle(
        device_runner_,
        taskComputeDistanceToHigher,
        outlierDeltaFactor_,
        dc_,
        static_cast<int>(points)));

    typename CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
        KernelFindClustersKappa taskFindClustersKappa;
    auto const kernelFindClustersKappa = (alpaka::KernelBundle(
        device_runner_,
        taskFindClustersKappa,
        outlierDeltaFactor_,
        dc_,
        kappa_,
        static_cast<int>(points)));

    typename CLUEAlgoAlpaka<TExecutor, TComputeDevice, TQueue, THostDevice, T, NLAYERS>::DeviceRunner::
        KernelAssignClusters taskAssignClusters;
    auto const kernelAssignClusters
        = (alpaka::KernelBundle(device_runner_, taskAssignClusters, numberOfClustersScalar));

    // Enqueue the kernel execution task

    queue_.enqueue(TExecutor{}, manualWorkDiv, kernelComputeHistogram);
#if ORDER_TILE
    alpaka::onHost::enqueue(queue_, TExecutor{}, manualWorkDiv, kernelSortHistogram);
#endif
    queue_.enqueue(TExecutor{}, manualWorkDiv, kernelComputeLocalDensity);
    queue_.enqueue(TExecutor{}, manualWorkDiv, kernelComputeDistanceToHigher);
    queue_.enqueue(TExecutor{}, manualWorkDiv, kernelFindClustersKappa);
    queue_.enqueue(TExecutor{}, manualWorkDiv, kernelAssignClusters);
}
