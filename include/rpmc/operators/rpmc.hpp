
// Extended Representative Particle Monte Carlo method, implemented with the bucketing scheme.


#ifndef INCLUDED_RPMC_OPERATORS_RPMC_HPP_
#define INCLUDED_RPMC_OPERATORS_RPMC_HPP_


#include <span>
#include <cmath>
#include <array>
#include <tuple>
#include <bitset>
#include <vector>
#include <ranges>
#include <limits>
#include <random>       // for uniform_real_distribution<>
#include <variant>
#include <utility>      // for move()
#include <cstddef>      // for size_t
#include <cstdint>      // for int32_t, int64_t
#include <concepts>
#include <optional>
#include <iostream>
#include <algorithm>    // for min_element(), lower_bound()
#include <type_traits>  // for type_identity<>, integral_constant<>, conditional_t<>

#include <gsl-lite/gsl-lite.hpp>  // for dim, index, narrow_cast<>(), gsl_Expects(), gsl_Assert()

#include <fmt/core.h>  // for format()

#include <makeshift/array.hpp>        // for array_fill<>(), array_transform()
#include <makeshift/tuple.hpp>        // for tie_members(), template_for(), template_any_of()
#include <makeshift/ranges.hpp>
#include <makeshift/variant.hpp>      // for expand(), visit(), variant_transform()
#include <makeshift/type_traits.hpp>  // for nth_type<>

#include <intervals/utility.hpp>   // for as_regular<>
#include <intervals/interval.hpp>

#include <rpmc/tools/soa.hpp>            // for gatherFromSoA()
#include <rpmc/tools/bucketing.hpp>      // for BucketingScheme<>
#include <rpmc/tools/math.hpp>           // for discreteInverseTransform()
#include <rpmc/tools/utility.hpp>        // for square_checked_failfast()
#include <rpmc/tools/particles.hpp>      // for PParticleData

#include <rpmc/operators/operator.hpp>     // for InspectionData, NoOpDiscreteOperator
#include <rpmc/operators/rpmc/common.hpp>  // for IsSelfInteraction

#include <rpmc/detail/rpmc.hpp>


namespace rpmc {

namespace gsl = gsl_lite;


struct RPMCOperatorParams
{
    double removalBucketUpdateDelay = 1.;
    double rejectionBucketUpdateDelay = 1.;
};

struct RPMCOperatorArgs : RPMCOperatorParams
{
    explicit RPMCOperatorArgs(RPMCOperatorParams const& params)
        : RPMCOperatorParams(params)
    {
        gsl_Expects(removalBucketUpdateDelay >= 0.);
        gsl_Expects(rejectionBucketUpdateDelay >= 0.);
    }
};

/*template <typename T>
concept CRPMCClassifier = requires(T const& cx, gsl::index i)
{
        // Returns `BucketLabelT` or `std::optional<BucketLabelT>`.
    { cx.classify(i) };

        // Yields `true` if the `std::optional<>` returned by `cx.classify(i)` holds a value, or `false` otherwise.
    { cx.isActive(i) } -> std::same_as<bool>;
};*/

template <typename RandomNumberGeneratorT, typename ClassifierT, typename LocatorT, typename... InteractionModelsT>
class RPMCOperator : public NoOpDiscreteOperator
{
    static_assert(sizeof...(InteractionModelsT) > 0);

    using LocationIndex = std::int32_t;
    using Index = std::int32_t;
    using Dim = std::int32_t;

    template <gsl::index I, template <typename> class TT = std::type_identity_t>
    using ParticleStateForModel = typename detail::Rebind<decltype(detail::gatherFromSoA(
        std::declval<typename makeshift::nth_type_t<I, InteractionModelsT...> const>().getState(),
        gsl::index(0))), TT>::type;
    template <gsl::index I, template <typename> class TT = std::type_identity_t>
    using ParticlePropertiesForModel = typename detail::Rebind<decltype(getParticleProperties(
        std::declval<typename makeshift::nth_type_t<I, InteractionModelsT...> const>().getArgs(),
        std::declval<ParticleStateForModel<I, TT> const&>())), TT>::type;
    template <template <typename> class TT = std::type_identity_t>
    using ModelParticleState = typename detail::ParticlePropertiesTuple_<std::index_sequence_for<InteractionModelsT...>, ParticleStateForModel, TT>::type;
    template <template <typename> class TT = std::type_identity_t>
    using ModelParticleProperties = typename detail::ParticlePropertiesTuple_<std::index_sequence_for<InteractionModelsT...>, ParticlePropertiesForModel, TT>::type;
    template <template <typename> class TT = std::type_identity_t>
    using ParticleProperties = makeshift::unique_sequence_t<ModelParticleProperties<TT>>;
    static constexpr gsl::dim numParticleProperties = std::tuple_size_v<ParticleProperties<>>;
    static constexpr std::array modelToPropertiesIndexMap = detail::firstTupleElementIndices<ModelParticleProperties<>, ParticleProperties<>>();
    static constexpr std::array propertiesToModelIndexMap = detail::firstTupleElementIndices<ParticleProperties<>, ModelParticleProperties<>>();
    static constexpr auto propertiesToModelIndexMapC = MAKESHIFT_CONSTVAL(propertiesToModelIndexMap);
    template <template <typename> class TT = std::type_identity_t>
    using ParticleState = detail::GatherSequence<ModelParticleState<TT>, std::remove_const_t<decltype(propertiesToModelIndexMapC)>>;

    using BucketLabel = std::remove_cvref_t<decltype(detail::getValue(std::declval<ClassifierT>().classify(gsl::index{ })))>;

    static constexpr bool haveLocator = !std::is_same_v<LocatorT, DefaultLocator>;
    static_assert(std::is_same_v<LocatorT, DefaultLocator> || ((std::is_same_v<typename InteractionModelsT::Locator, DefaultLocator> || std::is_same_v<typename InteractionModelsT::Locator, LocatorT>) && ...));

    static constexpr gsl::dim numLocalInteractionModels = haveLocator
        ? (int(!std::is_same_v<typename InteractionModelsT::Locator, DefaultLocator>) + ... + 0)
        : 0;
    static constexpr bool haveLocality = numLocalInteractionModels > 0;
    using Location = decltype(std::declval<LocatorT>().location(gsl::index(0)));
    using SubBucketLabel = std::array<LocationIndex, haveLocality ? 1 : 0>;
    static constexpr std::array modelsHaveLocality = { (haveLocality && !std::is_same_v<typename InteractionModelsT::Locator, DefaultLocator>)... };
    static constexpr std::array modelInteractionDistanceIndices = detail::modelInteractionDistanceIndices<InteractionModelsT...>();

    static constexpr double inf = std::numeric_limits<double>::infinity();

    using FastDs = float;  // actually holding an integer
    using InteractionRates = std::array<double, sizeof...(InteractionModelsT)>;
    using InteractionTimes = std::array<double, sizeof...(InteractionModelsT)>;
    //using InteractionRadii = std::array<double, numLocalInteractionModels>;
    using InteractionDistances = std::array<FastDs, numLocalInteractionModels>;  
    using NumsInReach = std::array<Dim, numLocalInteractionModels>;
    using NSqInReach = std::array<std::int64_t, numLocalInteractionModels>;
    using UpdateBits = std::bitset<sizeof...(InteractionModelsT)>;

    struct NonlocalBucketData
    {
        Dim n;
        Dim updatesBeforeRecomputation1 = 0;
        Dim updatesBeforeRecomputation2 = 0;
        intervals::as_regular<ParticleProperties<intervals::set_of_t>> properties;
        InteractionRates cumulativeInteractionRates;
        UpdateBits needRecomputeInteractionRates = { };
        bool initialized : 1 = false;
        bool touched : 1 = false;
        bool haveProperDs : 1 = false;  // logically belongs to `LocalBucketData`
    };
    struct LocalBucketData : NonlocalBucketData
    {
        FastDs ds = { };
    };
    using BucketData = std::conditional_t<haveLocality, LocalBucketData, NonlocalBucketData>;

    struct NonlocalBucketBucketData
    {
        InteractionRates interactionRates;
    };
    struct LocalBucketBucketData : NonlocalBucketBucketData
    {
        //InteractionRadii interactionRadii;
        InteractionDistances ds12s;
        NSqInReach n1n2;

            // TODO: This is only used temporarily in `updateInteractionRatesForBucket()` and could thus be stored in `BucketData`,
            // at the cost of making `updateInteractionRatesForBucket()` even less reëntrant.
        UpdateBits needRecount = { };
    };
    using BucketBucketData = std::conditional_t<haveLocality, LocalBucketBucketData, NonlocalBucketBucketData>;
    template <gsl::index IModel>
    static inline FastDs&
    ds12Of(BucketBucketData& bucketBucketData)
    {
        static_assert(modelsHaveLocality[IModel]);
        constexpr gsl::index iDistance = modelInteractionDistanceIndices[IModel];
        return bucketBucketData.ds12s[iDistance];
    }
    template <gsl::index IModel>
    static inline std::int64_t&
    n1n2Of(BucketBucketData& bucketBucketData)
    {
        static_assert(modelsHaveLocality[IModel]);
        constexpr gsl::index iDistance = modelInteractionDistanceIndices[IModel];
        return bucketBucketData.n1n2[iDistance];
    }
    template <gsl::index IModel>
    static inline UpdateBits&
    needRecountFlagsOf(BucketBucketData& bucketBucketData)
    {
        static_assert(modelsHaveLocality[IModel]);
        return bucketBucketData.needRecount;
    }

    struct LocalSubBucketData
    {
        Dim n;
    };
    using SubBucketData = std::conditional_t<haveLocality, LocalSubBucketData, void>;

    using LBucketingScheme = BucketingScheme<BucketLabel, BucketData, BucketBucketData, SubBucketLabel, SubBucketData>;
    using Bucket = typename LBucketingScheme::Bucket;
    using ConstBucket = typename LBucketingScheme::ConstBucket;
    using SubBucket = typename LBucketingScheme::SubBucket;
    using ConstSubBucket = typename LBucketingScheme::ConstSubBucket;
    using SubBucketUpdate = typename LBucketingScheme::SubBucketUpdate;

private:
        // Model
    RPMCOperatorArgs args_;
    ClassifierT classifier_;
    LocatorT locator_;
    std::tuple<InteractionModelsT...> interactionModels_;

        // Bucket and particle data
    PParticleData particleData_;
    LBucketingScheme bucketingScheme_;
    std::vector<ParticleProperties<>> particleProperties_;
    InteractionRates totalBucketInteractionRates_ = { };

        // Simulation state
    RandomNumberGeneratorT randomNumberGenerator_;
    InteractionTimes nextInteractionTimes_ = makeshift::array_fill<sizeof...(InteractionModelsT)>(-inf);
    gsl::index nextInteractionTimeIndex_ = -1;
    double time_ = 0;

        // Temporary state
    std::vector<gsl::index> updatedParticleIndexList_;
    std::vector<gsl::index> internalUpdatedParticleIndexList_;

        // Profiling data
    std::int64_t numEvents_ = 0;
    std::int64_t numOutOfReach_ = 0;
    std::int64_t numBucketsOutOfReach_ = 0;
    std::int64_t numRejections_ = 0;
    std::int64_t numExcessSamplings_ = 0;
    std::int64_t numUpdates_ = 0;
    std::int64_t numBucketUpdates_ = 0;
    std::int64_t numBucketRecomputes_ = 0;
    std::int64_t numBucketChanges_ = 0;
    gsl::dim minNumBuckets_ = std::numeric_limits<gsl::dim>::max();
    gsl::dim maxNumBuckets_ = 0;
    gsl::dim minNumActiveParticles_ = std::numeric_limits<gsl::dim>::max();
    gsl::dim maxNumActiveParticles_ = 0;
    std::int64_t numBucketEvents_ = 0;
    std::int64_t numActiveParticleEvents_ = 0;


    class CollideCallback
    {
        friend RPMCOperator;

    private:
        RPMCOperator* self_;
        std::span<Index const> inactiveEntries_;

        explicit CollideCallback(RPMCOperator* _self)
            : self_(_self),
              inactiveEntries_(_self->bucketingScheme_.inactiveEntries())
        {
        }

    public:
        CollideCallback(CollideCallback const&) = delete;
        CollideCallback& operator =(CollideCallback const&) = delete;

        void
        invalidate(gsl::index j)
        {
            gsl_Expects(j >= 0 && j < self_->particleData_.num());

            self_->updatedParticleIndexList_.push_back(j);
        }
        gsl::index
        tryClone(gsl::index j)
        {
            gsl_Expects(j >= 0 && j < self_->particleData_.num());

            if (!inactiveEntries_.empty())
            {
                gsl::index k = inactiveEntries_.back();
                self_->updatedParticleIndexList_.push_back(k);
                self_->particleData_.clone(j, k);
                inactiveEntries_ = inactiveEntries_.subspan(0, inactiveEntries_.size() - 1);
                return k;
            }
            else
            {
                return -1;
            }
        }
    };

    ParticleProperties<>
    getAllParticleProperties(
        gsl::index j) const
    {
        return makeshift::tuple_transform<numParticleProperties>(
            [this, j](auto propertiesIndexC)
            {
                constexpr gsl::index modelIndex = propertiesToModelIndexMap[propertiesIndexC];
                auto const& interactionModel = std::get<modelIndex>(interactionModels_);

                auto ps = detail::gatherFromSoA(interactionModel.getState(), j);
                return getParticleProperties(interactionModel.getArgs(), ps);
            },
            makeshift::tuple_index);
    }

    gsl::dim
    getNumActiveParticles() const
    {
        return particleData_.num() - std::ssize(bucketingScheme_.inactiveEntries());
    }

    static void
    initSubBucketData(Bucket bucket)
    {
        for (auto subBucket : bucket.subBuckets())
        {
            subBucket.subBucketData().n = gsl::narrow_cast<Dim>(subBucket.numEntries());
        }
    }

    class BucketClassifier
    {
        RPMCOperator const& self_;

    public:
        BucketClassifier(RPMCOperator const& _self)
            : self_(_self)
        {
        }

        auto  // may be BucketLabel or std::optional<BucketLabel>
        classify(gsl::index j) const
        {
            return self_.classifier_.classify(j);
        }
        SubBucketLabel
        subclassify(ConstBucket bucket, gsl::index j) const
        {
            auto& bucketData = bucket.bucketData();
            auto loc = self_.locator_.location(j);
            return { gsl::narrow<LocationIndex>(std::floor(loc/bucketData.ds + 0.5)) };
        }
    };

    FastDs
    interactionWidthToDs(double w) const
    {
        auto P = locator_.interactionWidthToDistance(w);
        auto ds = gsl::narrow<FastDs>(std::ceil(P));
        return ds;
    }

    template <intervals::interval_arg LocationT>
    static std::tuple<LocationIndex, LocationIndex>
    subBucketsInReach(
        LocationT s1,
        FastDs ds1, FastDs ds2,
        FastDs ds12)
    {
        using namespace intervals::math;

        auto s1Min = infimum(s1);
        auto s1Max = supremum(s1);
        FastDs num1 = ds1*(2*s1Min - 1) - 2*ds12 + ds2;
        FastDs num2 = ds1*(2*s1Max + 1) + 2*ds12 - ds2;
        float denom = 2*ds2;
        return {
            gsl::narrow<LocationIndex>(std::floor(num1/denom)),
            gsl::narrow<LocationIndex>(std::ceil(num2/denom))
        };
    }

    template <gsl::index IModel>
    std::int64_t
    countInReach(
        ConstBucket bucket1, ConstBucket bucket2,
        FastDs ds12) const
    {
        if (std::isinf(ds12))
        {
            return bucket1.bucketData().n*std::int64_t(bucket2.bucketData().n);
        }

        //auto ds12 = ds12Of<IModel>(bucketingScheme_.bucketBucketData(bucket1, bucket2));
        auto subBuckets1 = bucket1.subBuckets();

        gsl::dim nsb1 = std::ranges::ssize(subBuckets1);
        if (nsb1 == 0)
        {
            return 0;
        }

            // TODO: check if determining the matching bucket subrange is worth it
        auto ds1 = bucket1.bucketData().ds;
        auto ds2 = bucket2.bucketData().ds;
        auto sb1FrontLabel = gsl::narrow_cast<FastDs>(subBuckets1[0].label()[0]);
        auto sb1BackLabel = gsl::narrow_cast<FastDs>(subBuckets1[nsb1 - 1].label()[0]);
        auto sb1Labels = intervals::interval{ sb1FrontLabel, sb1BackLabel };
        auto [sb2LabelsLo, sb2LabelsHi] = subBucketsInReach(sb1Labels, ds1, ds2, ds12);
        auto subBuckets2InRange = bucket2.subBucketsInRange({ sb2LabelsLo }, { gsl::narrow_cast<LocationIndex>(sb2LabelsHi + 1) });

            // We co-traverse all sub-buckets using the interaction radius to determine the reachability of other sub-buckets.
        //auto subBuckets2 = bucket2.subBuckets();
        //auto sb2First = subBuckets2.begin();
        //auto sb2End = subBuckets2.end();
        auto sb2First = subBuckets2InRange.begin();
        auto sb2End = subBuckets2InRange.end();
        std::int64_t n1n2 = 0;
        if (sb2First != sb2End)
        {
            auto sb2Last = sb2First;
            std::int64_t sn2w = 0;

            for (auto sb1 : subBuckets1)
            {
                std::int64_t sn1 = sb1.subBucketData().n;
                if (sn1 == 0)  // can happen especially at the fringes because of how `ExpandoArray<>` expands
                {
                    continue;
                }

                    // Move sliding window of reachability.
                auto sb1Label = gsl::narrow_cast<FastDs>(sb1.label()[0]);
                auto [sb2LabelLo, sb2LabelHi] = subBucketsInReach(sb1Label, ds1, ds2, ds12);
                auto sb2FirstLabel = (*sb2First).label()[0];
                while (sb2FirstLabel < sb2LabelLo)
                {
                    if (sb2Last == sb2First)
                    {
                        ++sb2Last;
                    }
                    else
                    {
                        std::int64_t sn2 = (*sb2First).subBucketData().n;
                        sn2w -= sn2;
                    }
                    ++sb2First;
                    if (sb2First == sb2End)
                    {
                        break;
                    }
                    sb2FirstLabel = (*sb2First).label()[0];
                }
                if (sb2First == sb2End)
                {
                    gsl_AssertDebug(sn2w == 0);
                    break;
                }
                if (sb2FirstLabel > sb2LabelHi)
                {
                    continue;
                }
                while (sb2Last != sb2End && (*sb2Last).label()[0] <= sb2LabelHi)
                {
                    std::int64_t sn2 = (*sb2Last).subBucketData().n;
                    sn2w += sn2;
                    ++sb2Last;
                }

                n1n2 += sn1*sn2w;
            }
        }
        gsl_Assert(n1n2 <= bucket1.bucketData().n*std::int64_t(bucket2.bucketData().n));  // TODO: this can be assumed only because of the processing order of sub-bucket deltas!
        return n1n2;
    }

    template <gsl::index IModel>
    std::tuple<ConstSubBucket, ConstSubBucket>
    locateSubBuckets(
        ConstBucket bucket1, ConstBucket bucket2,
        FastDs ds12,
        std::int64_t n1n2Stop) const
    {
        //auto ds12 = ds12Of<IModel>(bucketingScheme_.bucketBucketData(bucket1, bucket2));
        auto subBuckets1 = bucket1.subBuckets();

        gsl::dim nsb1 = std::ranges::ssize(subBuckets1);
        gsl_Assert(nsb1 != 0);

            // TODO: check if determining the matching bucket subrange is worth it
        auto ds1 = bucket1.bucketData().ds;
        auto ds2 = bucket2.bucketData().ds;
        auto sb1FrontLabel = gsl::narrow_cast<FastDs>(subBuckets1[0].label()[0]);
        auto sb1BackLabel = gsl::narrow_cast<FastDs>(subBuckets1[nsb1 - 1].label()[0]);
        auto sb1Labels = intervals::interval{ sb1FrontLabel, sb1BackLabel };
        auto [sb2LabelsLo, sb2LabelsHi] = subBucketsInReach(sb1Labels, ds1, ds2, ds12);
        auto subBuckets2InRange = bucket2.subBucketsInRange({ sb2LabelsLo }, { gsl::narrow_cast<LocationIndex>(sb2LabelsHi + 1) });

            // We co-traverse all sub-buckets using the interaction radius to determine the reachability of other sub-buckets.
        //auto subBuckets2 = bucket2.subBuckets();
        //auto sb2First = subBuckets2.begin();
        //auto sb2End = subBuckets2.end();
        auto sb2First = subBuckets2InRange.begin();
        auto sb2End = subBuckets2InRange.end();
        gsl_Assert(sb2First != sb2End);
        std::int64_t n1n2 = 0;
        auto sb2Last = sb2First;
        std::int64_t sn2w = 0;
        for (auto sb1 : bucket1.subBuckets())
        {
            std::int64_t sn1 = sb1.numEntries();
            if (sn1 == 0)  // can happen especially at the fringes because of how `ExpandoArray<>` expands
            {
                continue;
            }

                // Move sliding window of reachability.
            auto sb1Label = gsl::narrow_cast<FastDs>(sb1.label()[0]);
            auto [sb2LabelLo, sb2LabelHi] = subBucketsInReach(sb1Label, ds1, ds2, ds12);
            auto sb2FirstLabel = (*sb2First).label()[0];
            while (sb2FirstLabel < sb2LabelLo)
            {
                if (sb2Last == sb2First)
                {
                    ++sb2Last;
                }
                else
                {
                    std::int64_t sn2 = (*sb2First).subBucketData().n;
                    sn2w -= sn2;
                }
                ++sb2First;
                gsl_Assert(sb2First != sb2End);
                sb2FirstLabel = (*sb2First).label()[0];
            }
            if (sb2FirstLabel > sb2LabelHi)
            {
                continue;
            }
            while (sb2Last != sb2End && (*sb2Last).label()[0] <= sb2LabelHi)
            {
                std::int64_t sn2 = (*sb2Last).subBucketData().n;
                sn2w += sn2;
                ++sb2Last;
            }

            std::int64_t delta = sn1*sn2w;
            if (n1n2Stop < n1n2 + delta)
            {
                    // Found sub-bucket 1. Now look for sub-bucket 2.
                for (; sb2First != sb2Last; ++sb2First)
                {
                    std::int64_t sn2 = (*sb2First).numEntries();
                    std::int64_t sdelta = sn1*sn2;
                    if (n1n2Stop < n1n2 + sdelta)
                    {
                            // Found sub-bucket 2.
                        return std::tuple{ sb1, *sb2First };
                    }
                    n1n2 += sdelta;
                }
                gsl_FailFast();  // some impossible sliding window inconsistency
            }
            n1n2 += delta;
        }
        gsl_FailFast();  // n1n2Stop ≥ n1n2
    }
    std::tuple<ConstSubBucket, ConstSubBucket>
    locateSubBuckets(
        ConstBucket bucket1, ConstBucket bucket2,
        std::int64_t n1n2Stop) const
    {
        gsl_Expects(n1n2Stop >= 0);

            // Assuming an interaction radius of ∞, we successively traverse all sub-buckets of buckets 1 and 2.
            // (Sub-bucketing is kind of stupid in this case.)
        std::int64_t n1n2 = 0;
        auto n2 = bucket2.numEntries();
        for (auto sb1 : bucket1.subBuckets())
        {
            auto sn1 = sb1.numEntries();
            if (sn1 == 0)  // can happen especially at the fringes because of how `ExpandoArray<>` expands
            {
                continue;
            }
            auto delta = sn1*n2;
            if (n1n2Stop < n1n2 + delta)
            {
                    // Found sub-bucket 1. Now look for sub-bucket 2.
                for (auto sb2 : bucket2.subBuckets())
                {
                    auto sn2 = sb2.numEntries();
                    auto sdelta = sn1*sn2;
                    if (n1n2Stop < n1n2 + sdelta)
                    {
                            // Found sub-bucket 2.
                        return { sb1, sb2 };
                    }
                    n1n2 += sdelta;
                }
                gsl_FailFast();  // some impossible sliding window inconsistency
            }
            n1n2 += delta;
        }
        gsl_FailFast();  // n1n2Stop ≥ n1n2
    }

    template <gsl::index IModel>
    std::int64_t
    updateInReach(
        ConstBucket bucket1, ConstBucket bucket2,
        SubBucketUpdate const& subBucket1Update,
        FastDs ds12) const
    {
        auto ds1 = bucket1.bucketData().ds;
        auto ds2 = bucket2.bucketData().ds;
        //auto ds12 = ds12Of<IModel>(bucketingScheme_.bucketBucketData(bucket1, bucket2));
        auto sb1Label = gsl::narrow_cast<FastDs>(subBucket1Update.subBucketLabel[0]);
        auto [sb2LabelLo, sb2LabelHi] = subBucketsInReach(sb1Label, ds1, ds2, ds12);
        auto subBuckets2InRange = bucket2.subBucketsInRange({ sb2LabelLo }, { gsl::narrow_cast<LocationIndex>(sb2LabelHi + 1) });
        std::int64_t sb2nDelta = 0;
        for (auto sb2 : subBuckets2InRange)
        {
            std::int64_t sb2n = sb2.subBucketData().n;
            sb2nDelta += sb2n;
        }
        std::int64_t result;
        if (bucket1 != bucket2)
        {
            result = subBucket1Update.delta*sb2nDelta;
        }
        else
        {
            result = (2*sb2nDelta + subBucket1Update.delta)*subBucket1Update.delta;
        }
        return result;
    }

    void
    recomputeParticlePropertiesFor(gsl::index j)
    {
        particleProperties_[j] = getAllParticleProperties(j);
    }

    void
    recomputeParticleProperties()
    {
        for (gsl::index j = 0, n = particleData_.num(); j != n; ++j)
        {
            if (bucketingScheme_.isEntryActive(j))
            {
                recomputeParticlePropertiesFor(j);
            }
        }
    }

    void
    recomputeBucketProperties(Bucket bucket)
    {
        BucketData& bucketData = bucket.bucketData();
        auto numBuckets = std::ssize(bucketingScheme_.buckets());
        auto factor1 = 1. + args_.removalBucketUpdateDelay*(numBuckets - 1);
        auto factor2 = 1. + args_.rejectionBucketUpdateDelay*(numBuckets - 1);
        bucketData.updatesBeforeRecomputation1 = gsl::narrow_cast<Dim>(bucket.numEntries()*factor1);  // TODO: use something else in case of locality, where this seems excessive?
        auto numActiveParticles = getNumActiveParticles();
        bucketData.updatesBeforeRecomputation2 = gsl::narrow_cast<Dim>(numActiveParticles*factor2);
        bucketData.properties = obtainParticlePropertyBounds(bucket);
        bucketData.cumulativeInteractionRates = { };
        bucketData.needRecomputeInteractionRates.set();
        bucketData.touched = false;
    }

    ParticleProperties<intervals::set_of_t>
    obtainParticlePropertyBounds(ConstBucket bucket) const
    {
        auto bucketParticleStateBounds = ParticleState<intervals::set_of_t>{ };
        for (gsl::index i : bucket.entries())
        {
            makeshift::template_for(
                [this, i]
                (auto propertyIndexC, auto& particleStateBounds)
                {
                    constexpr gsl::index modelIndex = propertiesToModelIndexMap[propertyIndexC];
                    auto const& interactionModel = std::get<modelIndex>(interactionModels_);

                    auto ps = detail::gatherFromSoA(interactionModel.getState(), i);
                    makeshift::template_for(
                        [](auto& mdst, auto const& msrc)
                        {
                            intervals::assign_partial(mdst, msrc);
                        },
                        makeshift::tie_members(particleStateBounds), makeshift::tie_members(ps));
                },
                makeshift::tuple_index, bucketParticleStateBounds);
        }
        return makeshift::tuple_transform(
            [this, &bucket]
            (auto propertyIndexC, auto const& ps)
            {
                constexpr gsl::index modelIndex = propertiesToModelIndexMap[propertyIndexC];
                auto const& interactionModel = std::get<modelIndex>(interactionModels_);

                auto wps = classifier_.widen(ps);
                return getParticleProperties(interactionModel.getArgs(), wps);
            },
            makeshift::tuple_index, bucketParticleStateBounds);
    }

    void
    recomputeBucketData()
    {
            // We prioritize memory efficiency over exception safety here, and thus release the old memory before allocating
            // new memory.
        detail::destroyAndConstructInPlace(bucketingScheme_, particleData_.num(), BucketClassifier(*this));

            // With the bucketing scheme available, we now recompute the particle properties (which requires querying the
            // active/passive state of particles).
        recomputeParticleProperties();

            // We then proceed to compute interaction rates.
        auto buckets = bucketingScheme_.buckets();
        for (gsl::index iJ = 0, N = std::ssize(buckets); iJ != N; ++iJ)
        {
            auto bucket1 = buckets[iJ];
            BucketData& bucketData1 = bucket1.bucketData();

                // First initialize the bucket data for `bucket1`.
            recomputeBucketProperties(bucket1);
            bucketData1.n = gsl::narrow_cast<Dim>(bucket1.numEntries());
            bucketData1.initialized = true;

                // Compute the self-interaction rate.
            BucketBucketData& bucketBucketData_JJ = bucketingScheme_.bucketBucketData(bucket1, bucket1);
            auto isSelfInteraction = bucketData1.n != 1
                ? intervals::set{ IsSelfInteraction::no, IsSelfInteraction::yes }
                : intervals::set{ IsSelfInteraction::yes };
            double avgDs = 0;
            gsl::dim numDsAdded = 0;
            makeshift::template_for(
                [this, isSelfInteraction, &bucketData1, &bucketBucketData_JJ, &avgDs, &numDsAdded]
                <gsl::index IModel>
                (std::integral_constant<gsl::index, IModel>, auto const& interactionModel)
                {
                    using namespace intervals::logic;

                    constexpr gsl::index propertiesIndex = modelToPropertiesIndexMap[IModel];
                    auto const& icp1 = std::get<propertiesIndex>(bucketData1.properties());

                    if (detail::hasValue(icp1))
                    {
                        auto interactionData = interactionModel.template computeTracerSwarmInteractionData<intervals::set_of_t>(
                            detail::getValue(icp1), detail::getValue(icp1), isSelfInteraction);
                        gsl_Assert(always(interactionData.interactionRate_jk >= 0));
                        gsl_Assert(always(interactionData.interactionRate_kj >= 0));
                        //gsl_Assert(interactionData.interactionRate_jk == interactionData.interactionRate_kj);  // we waive this requirement to allow for numerical inaccuracy
                        bucketBucketData_JJ.interactionRates[IModel] = interactionData.interactionRate_jk.upper();

                        if constexpr (modelsHaveLocality[IModel])
                        {
                            if (bucketBucketData_JJ.interactionRates[IModel] > 0)
                            {
                                auto w = classifier_.widenInteractionWidth(interactionData.interactionWidth.upper());
                                gsl_Assert(w >= 0);
                                FastDs ds12 = interactionWidthToDs(w);
                                ds12Of<IModel>(bucketBucketData_JJ) = ds12;
                                if (ds12 > 0)
                                {
                                    avgDs += ds12;
                                    ++numDsAdded;
                                }
                            }
                        }
                    }
                    else
                    {
                        bucketBucketData_JJ.interactionRates[IModel] = 0;
                        if constexpr (modelsHaveLocality[IModel])
                        {
                            ds12Of<IModel>(bucketBucketData_JJ) = 0;
                        }
                    }
                },
                makeshift::tuple_index, interactionModels_);

                // Initialize sub-bucket classification for bucket.
            if constexpr (haveLocality)
            {
                if (numDsAdded > 1)
                {
                    avgDs /= numDsAdded;
                }
                auto ds1 = gsl::narrow<FastDs>(std::max<double>(locator_.minBinSize(), std::ceil(avgDs)));
                bucketData1.ds = ds1;
                bucketData1.haveProperDs = numDsAdded > 0;
                bucketingScheme_.updateLocationSubBuckets(bucket1, BucketClassifier(*this));
                initSubBucketData(bucket1);
            }

                // Compute cumulative self-interaction rate.
            makeshift::template_for<sizeof...(InteractionModelsT)>(
                [this, &bucket1, &bucketData1, &bucketBucketData_JJ]
                <gsl::index IModel>
                (std::integral_constant<gsl::index, IModel>)
                {
                    double& cumulativeInteractionRate_J = bucketData1.cumulativeInteractionRates[IModel];

                    if (bucketBucketData_JJ.interactionRates[IModel] > 0)
                    {
                        std::int64_t n1n2;
                        if constexpr (modelsHaveLocality[IModel])
                        {
                            auto ds12 = ds12Of<IModel>(bucketBucketData_JJ);
                            n1n2 = countInReach<IModel>(bucket1, bucket1, ds12);
                            n1n2Of<IModel>(bucketBucketData_JJ) = n1n2;
                        }
                        else
                        {
                            std::int64_t n1 = bucketData1.n;
                            n1n2 = n1*n1;
                        }

                        double newCumulativeInteractionRate_J = bucketBucketData_JJ.interactionRates[IModel]*n1n2;
                        cumulativeInteractionRate_J += newCumulativeInteractionRate_J;
                    }
                    else
                    {
                        if constexpr (modelsHaveLocality[IModel])
                        {
                            n1n2Of<IModel>(bucketBucketData_JJ) = 0;
                        }
                    }
                },
                makeshift::tuple_index);

                // Then iterate over the lower half of the bucket–bucket interaction matrix to make sure that all
                // bucket-specific data has already been computed.
            for (gsl::index iK = 0; iK < iJ; ++iK)
            {
                auto bucket2 = buckets[iK];
                BucketData& bucketData2 = bucket2.bucketData();

                BucketBucketData& bucketBucketData_JK = bucketingScheme_.bucketBucketData(bucket1, bucket2);
                BucketBucketData& bucketBucketData_KJ = bucketingScheme_.bucketBucketData(bucket2, bucket1);
                makeshift::template_for(
                    [this, &bucket1, &bucket2, &bucketData1, &bucketData2, &bucketBucketData_JK, &bucketBucketData_KJ]
                    <gsl::index IModel>
                    (std::integral_constant<gsl::index, IModel>, auto const& interactionModel)
                    {
                        using namespace intervals::logic;

                        constexpr gsl::index propertiesIndex = modelToPropertiesIndexMap[IModel];
                        auto const& icp1 = std::get<propertiesIndex>(bucketData1.properties());
                        auto const& icp2 = std::get<propertiesIndex>(bucketData2.properties());

                        double& cumulativeInteractionRate_J = bucketData1.cumulativeInteractionRates[IModel];
                        double& cumulativeInteractionRate_K = bucketData2.cumulativeInteractionRates[IModel];

                        if (detail::hasValue(icp1) && detail::hasValue(icp2))
                        {
                            auto interactionData = interactionModel.template computeTracerSwarmInteractionData<intervals::set_of_t>(
                                detail::getValue(icp1), detail::getValue(icp2), intervals::set{ IsSelfInteraction::no });
                            gsl_Assert(always(interactionData.interactionRate_jk >= 0));
                            gsl_Assert(always(interactionData.interactionRate_kj >= 0));
                            double newInteractionRate_jk = interactionData.interactionRate_jk.upper();
                            double newInteractionRate_kj = interactionData.interactionRate_kj.upper();
                            bucketBucketData_JK.interactionRates[IModel] = newInteractionRate_jk;
                            bucketBucketData_KJ.interactionRates[IModel] = newInteractionRate_kj;

                            std::int64_t n1n2;
                            if constexpr (modelsHaveLocality[IModel])
                            {
                                if (bucketBucketData_JK.interactionRates[IModel] > 0 || bucketBucketData_KJ.interactionRates[IModel] > 0)
                                {
                                    auto w = classifier_.widenInteractionWidth(interactionData.interactionWidth.upper());
                                    gsl_Assert(w >= 0);
                                    FastDs ds12 = interactionWidthToDs(w);
                                    ds12Of<IModel>(bucketBucketData_JK) = ds12;
                                    ds12Of<IModel>(bucketBucketData_KJ) = ds12;
                                    n1n2 = countInReach<IModel>(bucket1, bucket2, ds12);
                                }
                                else
                                {
                                    n1n2 = 0;
                                }
                                n1n2Of<IModel>(bucketBucketData_JK) = n1n2;
                                n1n2Of<IModel>(bucketBucketData_KJ) = n1n2;
                            }
                            else
                            {
                                std::int64_t n1 = bucketData1.n;
                                std::int64_t n2 = bucketData2.n;
                                n1n2 = n1*n2;
                            }

                            double newCumulativeInteractionRate_J = bucketBucketData_JK.interactionRates[IModel]*n1n2;
                            double newCumulativeInteractionRate_K = bucketBucketData_KJ.interactionRates[IModel]*n1n2;
                            cumulativeInteractionRate_J += newCumulativeInteractionRate_J;
                            cumulativeInteractionRate_K += newCumulativeInteractionRate_K;
                        }
                        else
                        {
                            bucketBucketData_JK.interactionRates[IModel] = 0;
                            bucketBucketData_KJ.interactionRates[IModel] = 0;
                            if constexpr (modelsHaveLocality[IModel])
                            {
                                n1n2Of<IModel>(bucketBucketData_JK) = 0;
                                n1n2Of<IModel>(bucketBucketData_KJ) = 0;
                                ds12Of<IModel>(bucketBucketData_JK) = 0;
                                ds12Of<IModel>(bucketBucketData_KJ) = 0;
                            }
                        }
                    },
                    makeshift::tuple_index, interactionModels_);
            }

            bucketData1.needRecomputeInteractionRates.reset();
        }
    }

    void
    recomputeTotalBucketInteractionRates()
    {
        auto totalBucketInteractionRates = InteractionRates{ };
        for (auto bucket : bucketingScheme_.buckets())
        {
            BucketData& bucketData = bucket.bucketData();
            makeshift::template_for(
                [](double& totalBucketInteractionRate, double cumulativeBucketInteractionRate)
                {
                    totalBucketInteractionRate += cumulativeBucketInteractionRate;
                },
                totalBucketInteractionRates, bucketData.cumulativeInteractionRates);
        }
        totalBucketInteractionRates_ = totalBucketInteractionRates;
    }

    void
    registerUpdateForProfiling(gsl::dim numBucketsUpdated, gsl::dim numBucketsRecomputed, gsl::dim numBucketChanges)
    {
        numBucketUpdates_ += numBucketsUpdated;
        numBucketRecomputes_ += numBucketsRecomputed;
        numBucketChanges_ += numBucketChanges;
    }
    void
    registerEventForProfiling()
    {
        auto numBuckets = std::ssize(bucketingScheme_.buckets());
        auto numActiveParticles = getNumActiveParticles();

        minNumBuckets_ = std::min(minNumBuckets_, numBuckets);
        maxNumBuckets_ = std::max(maxNumBuckets_, numBuckets);
        minNumActiveParticles_ = std::min(minNumActiveParticles_, numActiveParticles);
        maxNumActiveParticles_ = std::max(maxNumActiveParticles_, numActiveParticles);
        numBucketEvents_ += numBuckets;
        numActiveParticleEvents_ += numActiveParticles;
        ++numEvents_;
    }
    void
    registerRejectionForProfiling()
    {
        ++numRejections_;
    }

    void
    recomputeAll()
    {
        recomputeBucketData();
        recomputeTotalBucketInteractionRates();

        gsl::dim numBuckets = std::ssize(bucketingScheme_.buckets());
        registerUpdateForProfiling(numBuckets, numBuckets, numBuckets);
    }

    bool
    particleMayExceedBucketProperties(gsl::index i)
    {
        // This function is non-`const` because it will update the properties in the particle property cache if the
        // particle is active.

        bool wasActive = bucketingScheme_.isEntryActive(i);
        bool isActive = classifier_.isActive(i);

        if (isActive)
        {
                // Update entry in particle property cache.
            recomputeParticlePropertiesFor(i);
        }

            // Always return `true` in the case of an active → inactive or inactive → active transition.
        if (isActive != wasActive)
        {
            return true;
        }

            // Nothing to do if the particle was and remains inactive.
        if (!isActive)
        {
            return false;
        }

            // We henceforth assume that the particle is and was active. It must therefore have a corresponding bucket.
        auto bucket = bucketingScheme_.findBucketOfEntry(i);

            // Return whether any of the particle properties exceeds any of the bucket properties.
        BucketData const& bucketData = bucket.bucketData();
        return makeshift::template_any_of(
            []
            (auto const& bucketProperties, auto const& particleProperties)
            {
                if (detail::hasValue(particleProperties))
                {
                    if (detail::hasValue(bucketProperties))
                    {
                        return makeshift::template_any_of(
                            [](auto const& qBucket, auto const& q)
                            {
                                return !qBucket.contains(q);
                            },
                            makeshift::tie_members(detail::getValue(bucketProperties)), makeshift::tie_members(detail::getValue(particleProperties)));
                    }
                    else
                    {
                        return true;
                    }
                }
                return true;
            },
            bucketData.properties(), particleProperties_[i]);
    }
    void
    updateParticlePropertiesInBucket(Bucket bucket, gsl::index i)
    {
        BucketData& bucketData = bucket.bucketData();

            // Widen bucket property bounds so as to include updated particle properties.
        makeshift::template_for(
            [&needRecomputeInteractionRates = bucketData.needRecomputeInteractionRates]
            (auto propertiesIndexC, auto& bucketProperties, auto const& particleProperties)
            {
                if (detail::hasValue(particleProperties))
                {
                    bool anyBucketPropertyExceeded = false;
                    if (detail::hasValue(bucketProperties))
                    {
                        anyBucketPropertyExceeded = makeshift::template_any_of(
                            [](auto const& qBucket, auto const& q)
                            {
                                return !qBucket.contains(q);
                            },
                            makeshift::tie_members(detail::getValue(bucketProperties)), makeshift::tie_members(detail::getValue(particleProperties)));
                    }
                    else
                    {
                        anyBucketPropertyExceeded = true;
                    }

                        // Extend bucket properties such that they contain the particle's properties, and set the flag
                        // for having interaction rates recomputed.
                    if (anyBucketPropertyExceeded)
                    {
                        needRecomputeInteractionRates.set(propertiesIndexC);
                        makeshift::template_for(
                            [](auto& mdst, auto const& msrc)
                            {
                                intervals::assign_partial(mdst, msrc);
                            },
                            makeshift::tie_members(detail::getOrInitValue(bucketProperties)),
                            makeshift::tie_members(detail::getValue(particleProperties)));
                    }
                }
            },
            makeshift::tuple_index, bucketData.properties(), particleProperties_[i]);

            // If the altered particle does not extend the bucket property bounds, this indicates a recomputation might
            // tighten the bounds. However, considering that other changes, such as subtracting mass from the swarm, may
            // also warrant a recomputation, we decrement the counter unconditionally.
        --bucketData.updatesBeforeRecomputation1;
    }

    bool
    updateInteractionRatesForBucket(Bucket bucket1, std::span<SubBucketUpdate const> bucket1SubBucketUpdates = { })
    {
        BucketData& bucketData1 = bucket1.bucketData();
        std::int64_t n1Old = bucketData1.n;
        std::int64_t n1New = bucket1.numEntries();

        if (n1New == n1Old && bucketData1.needRecomputeInteractionRates.none() && bucket1SubBucketUpdates.empty())
        {
                // The bucket and sub-bucket particle counts haven't changed, and the particle properties haven't changed, either.
                // Interaction rates thus need not be recomputed.
            return false;
        }

            // First, recompute all interaction rates and interaction radii that need to be recomputed.
        bool subBucketsInit = !bucketData1.initialized;
        if constexpr (haveLocality)
        {
            gsl_AssertDebug(!subBucketsInit || bucketData1.needRecomputeInteractionRates.all());
        }
        [[maybe_unused]] bool needsUpdating = false;
        //for (Bucket bucket2 : bucketingScheme_.buckets())
        auto buckets = bucketingScheme_.buckets();
        gsl::dim nK = std::ssize(buckets);
        for (gsl::index iK = 0; iK < nK; ++iK)
        {
            auto&& bucket2 = buckets[iK];
            bool sameBucket = bucket2 == bucket1;
            BucketData& bucketData2 = bucket2.bucketData();
            if (!sameBucket && !bucketData2.initialized)
            {
                    // The bucket has just been added and will be updated on its own later.
                continue;
            }

            BucketBucketData& bucketBucketData_JK = bucketingScheme_.bucketBucketData(bucket1, bucket2);
            BucketBucketData& bucketBucketData_KJ = bucketingScheme_.bucketBucketData(bucket2, bucket1);

            auto isSelfInteraction = !sameBucket ? intervals::set{ IsSelfInteraction::no }
                : n1New != 1 ? intervals::set{ IsSelfInteraction::no, IsSelfInteraction::yes }
                : intervals::set{ IsSelfInteraction::yes };
            std::int64_t n2New = sameBucket ? n1New : bucketData2.n;
            bool lneedsUpdating = false;
            makeshift::template_for(
                [
                    this,  // for `classifier_`
                    sameBucket, isSelfInteraction,
                    &bucketData1, &bucketData2,
                    &bucketBucketData_JK, &bucketBucketData_KJ,
                    n1Old, n1New, n2New,
                    subBucketsInit,
                    &lneedsUpdating
                ]
                <gsl::index IModel>
                (std::integral_constant<gsl::index, IModel>, auto const& interactionModel)
                {
                    using namespace intervals::logic;

                        // Subtract the previous interaction rate from the cumulative interaction rate.
                    std::int64_t n2n1Old;
                    if constexpr (modelsHaveLocality[IModel])
                    {
                        n2n1Old = n1n2Of<IModel>(bucketBucketData_KJ);
                    }
                    else
                    {
                        std::int64_t n2Old = bucketData2.n;
                        n2n1Old = n2Old*n1Old;
                    }
                    double& cumulativeInteractionRate_K = bucketData2.cumulativeInteractionRates[IModel];
                    double cumulativeBucketBucketInteractionRateDelta_KJ = -bucketBucketData_KJ.interactionRates[IModel]*n2n1Old;
                    cumulativeInteractionRate_K += cumulativeBucketBucketInteractionRateDelta_KJ;

                        // If necessary, recompute the interaction rate.
                    constexpr gsl::index propertiesIndex = modelToPropertiesIndexMap[IModel];
                    auto const& icp1 = std::get<propertiesIndex>(bucketData1.properties());
                    auto const& icp2 = std::get<propertiesIndex>(bucketData2.properties());
                    if (n1New > 0 && n2New > 0 && detail::hasValue(icp1) && detail::hasValue(icp2))
                    {
                        if (bucketData1.needRecomputeInteractionRates[propertiesIndex])  // implied by `subBucketsInit`
                        {
                            auto interactionData = interactionModel.template computeTracerSwarmInteractionData<intervals::set_of_t>(
                                detail::getValue(icp1), detail::getValue(icp2), isSelfInteraction);
                            gsl_Assert(always(interactionData.interactionRate_jk >= 0));
                            gsl_Assert(always(interactionData.interactionRate_kj >= 0));
                            double newInteractionRate_jk = interactionData.interactionRate_jk.upper();
                            double newInteractionRate_kj = interactionData.interactionRate_kj.upper();
                            [[maybe_unused]] bool interactionRateWas0 = !(bucketBucketData_JK.interactionRates[IModel] > 0 || bucketBucketData_KJ.interactionRates[IModel] > 0);
                            bucketBucketData_JK.interactionRates[IModel] = newInteractionRate_jk;
                            bucketBucketData_KJ.interactionRates[IModel] = newInteractionRate_kj;
                            if constexpr (modelsHaveLocality[IModel])
                            {
                                    // If the interaction rate is nonzero, and if it either was zero before, or the interaction radius
                                    // exceeds the old radius, or the entire bucket is being initialized, we need to revisit the full
                                    // range of sub-buckets in a sliding window check.
                                bool llneedsUpdating = true;
                                if (newInteractionRate_jk > 0 || newInteractionRate_kj > 0)
                                {
                                    auto wRaw = interactionData.interactionWidth.upper();
                                    gsl_Assert(wRaw >= 0);
                                    FastDs ds12Raw = interactionWidthToDs(wRaw);
                                    if (subBucketsInit || interactionRateWas0 || ds12Raw > ds12Of<IModel>(bucketBucketData_JK))
                                    {
                                            // Widen only when recounting.
                                        auto w = classifier_.widenInteractionWidth(wRaw);
                                        FastDs ds12 = interactionWidthToDs(w);
                                        ds12Of<IModel>(bucketBucketData_JK) = ds12;
                                        ds12Of<IModel>(bucketBucketData_KJ) = ds12;

                                            // This is a temporary flag; it is only ever set for "JK", not "KJ", and will be cleared again before leaving the routine.
                                        needRecountFlagsOf<IModel>(bucketBucketData_JK).set(IModel);
                                        llneedsUpdating = false;
                                    }
                                }

                                    // Unless a recount flag has been set, there is at least one interaction radius which hasn't grown and
                                    // for which the sub-buckets don't need to be recounted, so we must apply the updates.
                                if (!interactionRateWas0)
                                {
                                    lneedsUpdating = lneedsUpdating || llneedsUpdating;
                                }
                            }
                        }
                        else
                        {
                                // There is at least one interaction radius which hasn't grown and for which the
                                // sub-buckets don't need to be recounted, so we must apply the updates.
                            lneedsUpdating = true;
                        }
                    }
                    else
                    {
                        bucketBucketData_JK.interactionRates[IModel] = 0;
                        bucketBucketData_KJ.interactionRates[IModel] = 0;
                        if constexpr (modelsHaveLocality[IModel])
                        {
                            ds12Of<IModel>(bucketBucketData_JK) = 0;
                            ds12Of<IModel>(bucketBucketData_KJ) = 0;
                        }
                    }
                },
                makeshift::tuple_index, interactionModels_);
            needsUpdating = needsUpdating || lneedsUpdating;
        }

            // Now update the bucket count and apply the sub-bucket updates sequentially, or initialize the sub-buckets if initialization is due.
        bucketData1.n = gsl::narrow_cast<Dim>(n1New);
        bucketData1.initialized = true;
        if constexpr (haveLocality)
        {
            if (subBucketsInit || !bucketData1.haveProperDs)
            {
                BucketBucketData& bucketBucketData_JJ = bucketingScheme_.bucketBucketData(bucket1, bucket1);
                double avgDs = 0;
                gsl::dim numDsAdded = 0;
                makeshift::template_for<sizeof...(InteractionModelsT)>(
                    [&bucketData1, &bucketBucketData_JJ, &avgDs, &numDsAdded]
                    <gsl::index IModel>
                    (std::integral_constant<gsl::index, IModel>)
                    {
                        if constexpr (modelsHaveLocality[IModel])
                        {
                            auto ds = ds12Of<IModel>(bucketBucketData_JJ);
                            if (ds > 0)
                            {
                                avgDs += ds;
                                ++numDsAdded;
                            }
                        }
                    },
                    makeshift::tuple_index);
                if (numDsAdded > 1)
                {
                    avgDs /= numDsAdded;
                }
                if (subBucketsInit || numDsAdded > 0)
                {
                        // We initialize sub-buckets that haven't been initialized yet, but also re-initialize them if the old
                        // bucket interaction distance was 0 (i.e., stopgap).
                    auto ds1 = gsl::narrow<FastDs>(std::max<double>(locator_.minBinSize(), std::ceil(avgDs)));
                    bucketData1.ds = ds1;
                    bucketData1.haveProperDs = numDsAdded > 0;
                    bucketingScheme_.updateLocationSubBuckets(bucket1, BucketClassifier(*this));
                    initSubBucketData(bucket1);
                    subBucketsInit = true;
                }
            }
            if (!subBucketsInit)
            {
                for (auto&& bucket1SubBucketUpdate : bucket1SubBucketUpdates)
                {
                    if (needsUpdating)
                    {
                        for (Bucket bucket2 : bucketingScheme_.buckets())
                        {
                            auto& bucketData2 = bucket2.bucketData();
                            if (!bucketData2.initialized)
                            {
                                    // The bucket has just been added and will be updated on its own later.
                                continue;
                            }

                                // Non-negative interaction radius
                                // ⇒ interaction radius still in range
                                // ⇒ an update will do
                            BucketBucketData& bucketBucketData_JK = bucketingScheme_.bucketBucketData(bucket1, bucket2);
                            BucketBucketData& bucketBucketData_KJ = bucketingScheme_.bucketBucketData(bucket2, bucket1);
                            makeshift::template_for<sizeof...(InteractionModelsT)>(
                                [
                                    this,  // for `classifier_`, via `updateInReach<>()`
                                    bucket1, bucket2,
                                    bucket1SubBucketUpdate,
                                    &bucketBucketData_JK, &bucketBucketData_KJ
                                ]
                                <gsl::index IModel>
                                (std::integral_constant<gsl::index, IModel>)
                                {
                                    if constexpr (modelsHaveLocality[IModel])
                                    {
                                        bool anyInteractionRate = bucketBucketData_JK.interactionRates[IModel] > 0 || bucketBucketData_KJ.interactionRates[IModel] > 0;
                                        if (anyInteractionRate && !needRecountFlagsOf<IModel>(bucketBucketData_JK).test(IModel))
                                        {
                                            std::int64_t n1n2 = n1n2Of<IModel>(bucketBucketData_JK);
                                            auto delta = updateInReach<IModel>(
                                                bucket1, bucket2, bucket1SubBucketUpdate, ds12Of<IModel>(bucketBucketData_JK));
                                            n1n2 += delta;
                                            gsl_AssertDebug(n1n2 >= 0 && n1n2 <= std::int64_t(bucket1.bucketData().n)*std::int64_t(bucket2.bucketData().n));  // TODO: this can be assumed only because of the processing order of sub-bucket deltas!
                                            n1n2Of<IModel>(bucketBucketData_JK) = n1n2;
                                            n1n2Of<IModel>(bucketBucketData_KJ) = n1n2;

#ifndef NDEBUG
                                                // TODO: remove
                                            auto sb1 = bucket1.subBucket(bucket1SubBucketUpdate.subBucketLabel);
                                            sb1.subBucketData().n += bucket1SubBucketUpdate.delta;
                                            auto n1n2Fiducial = countInReach<IModel>(bucket1, bucket2, ds12Of<IModel>(bucketBucketData_JK));
                                            gsl_Assert(n1n2 == n1n2Fiducial);
                                            sb1.subBucketData().n -= bucket1SubBucketUpdate.delta;
#endif // NDEBUG
                                        }
                                    }
                                },
                                makeshift::tuple_index);
                        }
                    }
                    auto sb1 = bucket1.subBucket(bucket1SubBucketUpdate.subBucketLabel);
                    gsl_AssertDebug(sb1.subBucketData().n + bucket1SubBucketUpdate.delta == sb1.numEntries());
                    sb1.subBucketData().n = gsl::narrow_cast<Dim>(sb1.numEntries());
                }
            }
        }

            // Initialize the cumulative self-interaction rate.
        makeshift::template_for<sizeof...(InteractionModelsT)>(
            [&bucketData1]
            <gsl::index IModel>
            (std::integral_constant<gsl::index, IModel>)
            {
                bucketData1.cumulativeInteractionRates[IModel] = 0;
            },
            makeshift::tuple_index);

            // Finally, go through all bucket–bucket interactions and update cumulative collision rates,
            // recounting possible interactions if necessary.
        for (Bucket bucket2 : bucketingScheme_.buckets())
        {
            BucketData& bucketData2 = bucket2.bucketData();
            if (!bucketData2.initialized)
            {
                    // The bucket has just been added and will be updated on its own later.
                continue;
            }

            BucketBucketData& bucketBucketData_JK = bucketingScheme_.bucketBucketData(bucket1, bucket2);
            BucketBucketData& bucketBucketData_KJ = bucketingScheme_.bucketBucketData(bucket2, bucket1);
            makeshift::template_for<sizeof...(InteractionModelsT)>(
                [
                    this, // for `classifier_`, via `countInReach<>()`
                    bucket1, bucket2,
                    &bucketData1, &bucketData2,
                    &bucketBucketData_JK, &bucketBucketData_KJ,
                    n1New,
                    subBucketsInit
                ]
                <gsl::index IModel>
                (std::integral_constant<gsl::index, IModel>)
                {
                    double& cumulativeInteractionRate_J = bucketData1.cumulativeInteractionRates[IModel];
                    double& cumulativeInteractionRate_K = bucketData2.cumulativeInteractionRates[IModel];

                    std::int64_t n1n2;
                    if constexpr (modelsHaveLocality[IModel])
                    {
                        if (subBucketsInit || needRecountFlagsOf<IModel>(bucketBucketData_JK).test(IModel))
                        {
                                // If the sub-buckets need to be recounted (e.g. because the new interaction radius exceeds the old value),
                                // we need to revisit the full range of buckets in a sliding window traversal.
                            n1n2 = countInReach<IModel>(bucket1, bucket2, ds12Of<IModel>(bucketBucketData_JK));
                            n1n2Of<IModel>(bucketBucketData_JK) = n1n2;
                            n1n2Of<IModel>(bucketBucketData_KJ) = n1n2;
                            needRecountFlagsOf<IModel>(bucketBucketData_JK).reset(IModel);
                        }
                        else
                        {
                            n1n2 = n1n2Of<IModel>(bucketBucketData_JK);
                        }
                    }
                    else
                    {
                        std::int64_t n2 = bucketData2.n;
                        n1n2 = n1New*n2;
                    }

                        // Order reads and writes such that the increment is added only once for J = K.
                    auto cumulativeBucketBucketInteractionRateDelta_JK = bucketBucketData_JK.interactionRates[IModel]*n1n2;
                    auto cumulativeBucketBucketInteractionRateDelta_KJ = bucketBucketData_KJ.interactionRates[IModel]*n1n2;
                    auto oldCumulativeInteractionRate_K = cumulativeInteractionRate_K;
                    cumulativeInteractionRate_J += cumulativeBucketBucketInteractionRateDelta_JK;
                    cumulativeInteractionRate_K = std::max(0., oldCumulativeInteractionRate_K + cumulativeBucketBucketInteractionRateDelta_KJ);
                },
                makeshift::tuple_index);
        }

        bucketData1.needRecomputeInteractionRates.reset();
        return true;
    }

    class BucketUpdater : public DefaultBucketUpdater<LBucketingScheme>
    {
    private:
        RPMCOperator& self_;
        gsl::dim numBucketUpdates_ = 0;
        gsl::dim numBucketRecomputes_ = 0;
        gsl::dim numBucketChanges_ = 0;
        bool anyInteractionRateUpdated_ = false;

    public:
        BucketUpdater(RPMCOperator& _self)
            : self_(_self)
        {
        }

        bool
        anyInteractionRateUpdated() const
        {
            return anyInteractionRateUpdated_;
        }
        gsl::dim
        numBucketUpdates() const
        {
            return numBucketUpdates_;
        }
        gsl::dim
        numBucketRecomputes() const
        {
            return numBucketRecomputes_;
        }
        gsl::dim
        numBucketChanges() const
        {
            return numBucketChanges_;
        }

        void
        onEntryAddedToBucket(Bucket bucket, gsl::index i)
        {
            BucketData& bucketData = bucket.bucketData();

            bucketData.touched = true;
            if (bucketData.updatesBeforeRecomputation1 > 0 && bucketData.updatesBeforeRecomputation2 > 0)  // recomputation is not due yet
            {
                    // Widen bucket properties to avoid recomputation.
                self_.updateParticlePropertiesInBucket(bucket, i);
            }
        }
        void
        onEntryUpdatedInBucket(Bucket bucket, gsl::index i)
        {
            BucketData& bucketData = bucket.bucketData();

            bucketData.touched = true;
            if (bucket.numEntries() == 1)
            {
                    // The particle is the only particle in the bucket; trigger an unconditional recomputation
                    // of bucket properties to avoid excessive bucket bounds.
                bucketData.updatesBeforeRecomputation1 = 0;
            }
            else if (bucketData.updatesBeforeRecomputation1 > 0 && bucketData.updatesBeforeRecomputation2 > 0)  // recomputation is not due yet
            {
                    // Widen bucket properties to avoid recomputation.
                self_.updateParticlePropertiesInBucket(bucket, i);
            }
        }
        void
        onRemovingEntryFromBucket(Bucket bucket, gsl::index /*i*/)
        {
            BucketData& bucketData = bucket.bucketData();

            bucketData.touched = true;
            --bucketData.updatesBeforeRecomputation1;
        }

        void
        onBucketAdded(Bucket bucket, std::span<SubBucketUpdate const> subBucketUpdates = { })
        {
            self_.recomputeBucketProperties(bucket);
            bool updated = self_.updateInteractionRatesForBucket(bucket, subBucketUpdates);
            ++numBucketChanges_;
            ++numBucketUpdates_;
            ++numBucketRecomputes_;
            gsl_Assert(updated);
            anyInteractionRateUpdated_ = true;
        }
        void
        onBucketUpdated(Bucket bucket, std::span<SubBucketUpdate const> subBucketUpdates = { })
        {
            BucketData& bucketData = bucket.bucketData();

            if (!(bucketData.updatesBeforeRecomputation1 > 0 && bucketData.updatesBeforeRecomputation2 > 0))  // recomputation is due
            {
                self_.recomputeBucketProperties(bucket);
            }
            ++numBucketChanges_;
            bool anyRateNeededUpdating = bucket.bucketData().needRecomputeInteractionRates.any();
            bool updated = self_.updateInteractionRatesForBucket(bucket, subBucketUpdates);
            if (updated)
            {
                ++numBucketUpdates_;
                if (anyRateNeededUpdating)
                {
                    ++numBucketRecomputes_;
                }
            }
            anyInteractionRateUpdated_ |= updated;
        }
        void
        onRemovingBucket(Bucket bucket, std::span<SubBucketUpdate const> subBucketUpdates = { })
        {
            bucket.bucketData().needRecomputeInteractionRates.reset();
            bool updated = self_.updateInteractionRatesForBucket(bucket, subBucketUpdates);
            gsl_Assert(updated);
            ++numBucketChanges_;
            ++numBucketUpdates_;
            anyInteractionRateUpdated_ = true;
        }
    };

    void
    updateParticles(std::span<gsl::index const> updatedParticleIndices)
    {
        numUpdates_ += std::ssize(updatedParticleIndices);

            // Skip all particles which still fit the same bucket.
        internalUpdatedParticleIndexList_.resize(updatedParticleIndices.size());
        gsl::index j = 0;
        for (gsl::index i = 0, nList = std::ssize(updatedParticleIndices); i != nList; ++i)
        {
            gsl::index pidx = updatedParticleIndices[i];
            if (particleMayExceedBucketProperties(pidx))
            {
                internalUpdatedParticleIndexList_[j] = pidx;
                ++j;
            }
        }
        internalUpdatedParticleIndexList_.resize(j);

        if (!internalUpdatedParticleIndexList_.empty())
        {
                // Update the remaining particles.
            auto updater = BucketUpdater{ *this };
            bucketingScheme_.updateEntries(internalUpdatedParticleIndexList_, BucketClassifier(*this), updater);
            if (updater.anyInteractionRateUpdated())
            {
                recomputeTotalBucketInteractionRates();
            }
            registerUpdateForProfiling(updater.numBucketUpdates(), updater.numBucketRecomputes(), updater.numBucketChanges());
        }
    }

    template <gsl::index IModel, typename AccessorFuncT>
    void
    inspectInteractionProperties(InspectionData const& inspectionData, AccessorFuncT&& accessor, bool transpose = false) const
    {
        auto& interactionModel = std::get<IModel>(interactionModels_);
        constexpr gsl::index propertiesIndex = modelToPropertiesIndexMap[IModel];

        gsl::dim numActiveParticles = getNumActiveParticles();
        gsl::stride sRow = transpose ? 1 : numActiveParticles;
        gsl::stride sCol = transpose ? numActiveParticles : 1;
        gsl_Assert(std::ssize(inspectionData.fdst) == rpmc::square_checked_failfast(numActiveParticles));
        gsl::index iRow = 0;
        auto buckets = bucketingScheme_.buckets();
        for (auto bucket1 : buckets)
        {
            for (gsl::index j : bucket1.entries())
            {
                gsl::index iCol = 0;
                for (auto bucket2 : buckets)
                {
                    for (gsl::index k : bucket2.entries())
                    {
                        auto& p1 = particleProperties_[j];
                        auto& p2 = particleProperties_[k];

                        auto& icp1 = std::get<propertiesIndex>(p1);
                        auto& icp2 = std::get<propertiesIndex>(p2);
                        if (detail::hasValue(icp1) && detail::hasValue(icp2))
                        {
                            auto isSelfInteraction = j == k ? IsSelfInteraction::yes : IsSelfInteraction::no;
                            auto result1 = interactionModel.template computeTracerSwarmInteractionData<std::type_identity_t>(
                                detail::getValue(icp1), detail::getValue(icp2), isSelfInteraction);
                            auto property = accessor(result1, transpose);
                            inspectionData.fdst[iRow*sRow + iCol*sCol] = property;
                        }
                        else
                        {
                            inspectionData.fdst[iRow*sRow + iCol*sCol] = 0;
                        }
                        ++iCol;
                    }
                }
                ++iRow;
            }
        }
    }
    template <gsl::index IModel, typename AccessorFuncT, typename BucketAccessorFuncT>
    void
    inspectAcceptanceProbabilities(InspectionData const& inspectionData, AccessorFuncT&& accessor, BucketAccessorFuncT&& bucketAccessor) const
    {
        auto& interactionModel = std::get<IModel>(interactionModels_);
        constexpr gsl::index propertiesIndex = modelToPropertiesIndexMap[IModel];

        gsl::dim numActiveParticles = getNumActiveParticles();
        gsl_Assert(std::ssize(inspectionData.fdst) == rpmc::square_checked_failfast(numActiveParticles));
        gsl::index iRow = 0;
        auto buckets = bucketingScheme_.buckets();
        for (auto bucket1 : buckets)
        {
            for (gsl::index j : bucket1.entries())
            {
                gsl::index iCol = 0;
                for (auto bucket2 : buckets)
                {
                    BucketBucketData const& bucketBucketData = bucketingScheme_.bucketBucketData(bucket1, bucket2);
                    //double bucketInteractionRate = bucketBucketData.interactionRates[IModel];
                    double bucketProperty = bucketAccessor(bucketBucketData);

                    for (gsl::index k : bucket2.entries())
                    {
                        auto& p1 = particleProperties_[j];
                        auto& p2 = particleProperties_[k];

                        auto& icp1 = std::get<propertiesIndex>(p1);
                        auto& icp2 = std::get<propertiesIndex>(p2);
                        if (detail::hasValue(icp1) && detail::hasValue(icp2))
                        {
                            auto isSelfInteraction = j == k ? IsSelfInteraction::yes : IsSelfInteraction::no;
                            auto result1 = interactionModel.template computeTracerSwarmInteractionData<std::type_identity_t>(
                                detail::getValue(icp1), detail::getValue(icp2), isSelfInteraction);
                            //auto interactionRate = result1.interactionRate_jk;
                            auto property = accessor(result1);
                            double pAccept = property/bucketProperty;
                            if (pAccept > 1.)
                            {
                                pAccept = NAN;
                            }
                            inspectionData.fdst[iRow*numActiveParticles + iCol] = pAccept;
                        }
                        else
                        {
                            inspectionData.fdst[iRow*numActiveParticles + iCol] = 0;
                        }
                        ++iCol;
                    }
                }
                ++iRow;
            }
        }
    }
    template <typename BucketAccessorFuncT>
    void
    inspectParticleBucketProperties(InspectionData const& inspectionData, BucketAccessorFuncT&& bucketAccessor) const
    {
        gsl::dim numActiveParticles = getNumActiveParticles();
        gsl_Assert(std::ssize(inspectionData.fdst) == rpmc::square_checked_failfast(numActiveParticles));
        gsl::index iRow = 0;
        auto buckets = bucketingScheme_.buckets();
        for (auto bucket1 : buckets)
        {
            for ([[maybe_unused]] gsl::index j : bucket1.entries())
            {
                gsl::index iCol = 0;
                for (auto bucket2 : buckets)
                {
                    BucketBucketData const& bucketBucketData = bucketingScheme_.bucketBucketData(bucket1, bucket2);
                    auto bucketProperty = bucketAccessor(bucketBucketData);
                    for ([[maybe_unused]] gsl::index k : bucket2.entries())
                    {
                        inspectionData.fdst[iRow*numActiveParticles + iCol] = bucketProperty;
                        ++iCol;
                    }
                }
                ++iRow;
            }
        }
    }
    template <gsl::index IModel, typename AccessorFuncT>
    void
    inspectTrueParticleBucketProperties(InspectionData const& inspectionData, AccessorFuncT&& accessor, bool transpose = false) const
    {
        gsl::dim numActiveParticles = getNumActiveParticles();
        gsl::stride sRow = transpose ? 1 : numActiveParticles;
        gsl::stride sCol = transpose ? numActiveParticles : 1;
        gsl_Assert(std::ssize(inspectionData.fdst) == rpmc::square_checked_failfast(numActiveParticles));
        gsl::index iRow = 0;
        auto buckets = bucketingScheme_.buckets();
        auto const& interactionModel = std::get<IModel>(interactionModels_);
        for (auto bucket1 : buckets)
        {
            auto b1p = obtainParticlePropertyBounds(bucket1);
            for ([[maybe_unused]] gsl::index j : bucket1.entries())
            {
                gsl::index iCol = 0;
                for (auto bucket2 : buckets)
                {
                    auto b2p = obtainParticlePropertyBounds(bucket2);
                    constexpr gsl::index propertiesIndex = modelToPropertiesIndexMap[IModel];
                    auto const& icp1 = std::get<propertiesIndex>(bucket1.bucketData().properties());
                    auto const& icp2 = std::get<propertiesIndex>(bucket2.bucketData().properties());
                    double interactionRate;
                    if (detail::hasValue(icp1) && detail::hasValue(icp2))
                    {
                        auto isSelfInteraction = bucket1 != bucket2 ? intervals::set{ IsSelfInteraction::no }
                            : bucket1.numEntries() == 1 ? intervals::set{ IsSelfInteraction::yes } : intervals::set{ IsSelfInteraction::no, IsSelfInteraction::yes };
                        auto interactionData = interactionModel.template computeTracerSwarmInteractionData<intervals::set_of_t>(
                            detail::getValue(icp1), detail::getValue(icp2), isSelfInteraction);
                        //interactionRate = interactionData.interactionRate_jk.upper();
                        interactionRate = accessor(interactionData, transpose).upper();
                    }
                    else
                    {
                        interactionRate = 0.;
                    }
    
                    for ([[maybe_unused]] gsl::index k : bucket2.entries())
                    {
                        inspectionData.fdst[iRow*sRow + iCol*sCol] = interactionRate;
                        ++iCol;
                    }
                }
                ++iRow;
            }
        }
    }
    template <typename BucketAccessorFuncT>
    void
    inspectBucketProperties(InspectionData const& inspectionData, BucketAccessorFuncT&& bucketAccessor) const
    {
        auto buckets = bucketingScheme_.buckets();
        gsl::dim numBuckets = std::ssize(buckets);
        gsl_Assert(std::ssize(inspectionData.fdst) == rpmc::square_checked_failfast(numBuckets));
        gsl::index iRow = 0;
        for (auto bucket1 : buckets)
        {
            gsl::index iCol = 0;
            for (auto bucket2 : buckets)
            {
                BucketBucketData const& bucketBucketData = bucketingScheme_.bucketBucketData(bucket1, bucket2);
                auto bucketProperty = bucketAccessor(bucketBucketData);
                inspectionData.fdst[iRow*numBuckets + iCol] = bucketProperty;
                ++iCol;
            }
            ++iRow;
        }
    }

    bool
    inspectInteractionModel(std::string_view quantity, InspectionData const& inspectionData, gsl::index iModel) const
    {
        using namespace std::literals;

        auto iModelV = makeshift::expand_failfast(iModel,
            MAKESHIFT_CONSTVAL(makeshift::array_iota<sizeof...(InteractionModelsT), gsl::index>()));
    
        auto interactionRateAccessor = []
        (auto const& interactionData, bool transpose = false)
        {
            return transpose
                ? interactionData.interactionRate_kj
                : interactionData.interactionRate_jk;
        };
        auto bucketInteractionRateAccessor = [iModel]
        (BucketBucketData const& bucketBucketData)
        {
            return bucketBucketData.interactionRates[iModel];
        };
        auto interactionDistanceAccessor = [this]
        (auto const& interactionData, bool /*transpose*/ = false)
        {
            return interactionWidthToDs(interactionData.interactionWidth);
        };
        auto bucketInteractionDistanceAccessor = [iDistance = modelInteractionDistanceIndices[iModel]]
        (LocalBucketBucketData const& bucketBucketData)
        {
            return bucketBucketData.ds12s[iDistance];
        };

        if (quantity == "interaction rates"sv)
        {
            makeshift::visit(
                [this, &inspectionData, &interactionRateAccessor]
                <gsl::index IModel>(std::integral_constant<gsl::index, IModel>)
                {
                    this->inspectInteractionProperties<IModel>(inspectionData, interactionRateAccessor, false);
                },
                iModelV);
            return true;
        }
        else if (quantity == "interaction rates, reverse"sv)
        {
            makeshift::visit(
                [this, &inspectionData, &interactionRateAccessor]
                <gsl::index IModel>(std::integral_constant<gsl::index, IModel>)
                {
                    this->inspectInteractionProperties<IModel>(inspectionData, interactionRateAccessor, true);
                },
                iModelV);
            return true;
        }
        else if (quantity == "interaction radii"sv)
        {
            if constexpr (haveLocality)
            {
                makeshift::visit(
                    [this, &inspectionData, &interactionDistanceAccessor]
                    <gsl::index IModel>(std::integral_constant<gsl::index, IModel>)
                    {
                        this->inspectInteractionProperties<IModel>(inspectionData, interactionDistanceAccessor);
                    },
                    iModelV);
                return true;
            }
        }
        else if (quantity == "acceptance probabilities"sv)
        {
            makeshift::visit(
                [this, &inspectionData, &interactionRateAccessor, &bucketInteractionRateAccessor]
                <gsl::index IModel>(std::integral_constant<gsl::index, IModel>)
                {
                    this->inspectAcceptanceProbabilities<IModel>(inspectionData, interactionRateAccessor, bucketInteractionRateAccessor);
                },
                iModelV);
            return true;
        }
        else if (quantity == "in-reach probabilities"sv)
        {
            if constexpr (haveLocality)
            {
                makeshift::visit(
                    [this, &inspectionData, &interactionDistanceAccessor, &bucketInteractionDistanceAccessor]
                    <gsl::index IModel>(std::integral_constant<gsl::index, IModel>)
                    {
                        this->inspectAcceptanceProbabilities<IModel>(inspectionData, interactionDistanceAccessor, bucketInteractionDistanceAccessor);
                    },
                    iModelV);
                return true;
            }
        }
        else if (quantity == "particle bucket interaction rates"sv)
        {
            inspectParticleBucketProperties(inspectionData, bucketInteractionRateAccessor);
            return true;
        }
        else if (quantity == "particle bucket interaction radii"sv)
        {
            if constexpr (haveLocality)
            {
                inspectParticleBucketProperties(inspectionData, bucketInteractionDistanceAccessor);
                return true;
            }
        }
        else if (quantity == "true particle bucket interaction rates"sv)
        {
            makeshift::visit(
                [this, &inspectionData, &interactionRateAccessor]
                <gsl::index IModel>(std::integral_constant<gsl::index, IModel>)
                {
                    this->inspectTrueParticleBucketProperties<IModel>(inspectionData, interactionRateAccessor, false);
                },
                iModelV);
            return true;
        }
        else if (quantity == "true particle bucket interaction rates, reverse"sv)
        {
            makeshift::visit(
                [this, &inspectionData, &interactionRateAccessor]
                <gsl::index IModel>(std::integral_constant<gsl::index, IModel>)
                {
                    this->inspectTrueParticleBucketProperties<IModel>(inspectionData, interactionRateAccessor, true);
                },
                iModelV);
            return true;
        }
        else if (quantity == "bucket interaction rates"sv)
        {
            inspectBucketProperties(inspectionData, bucketInteractionRateAccessor);
            return true;
        }
        else if (quantity == "bucket interaction radii"sv)
        {
            if constexpr (haveLocality)
            {
                inspectBucketProperties(inspectionData, bucketInteractionDistanceAccessor);
                return true;
            }
        }
        return false;
    }

public:
    RPMCOperator(
        RPMCOperatorParams const& params,
        PParticleData particleData,
        RandomNumberGeneratorT randomNumberGenerator,
        ClassifierT classifier,
        LocatorT locator,
        InteractionModelsT... interactionModels)
        : args_(params),
          classifier_(std::move(classifier)),
          locator_(std::move(locator)),
          interactionModels_{ std::move(interactionModels)... },
          particleData_(particleData),
          particleProperties_(particleData.num()),
          randomNumberGenerator_(std::move(randomNumberGenerator))
    {
    }

    void
    initialize()
    {
        std::apply(
            [](auto&&... interactionModels)
            {
                (interactionModels.initialize(), ...);
            },
            interactionModels_);

        recomputeAll();
    }

    double
    nextEventTime()
    {
        if (nextInteractionTimeIndex_ < 0)
        {
            nextInteractionTimes_ = makeshift::array_transform(
                [this]
                (double totalBucketInteractionRate)
                {
                    auto dist = std::uniform_real_distribution<double>{ };
                    return time_ + -1./totalBucketInteractionRate * std::log(1. - dist(randomNumberGenerator_));
                },
                totalBucketInteractionRates_);
            auto nextInteractionTimeIt = std::min_element(nextInteractionTimes_.begin(), nextInteractionTimes_.end());
            nextInteractionTimeIndex_ = nextInteractionTimeIt - nextInteractionTimes_.begin();
        }
        return nextInteractionTimes_[nextInteractionTimeIndex_];
    }

    gsl::index
    sampleModel()
    {
            // Determine the time of the next interaction and the interaction model that simulates it.
        gsl_Assert(nextInteractionTimeIndex_ >= 0);
        gsl::index iModel = std::exchange(nextInteractionTimeIndex_, -1);
        return iModel;
    }
    std::optional<std::tuple<Bucket, Bucket, double>>
    trySampleBuckets(gsl::index iModel)
    {
        auto buckets = bucketingScheme_.buckets();
        auto dist = std::uniform_real_distribution<double>{ };
        auto JRange = totalBucketInteractionRates_[iModel];
        auto JVal = dist(randomNumberGenerator_)*JRange;
        auto bucket1It = rpmc::discreteInverseTransform(std::ranges::begin(buckets), std::ranges::end(buckets), JVal,
            [iModel]
            (Bucket bucket1)
            {
                gsl_AssertDebug(bucket1.bucketData().n == bucket1.numEntries());
                return bucket1.bucketData().cumulativeInteractionRates[iModel];
            });
        if (bucket1It.pos == std::ranges::end(buckets))
        {
            return std::nullopt;
        }
        auto bucket1 = *bucket1It.pos;
        auto KRange = bucket1.bucketData().cumulativeInteractionRates[iModel];
        auto KVal = dist(randomNumberGenerator_)*KRange;
        decltype(bucket1It) bucket2It;
        bool haveBucket2 = false;
        if constexpr (haveLocality)
        {
            if (modelsHaveLocality[iModel])
            {
                haveBucket2 = true;
                bucket2It = rpmc::discreteInverseTransform(std::ranges::begin(buckets), std::ranges::end(buckets), KVal,
                    [
                        iModel,
                        iDistance = modelInteractionDistanceIndices[iModel],
                        bucket1,
                        &bucketingScheme = bucketingScheme_
                    ]
                    (Bucket bucket2)
                    {
                        auto& bucketBucketData_JK = bucketingScheme.bucketBucketData(bucket1, bucket2);
                        gsl_AssertDebug(bucket2.bucketData().n == bucket2.numEntries());
                        std::int64_t n1n2 = bucketBucketData_JK.n1n2[iDistance];
                        return bucketBucketData_JK.interactionRates[iModel]*n1n2;
                    });
            }
        }
        if (!haveBucket2)
        {
            bucket2It = rpmc::discreteInverseTransform(std::ranges::begin(buckets), std::ranges::end(buckets), KVal,
                [
                    iModel,
                    bucket1,
                    &bucketingScheme = bucketingScheme_
                ]
                (Bucket bucket2)
                {
                    auto& bucketBucketData_JK = bucketingScheme.bucketBucketData(bucket1, bucket2);
                    gsl_AssertDebug(bucket2.bucketData().n == bucket2.numEntries());
                    std::int64_t n1 = bucket1.bucketData().n;
                    std::int64_t n2 = bucket2.bucketData().n;
                    return bucketBucketData_JK.interactionRates[iModel]*(n1*n2);
                });
        }
        if (bucket2It.pos == std::ranges::end(buckets))
        {
            return std::nullopt;
        }
        return std::tuple{ *bucket1It.pos, *bucket2It.pos, bucket2It.residual };
    }
    template <gsl::index IModel>
    std::tuple<ConstSubBucket, ConstSubBucket>
    //sampleSubBuckets(gsl::index iModel, ConstBucket bucket1, ConstBucket bucket2)
    selectSubBuckets(
        std::integral_constant<gsl::index, IModel>, ConstBucket bucket1, ConstBucket bucket2,
        double rnd)  // ∈ [0,1)
    {
        if constexpr (modelsHaveLocality[IModel])
        {
            constexpr auto iDistance = modelInteractionDistanceIndices[IModel];
            auto& bucketBucketData = bucketingScheme_.bucketBucketData(bucket1, bucket2);
            auto n1n2 = bucketBucketData.n1n2[iDistance];
            gsl_Assert(n1n2 > 0);
            gsl_AssertDebug(n1n2 <= std::int64_t(bucket1.numEntries())*std::int64_t(bucket2.numEntries()));  // TODO: this can be assumed only because of the processing order of sub-bucket deltas!
            //auto n1n2StopDist = std::uniform_int_distribution<std::int64_t>(0, n1n2 - 1);
            //auto n1n2Stop = n1n2StopDist(randomNumberGenerator_);
            auto n1n2Stop = std::min(std::int64_t(rnd*n1n2), n1n2 - 1);
            return locateSubBuckets<IModel>(bucket1, bucket2, bucketBucketData.ds12s[iDistance], n1n2Stop);
        }
        else
        {
            auto n1n2 = bucket1.numEntries()*bucket2.numEntries();
            gsl_Assert(n1n2 > 0);
            //auto n1n2StopDist = std::uniform_int_distribution<std::int64_t>(0, n1n2 - 1);
            //auto n1n2Stop = n1n2StopDist(randomNumberGenerator_);
            auto n1n2Stop = std::min(std::int64_t(rnd*n1n2), n1n2 - 1);
            return locateSubBuckets(bucket1, bucket2, n1n2Stop);
        }
    }
    std::tuple<Index, Index>
    sampleParticles(std::span<Index const> particles1, std::span<Index const> particles2)
    {
        auto ejdist = std::uniform_int_distribution<gsl::index>(0, std::ssize(particles1) - 1);
        auto ij = ejdist(randomNumberGenerator_);
        auto j = particles1[ij];
        auto ekdist = std::uniform_int_distribution<gsl::index>(0, std::ssize(particles2) - 1);
        auto ik = ekdist(randomNumberGenerator_);
        auto k = particles2[ik];
        return { j, k };
    }

    std::span<gsl::index const>
    simulateNextEvent()
    {
        auto iModel = sampleModel();
        auto iModelV = makeshift::expand(iModel,
            MAKESHIFT_CONSTVAL(detail::array_iota<gsl::index, sizeof...(InteractionModelsT)>()));

            // Choose tracer bucket and swarm bucket by discrete inverse transform sampling.
        auto maybeBuckets = trySampleBuckets(iModel);
        if (!maybeBuckets.has_value())
        {
            ++numExcessSamplings_;
            return { };
        }
        auto [bucket1, bucket2, residual] = *maybeBuckets;

            // If locality is enabled, determine the sub-bucket of the active particle by discrete inverse transform sampling.
        std::span<Index const> particles1;
        std::span<Index const> particles2;
        if constexpr (haveLocality)
        {
            //auto [sb1, sb2] = sampleSubBuckets(iModel, bucket1, bucket2);
            auto [sb1, sb2] = makeshift::visit(
                [&](auto iModelC)
                {
                    return selectSubBuckets(iModelC, bucket1, bucket2, residual);
                },
                iModelV);
            gsl_AssertDebug(sb1.numEntries() == sb1.subBucketData().n && sb2.numEntries() == sb2.subBucketData().n);
            particles1 = sb1.entries();
            particles2 = sb2.entries();
        }
        else
        {
            particles1 = bucket1.entries();
            particles2 = bucket2.entries();
        }

            // To choose a particle pair from a pair of buckets by rejection sampling, first determine a pair of particles by
            // uniform sampling.
        auto [j, k] = sampleParticles(particles1, particles2);

            // Now compute interaction rate and factors.
        updatedParticleIndexList_.clear();
        bool wasAccepted = makeshift::visit(
            [
                this,
                &bucketBucketData = bucketingScheme_.bucketBucketData(bucket1, bucket2),
                j = j, k = k
            ]
            <gsl::index IModel>(std::integral_constant<gsl::index, IModel>)
            {
                constexpr gsl::index propertiesIndex = modelToPropertiesIndexMap[IModel];

                auto& interactionModel = std::get<IModel>(interactionModels_);

                    // Fast reject: if locality is supported, check whether the two particles are possibly in reach using the
                    // bucket–bucket interaction radius upper bound, and avoid calculating the true interaction rate if the
                    // particles are out of reach.
                [[maybe_unused]] double dr12;
                if constexpr (modelsHaveLocality[IModel])
                {
                    FastDs ds12 = ds12Of<IModel>(bucketBucketData);
                    dr12 = locator_.location(j) - locator_.location(k);
                    if (dr12*dr12 > ds12*ds12)
                    {
                        ++numBucketsOutOfReach_;
                        ++numOutOfReach_;
                        return false;
                    }
                }

                    // Compute true interaction rate.
                auto& p1 = particleProperties_[j];
                auto& p2 = particleProperties_[k];
                auto& ip1 = std::get<propertiesIndex>(p1);
                auto& ip2 = std::get<propertiesIndex>(p2);
                if (!detail::hasValue(ip1) || !detail::hasValue(ip2))
                {
                        // We have particle properties for the bucket, but the individual particle's properties are `nullopt`,
                        // and thus have not contributed to the bucket properties.
                        // This can happen, though it tends to be a sign of a suboptimal bucketing scheme.
                    return false;
                }
                auto& ip1v = detail::getValue(ip1);
                auto& ip2v = detail::getValue(ip2);
                auto isSelfInteraction = IsSelfInteraction{ j == k };
                auto interactionData = interactionModel.template computeTracerSwarmInteractionData<std::type_identity_t>(
                    ip1v, ip2v, isSelfInteraction);
                gsl_Assert(interactionData.interactionRate_jk >= 0);
                gsl_Assert(interactionData.interactionRate_kj >= 0);

                    // Slow reject: if locality is supported, check whether the two particles are in reach using the
                    // mutual interaction radius.
                if constexpr (modelsHaveLocality[IModel])
                {
                    auto d12 = locator_.interactionWidthToDistance(interactionData.interactionWidth);
                    if (dr12*dr12 > d12*d12)
                    {
                        //gsl_Assert(interactionData.interactionRate_jk == 0);
                        ++numOutOfReach_;
                        return false;
                    }
                }

                    // Perform rejection sampling.
                double interactionRate = interactionData.interactionRate_jk;
                double bucketBucketInteractionRateUpperBound = bucketBucketData.interactionRates[IModel];
                gsl_Assert(bucketBucketInteractionRateUpperBound > 0);
                auto pAccept = interactionRate/bucketBucketInteractionRateUpperBound;
                auto dist = std::uniform_real_distribution<double>{ };
                bool accept = pAccept > 0 && dist(randomNumberGenerator_) <= pAccept;
                if (accept)
                {
                        // If sampling is accepted, carry out the interaction.
                    interactionModel.interact(
                        CollideCallback(this),
                        randomNumberGenerator_,
                        j, k,
                        ip1v, ip2v,
                        interactionData);
                }
                return accept;
            },
            iModelV);
        if (wasAccepted)
        {
            registerEventForProfiling();
        }
        else
        {
            registerRejectionForProfiling();

                // Make sure to recompute the buckets sometime if we keep rejecting events too often.
            auto& bucketData1 = bucket1.bucketData();
            --bucketData1.updatesBeforeRecomputation2;
            if (bucketData1.updatesBeforeRecomputation2 <= 0)
            {
                if (bucketData1.touched)
                {
                    updatedParticleIndexList_.push_back(j);
                }
                else
                {
                    bucketData1.updatesBeforeRecomputation2 = 0;
                }
            }
            if (k != j)
            {
                auto& bucketData2 = bucket2.bucketData();
                --bucketData2.updatesBeforeRecomputation2;
                if (bucketData2.updatesBeforeRecomputation2 <= 0)
                {
                    if (bucketData2.touched)
                    {
                        updatedParticleIndexList_.push_back(k);
                    }
                    else
                    {
                        bucketData2.updatesBeforeRecomputation2 = 0;
                    }
                }
            }
        }
        return updatedParticleIndexList_;
    }

    void
    invalidate(std::span<gsl::index const> updatedParticleIndices)
    {
        updateParticles(updatedParticleIndices);

        nextInteractionTimeIndex_ = -1;
    }

    void
    invalidateAll()
    {
        recomputeAll();

        nextInteractionTimeIndex_ = -1;
    }

    void
    synchronize()
    {
        std::apply(
            [](auto&... interactionModels)
            {
                (interactionModels.synchronize(), ...);
            },
            interactionModels_);
    }

    void
    advanceTo(double tEnd)
    {
        gsl_Expects(tEnd >= time_);

        time_ = tEnd;
    }

    void
    tidyUp()
    {
    }

    bool
    inspect(std::string_view quantity, InspectionData const& inspectionData) const
    {
        using namespace std::literals;

        if (quantity == "statistics"sv)
        {
            std::cout <<
                "Statistics:\n"
                "===========\n"
                "    simulation-time: " << time_ << "\n"
                "    num-events: " << numEvents_ << "\n"
                "    num-rejections: " << numRejections_ << "\n"
                "    num-excess-samplings: " << numExcessSamplings_ << "\n"
                "    num-buckets-out-of-reach: " << numBucketsOutOfReach_ << "\n"
                "    num-out-of-reach: " << numOutOfReach_ << "\n"
                "    buckets-out-of-reach-probability: " << (numRejections_ != 0 ? double(numBucketsOutOfReach_)/numRejections_ : 0.) << "\n"
                "    out-of-reach-probability: " << (numRejections_ != 0 ? double(numOutOfReach_)/numRejections_ : 0.) << "\n"
                "    avg-acceptance-probability: " << double(numEvents_)/(numEvents_  + numRejections_) << "\n"
                "    num-updates: " << numUpdates_ << "\n"
                "    num-bucket-updates: " << numBucketUpdates_ << "\n"
                "    num-recomputes: " << numBucketRecomputes_ << "\n"
                "    num-bucket-changes: " << numBucketChanges_ << "\n"
                "    recompute-probability: " << (numUpdates_ != 0 ? double(numBucketRecomputes_)/numUpdates_ : 0.) << "\n"
                "    bucket-update-probability: " << (numUpdates_ != 0 ? double(numBucketUpdates_)/numUpdates_ : 0.) << "\n"
                "    bucket-change-probability: " << (numUpdates_ != 0 ? double(numBucketChanges_)/numUpdates_ : 0.) << "\n"
                "    min-num-buckets: " << minNumBuckets_ << "\n"
                "    max-num-buckets: " << maxNumBuckets_ << "\n"
                "    avg-num-buckets: " << (numEvents_ != 0 ? double(numBucketEvents_)/numEvents_ : 0.) << "\n"
                "    min-num-active-particles: " << minNumActiveParticles_ << "\n"
                "    max-num-active-particles: " << maxNumActiveParticles_ << "\n"
                "    avg-num-active-particles: " << (numEvents_ != 0 ? double(numActiveParticleEvents_)/numEvents_ : 0.) << "\n\n";
            return true;
        }
        else if (quantity == "num-events"sv)
        {
            gsl_Assert(std::ssize(inspectionData.idst) == 1);
            inspectionData.idst[0] = numEvents_;
            return true;
        }
        else if (quantity == "num-rejections"sv)
        {
            gsl_Assert(std::ssize(inspectionData.idst) == 1);
            inspectionData.idst[0] = numRejections_;
            return true;
        }
        else if (quantity == "num-updates"sv)
        {
            gsl_Assert(std::ssize(inspectionData.idst) == 1);
            inspectionData.idst[0] = numUpdates_;
            return true;
        }
        else if (quantity == "num-bucket-updates"sv)
        {
            gsl_Assert(std::ssize(inspectionData.idst) == 1);
            inspectionData.idst[0] = numBucketUpdates_;
            return true;
        }
        else if (quantity == "num-bucket-recomputes"sv)
        {
            gsl_Assert(std::ssize(inspectionData.idst) == 1);
            inspectionData.idst[0] = numBucketRecomputes_;
            return true;
        }
        else if (quantity == "num-bucket-changes"sv)
        {
            gsl_Assert(std::ssize(inspectionData.idst) == 1);
            inspectionData.idst[0] = numBucketChanges_;
            return true;
        }
        else if (quantity == "num-buckets"sv)
        {
            gsl_Assert(std::ssize(inspectionData.idst) >= 1 && std::ssize(inspectionData.idst) <= 4);
            inspectionData.idst[0] = std::ssize(bucketingScheme_.buckets());
            if (std::ssize(inspectionData.idst) >= 2)
            {
                inspectionData.idst[1] = minNumBuckets_;
            }
            if (std::ssize(inspectionData.idst) >= 3)
            {
                inspectionData.idst[2] = maxNumBuckets_;
            }
            if (std::ssize(inspectionData.idst) >= 4)
            {
                inspectionData.idst[3] = numBucketEvents_;
            }
            return true;
        }
        else if (quantity == "num-active-particles"sv)
        {
            gsl_Assert(std::ssize(inspectionData.idst) >= 1 && std::ssize(inspectionData.idst) <= 4);
            gsl::dim numActiveParticles = getNumActiveParticles();
            inspectionData.idst[0] = numActiveParticles;
            if (std::ssize(inspectionData.idst) >= 2)
            {
                inspectionData.idst[1] = minNumActiveParticles_;
            }
            if (std::ssize(inspectionData.idst) >= 3)
            {
                inspectionData.idst[2] = maxNumActiveParticles_;
            }
            if (std::ssize(inspectionData.idst) >= 4)
            {
                inspectionData.idst[3] = numActiveParticleEvents_;
            }
            return true;
        }
        else if (quantity == "indices"sv)
        {
            gsl::dim numActiveParticles = getNumActiveParticles();
            gsl_Assert(std::ssize(inspectionData.idst) == numActiveParticles);
        
            gsl::index i = 0;
            for (auto bucket : bucketingScheme_.buckets())
            {
                for (gsl::index entryIndex : bucket.entries())
                {
                    inspectionData.idst[i] = entryIndex;
                    ++i;
                }
            }
            return true;
        }
        else if (quantity == "bucket-indices"sv)
        {
            gsl::dim numActiveParticles = getNumActiveParticles();
            gsl_Assert(std::ssize(inspectionData.idst) == numActiveParticles);
        
            gsl::index i = 0;
            gsl::index iBucket = 0;
            for (auto bucket : bucketingScheme_.buckets())
            {
                for ([[maybe_unused]] gsl::index _ : bucket.entries())
                {
                    inspectionData.idst[i] = iBucket;
                    ++i;
                }
                ++iBucket;
            }
            return true;
        }
        else if (quantity == "bucket-particle-counts"sv)
        {
            gsl_Assert(std::ssize(inspectionData.idst) == std::ssize(bucketingScheme_.buckets()));
        
            gsl::index i = 0;
            for (auto bucket : bucketingScheme_.buckets())
            {
                inspectionData.idst[i] = bucket.numEntries();
                ++i;
            }
            return true;
        }
        else
        {
            for (gsl::index iModel = 0; iModel != gsl::dim(sizeof...(InteractionModelsT)); ++iModel)
            {
                std::string prefix = fmt::format("interaction-model-{}/", iModel);
                if (quantity.starts_with(prefix))
                {
                    bool handled = inspectInteractionModel(quantity.substr(prefix.size()), inspectionData, iModel);
                    if (handled)
                    {
                        return true;
                    }
                }
            }
        }
        return false;
    }
};
template <typename RandomNumberGeneratorT, typename ClassifierT, typename LocatorT, typename... InteractionModelsT>
RPMCOperator(RPMCOperatorParams, PParticleData, RandomNumberGeneratorT, ClassifierT, LocatorT, InteractionModelsT...) -> RPMCOperator<RandomNumberGeneratorT, ClassifierT, LocatorT, InteractionModelsT...>;


} // namespace rpmc


#endif // INCLUDED_RPMC_OPERATORS_RPMC_HPP_
