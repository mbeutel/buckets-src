
// Extended Representative Particle Monte Carlo method, implemented in the traditional qway.


#ifndef INCLUDED_RPMC_OPERATORS_RPMCTRADITIONAL_HPP_
#define INCLUDED_RPMC_OPERATORS_RPMCTRADITIONAL_HPP_


#include <span>
#include <cmath>
#include <array>
#include <tuple>
#include <vector>
#include <random>      // for uniform_real_distribution<>
#include <compare>
#include <variant>
#include <cassert>
#include <utility>     // for move(), swap()
#include <cstddef>     // for size_t
#include <cstdint>     // for uint8_t, int64_t
#include <numeric>     // for accumulate(), transform_reduce()
#include <optional>
#include <iostream>
#include <algorithm>   // for for_each(), min(), max(), reverse()

#include <gsl-lite/gsl-lite.hpp>  // for dim, index, ssize(), gsl_Expects(), gsl_Assert(), gsl_FailFast(), narrow_cast<>(), narrow_failfast<>()

#include <fmt/core.h>  // for format()

#include <makeshift/array.hpp>        // for mdarray<>, array_fill<>(), array_transform()
#include <makeshift/tuple.hpp>        // for tie_members(), apply(), template_for(), template_any_of()
#include <makeshift/variant.hpp>      // for expand(), visit(), variant_transform()
#include <makeshift/iomanip.hpp>      // for as_enum()
#include <makeshift/metadata.hpp>     // for value_names<>(), is_available()
#include <makeshift/type_traits.hpp>  // for nth_type<>

#include <intervals/math.hpp>

#include <rpmc/tools/soa.hpp>                    // for gatherFromSoA()
#include <rpmc/tools/math.hpp>                   // for discreteInverseTransform()
#include <rpmc/tools/utility.hpp>                // for square_checked_failfast(), KahanAccumulator<>
#include <rpmc/tools/particles.hpp>              // for PParticleData

#include <rpmc/operators/operator.hpp>     // for InspectionData, NoOpDiscreteOperator
#include <rpmc/operators/rpmc/common.hpp>  // for InteractionRates<>, IsSelfInteraction

#include <rpmc/detail/rpmc.hpp>



namespace rpmc {

namespace gsl = gsl_lite;


struct RPMCTraditionalOperatorArgs
{
};
template <typename RandomNumberGeneratorT, typename ClassifierT, typename... InteractionModelsT>
class RPMCTraditionalOperator : NoOpDiscreteOperator
{
    static_assert(sizeof...(InteractionModelsT) > 0);

    template <gsl::index I, template <typename> class TT = std::type_identity_t>
    using ParticleStateForModel = typename detail::Rebind<decltype(detail::gatherFromSoA(
        std::declval<typename makeshift::nth_type_t<I, InteractionModelsT...> const>().getState(),
        gsl::index(0))), TT>::type;
    template <gsl::index I, template <typename> class TT = std::type_identity_t>
    using ParticlePropertiesForModel = typename detail::Rebind<decltype(getParticleProperties(
        std::declval<typename makeshift::nth_type_t<I, InteractionModelsT...> const>().getArgs(),
        std::declval<ParticleStateForModel<I, TT> const&>())), TT>::type;
    template <template <typename> class TT = std::type_identity_t>
    using ModelParticleProperties = typename detail::ParticlePropertiesTuple_<std::index_sequence_for<InteractionModelsT...>, ParticlePropertiesForModel, TT>::type;
    template <template <typename> class TT = std::type_identity_t>
    using ParticleProperties = makeshift::unique_sequence_t<ModelParticleProperties<TT>>;
    static constexpr gsl::dim numParticleProperties = std::tuple_size_v<ParticleProperties<>>;
    static constexpr std::array modelToPropertiesIndexMap = detail::firstTupleElementIndices<ModelParticleProperties<>, ParticleProperties<>>();
    static constexpr std::array propertiesToModelIndexMap = detail::firstTupleElementIndices<ParticleProperties<>, ModelParticleProperties<>>();

    static constexpr double inf = std::numeric_limits<double>::infinity();

    using InteractionRates = std::array<double, sizeof...(InteractionModelsT)>;
    //using Accumulator = KahanAccumulator<double>;
    using Accumulator = double;
    using Accumulators = std::array<Accumulator, sizeof...(InteractionModelsT)>;

    struct InteractionData
    {
        std::array<double, sizeof...(InteractionModelsT)> nextInteractionTimes = makeshift::array_fill<sizeof...(InteractionModelsT)>(-inf);
        gsl::index nextInteractionTimeIndex = -1;

        std::vector<ParticleProperties<>> particleProperties;

        std::vector<gsl::index> inactiveParticleListIndex;  // index lookup table for "castling" maneuver
        std::vector<gsl::index> inactiveParticleList;
        gsl::index succOfLastActiveParticleIndex;

        std::vector<InteractionRates> interactionRates;         // (time⁻¹)
        std::vector<Accumulators> cumulativeInteractionRates;   // (time⁻¹)
        Accumulators totalInteractionRates;                     // (time⁻¹)

        explicit InteractionData(gsl::dim n)
            : particleProperties(n),
              inactiveParticleListIndex(n, -1),
              succOfLastActiveParticleIndex(n),
              interactionRates(rpmc::square_checked_failfast(n)),
              cumulativeInteractionRates(n)
        {
        }
    };

    class CollideCallback
    {
        friend RPMCTraditionalOperator;

    private:
        RPMCTraditionalOperator* self_;
        std::span<gsl::index const> inactiveEntries_;

        explicit CollideCallback(RPMCTraditionalOperator* _self)
            : self_(_self),
              inactiveEntries_(_self->interactionData_.inactiveParticleList)
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

private:
    PParticleData particleData_;
    RandomNumberGeneratorT randomNumberGenerator_;
    ClassifierT classifier_;
    std::tuple<InteractionModelsT...> interactionModels_;
    double time_;
    InteractionData interactionData_;
    std::vector<gsl::index> updatedParticleIndexList_;

        // Profiling data
    std::int64_t numEvents_ = 0;
    std::int64_t numExcessSamplings_ = 0;
    gsl::dim minNumActiveParticles_ = std::numeric_limits<gsl::dim>::max();
    gsl::dim maxNumActiveParticles_ = 0;
    std::int64_t numActiveParticleEvents_ = 0;


    bool
    isEnlistedAsActive(gsl::index j) const
    {
        return interactionData_.inactiveParticleListIndex[j] < 0;
    }


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

    void
    registerEventForProfiling()
    {
        auto numActiveParticles = particleData_.num() - std::ssize(interactionData_.inactiveParticleList);

        minNumActiveParticles_ = std::min(minNumActiveParticles_, numActiveParticles);
        maxNumActiveParticles_ = std::max(maxNumActiveParticles_, numActiveParticles);
        numActiveParticleEvents_ += numActiveParticles;
        ++numEvents_;
    }

    void
    recomputeInteractionRates()
    {
        gsl::dim n = particleData_.num();

            // Recompute particle properties.
        auto newInactiveParticleList = std::vector<gsl::index>{ };
        gsl::index maxActiveParticleIndex = -1;
        for (gsl::index j = 0; j != n; ++j)
        {
            bool isActive = classifier_.isActive(j);
            if (isActive)
            {
                interactionData_.particleProperties[j] = getAllParticleProperties(j);
                maxActiveParticleIndex = j;
            }
            else
            {
                newInactiveParticleList.push_back(j);
            }
        }

            // Store list of inactive particles in reverse order and keep track of index of max active particle to optimize
            // for overallocation.
        std::reverse(newInactiveParticleList.begin(), newInactiveParticleList.end());
        interactionData_.inactiveParticleListIndex.assign(interactionData_.inactiveParticleListIndex.size(), -1);
        for (gsl::index ii = 0, in = gsl::ssize(newInactiveParticleList); ii != in; ++ii)
        {
            gsl::index i = newInactiveParticleList[ii];
            interactionData_.inactiveParticleListIndex[i] = ii;
        }
        interactionData_.inactiveParticleList = std::move(newInactiveParticleList);
        interactionData_.succOfLastActiveParticleIndex = maxActiveParticleIndex + 1;

            // Recompute tracer–swarm interaction rates.
        for (gsl::index j = 0, nsub = interactionData_.succOfLastActiveParticleIndex; j != nsub; ++j)
        {
            auto const& p1 = interactionData_.particleProperties[j];
            bool p1Active = isEnlistedAsActive(j);

            auto cumulativeInteractionRates_j = Accumulators{ };
            auto newInteractionRates_jj = InteractionRates{ };
            if (p1Active)
            {
                makeshift::template_for(
                    [&p1]
                    (auto modelIndexC, auto const& interactionModel,
                     double& interactionRate_jj,
                     Accumulator& cumulativeInteractionRate_j)
                    {
                        constexpr gsl::index propertiesIndex = modelToPropertiesIndexMap[modelIndexC];
                        auto& ip1 = std::get<propertiesIndex>(p1);
                        if (detail::hasValue(ip1))
                        {
                            auto interactionData = interactionModel.template computeTracerSwarmInteractionData<std::type_identity_t>(
                                detail::getValue(ip1), detail::getValue(ip1), IsSelfInteraction::yes);
                            interactionRate_jj = interactionData.interactionRate_jk;
                            cumulativeInteractionRate_j = interactionData.interactionRate_jk;
                        }
                    },
                    makeshift::tuple_index, interactionModels_,
                    newInteractionRates_jj,
                    cumulativeInteractionRates_j);
            }
            interactionData_.interactionRates[j*n + j] = newInteractionRates_jj;

                // We iterate over the lower half of the interaction matrix to ensure the cumulative additions to
                // `cumulativeInteractionRates_k` aren't overwritten.
            for (gsl::index k = 0; k < j; ++k)
            {
                auto const& p2 = interactionData_.particleProperties[k];
                bool p2Active = isEnlistedAsActive(k);

                auto newInteractionRates_jk = InteractionRates{ };
                auto newInteractionRates_kj = InteractionRates{ };
                if (p1Active && p2Active)
                {
                    makeshift::template_for(
                        [&p1, &p2]
                        (auto modelIndexC, auto const& interactionModel,
                         double& newInteractionRate_jk, double& newInteractionRate_kj,
                         Accumulator& cumulativeInteractionRate_j, Accumulator& cumulativeInteractionRate_k)
                        {
                            constexpr gsl::index propertiesIndex = modelToPropertiesIndexMap[modelIndexC];
                            auto& ip1 = std::get<propertiesIndex>(p1);
                            auto& ip2 = std::get<propertiesIndex>(p2);
                            if (detail::hasValue(ip1) && detail::hasValue(ip2))
                            {
                                auto interactionData = interactionModel.template computeTracerSwarmInteractionData<std::type_identity_t>(
                                    detail::getValue(ip1), detail::getValue(ip2), IsSelfInteraction::no);
                                newInteractionRate_jk = interactionData.interactionRate_jk;
                                newInteractionRate_kj = interactionData.interactionRate_kj;
                                cumulativeInteractionRate_j += interactionData.interactionRate_jk;
                                cumulativeInteractionRate_k += interactionData.interactionRate_kj;
                            }
                        },
                        makeshift::tuple_index, interactionModels_,
                        newInteractionRates_jk, newInteractionRates_kj,
                        cumulativeInteractionRates_j, interactionData_.cumulativeInteractionRates[k]);
                }
                interactionData_.interactionRates[k*n + j] = newInteractionRates_kj;
                interactionData_.interactionRates[j*n + k] = newInteractionRates_jk;
            }
            interactionData_.cumulativeInteractionRates[j] = cumulativeInteractionRates_j;
        }

            // Recompute total interaction rates.
        auto totalInteractionRates = Accumulators{ };
        for (gsl::index j = 0, nsub = interactionData_.succOfLastActiveParticleIndex; j != nsub; ++j)
        {
            auto const& cumulativeInteractionRates_j = interactionData_.cumulativeInteractionRates[j];
            makeshift::template_for(
                [](Accumulator& lhs, Accumulator const& rhs)
                {
                    lhs += rhs;
                },
                totalInteractionRates, cumulativeInteractionRates_j);
        }
        interactionData_.totalInteractionRates = totalInteractionRates;
    }

    void
    updateInteractionRatesForParticle(
        gsl::index j, bool isActive)
    {
        gsl::dim n = particleData_.num();
        auto const& p1 = interactionData_.particleProperties[j];
        bool p1Active = isActive;

        auto newInteractionRates_jj = InteractionRates{ };
        auto cumulativeInteractionRates_j = Accumulators{ };
        if (p1Active)
        {
            makeshift::template_for(
                [&p1]
                (auto modelIndexC, auto const& interactionModel,
                 double& newInteractionRate_jj,
                 Accumulator& cumulativeInteractionRate_j)
                {
                    constexpr gsl::index propertiesIndex = modelToPropertiesIndexMap[modelIndexC];
                    auto& ip1 = std::get<propertiesIndex>(p1);
                    if (detail::hasValue(ip1))
                    {
                        auto interactionData = interactionModel.template computeTracerSwarmInteractionData<std::type_identity_t>(
                            detail::getValue(ip1), detail::getValue(ip1), IsSelfInteraction::yes);
                        newInteractionRate_jj = interactionData.interactionRate_jk;
                        cumulativeInteractionRate_j = interactionData.interactionRate_jk;
                    }
                },
                makeshift::tuple_index, interactionModels_,
                newInteractionRates_jj,
                cumulativeInteractionRates_j);
        }
        interactionData_.interactionRates[j*n + j] = newInteractionRates_jj;

        auto totalInteractionRatesDelta = Accumulators{ };
        for (gsl::index k = 0, nsub = interactionData_.succOfLastActiveParticleIndex; k != nsub; ++k)
        {
            if (k == j) continue;

            auto const& p2 = interactionData_.particleProperties[k];
            bool p2Active = isEnlistedAsActive(k);
            makeshift::template_for(
                [p1Active, p2Active, &p1, &p2]
                (auto modelIndexC, auto const& interactionModel,
                 double& interactionRate_jk, double& interactionRate_kj,
                 Accumulator& cumulativeInteractionRate_j, Accumulator& cumulativeInteractionRate_k,
                 Accumulator& totalInteractionRateDelta)
                {
                    using namespace intervals::math;

                    auto cumulativeInteractionRateDelta_k = Accumulator{ -interactionRate_kj };
                    totalInteractionRateDelta -= interactionRate_kj;

                    interactionRate_jk = 0;
                    interactionRate_kj = 0;

                    if (p1Active && p2Active)
                    {
                        constexpr gsl::index propertiesIndex = modelToPropertiesIndexMap[modelIndexC];
                        auto& ip1 = std::get<propertiesIndex>(p1);
                        auto& ip2 = std::get<propertiesIndex>(p2);
                        if (detail::hasValue(ip1) && detail::hasValue(ip2))
                        {
                            auto interactionData = interactionModel.template computeTracerSwarmInteractionData<std::type_identity_t>(
                                detail::getValue(ip1), detail::getValue(ip2), IsSelfInteraction::no);
                            interactionRate_jk = interactionData.interactionRate_jk;
                            interactionRate_kj = interactionData.interactionRate_kj;
                            cumulativeInteractionRate_j += interactionData.interactionRate_jk;
                            cumulativeInteractionRateDelta_k += interactionData.interactionRate_kj;
                            totalInteractionRateDelta += interactionData.interactionRate_kj;
                        }
                    }

                    cumulativeInteractionRate_k = max(0., cumulativeInteractionRate_k + cumulativeInteractionRateDelta_k);
                },
                makeshift::tuple_index, interactionModels_,
                interactionData_.interactionRates[j*n + k], interactionData_.interactionRates[k*n + j],
                cumulativeInteractionRates_j, interactionData_.cumulativeInteractionRates[k],
                totalInteractionRatesDelta);
        }
        makeshift::template_for(
            [](Accumulator& totalInteractionRate,
               Accumulator& totalInteractionRateDelta,
               Accumulator& cumulativeInteractionRates, Accumulator const& cumulativeInteractionRatesNew)
            {
                using namespace intervals::math;

                totalInteractionRateDelta -= cumulativeInteractionRates;
                totalInteractionRateDelta += cumulativeInteractionRatesNew;
                totalInteractionRate = max(0., totalInteractionRate + totalInteractionRateDelta);
                cumulativeInteractionRates = cumulativeInteractionRatesNew;
            },
            interactionData_.totalInteractionRates,
            totalInteractionRatesDelta,
            interactionData_.cumulativeInteractionRates[j], cumulativeInteractionRates_j);
    }
    void
    updateInactiveParticleListForParticle(
        gsl::index j, bool wasActive, bool isActive)
    {
        if (wasActive && !isActive)
        {
                // Add particle to list of inactive entries.
            interactionData_.inactiveParticleList.push_back(j);
            interactionData_.inactiveParticleListIndex[j] = std::ssize(interactionData_.inactiveParticleList) - 1;

                // If this was the last active particle covered, update the index of the successor of the last active particle.
            if (interactionData_.succOfLastActiveParticleIndex == j + 1)
            {
                while (interactionData_.succOfLastActiveParticleIndex > 0
                    && !isEnlistedAsActive(interactionData_.succOfLastActiveParticleIndex - 1))
                {
                    --interactionData_.succOfLastActiveParticleIndex;
                }
            }
        }
        else if (isActive && !wasActive)
        {
                // Remove from list of inactive entries.
                // If the entry isn't at the end of the list, use a "castling" maneuver to remove it in  𝒪(1)  steps.
            gsl::index oldEntryIndex = interactionData_.inactiveParticleListIndex[j];
            interactionData_.inactiveParticleListIndex[j] = -1;
            gsl::dim numInactiveParticles = std::ssize(interactionData_.inactiveParticleList);
            if (oldEntryIndex != numInactiveParticles - 1)
            {
                auto& swapEntryIndex = interactionData_.inactiveParticleList[oldEntryIndex];
                std::swap(swapEntryIndex, interactionData_.inactiveParticleList.back());
                gsl_Assert(interactionData_.inactiveParticleListIndex[swapEntryIndex] == numInactiveParticles - 1);
                interactionData_.inactiveParticleListIndex[swapEntryIndex] = oldEntryIndex;
            }
            interactionData_.inactiveParticleList.pop_back();

                // Make sure the index of the successor of the last active particle reaches beyond this one.
            interactionData_.succOfLastActiveParticleIndex = std::max(interactionData_.succOfLastActiveParticleIndex, j + 1);
        }
    }

    void
    updateInteractionData(std::span<gsl::index const> updatedParticleIndices)
    {
            // Update particle properties and interaction rates.
        for (gsl::index j : updatedParticleIndices)
        {
            gsl_Expects(j >= 0 && j < particleData_.num());

            bool wasActive = isEnlistedAsActive(j);
            bool isActive = classifier_.isActive(j);
            if (isActive)
            {
                interactionData_.particleProperties[j] = getAllParticleProperties(j);
            }
            updateInteractionRatesForParticle(j, isActive);
            updateInactiveParticleListForParticle(j, wasActive, isActive);
        }
    }

    bool
    inspectInteractionModel(std::string_view quantity, InspectionData const& inspectionData, gsl::index I) const
    {
        using namespace std::literals;

        if (quantity == "interaction rates"sv)
        {
            gsl::dim n = particleData_.num();
            auto numActiveParticles = particleData_.num() - std::ssize(interactionData_.inactiveParticleList);
            gsl_Assert(gsl::ssize(inspectionData.fdst) == rpmc::square_checked_failfast(numActiveParticles));

            gsl::dim iRow = 0;
            for (gsl::index j = 0, nsub = interactionData_.succOfLastActiveParticleIndex; j != nsub; ++j)
            {
                if (!isEnlistedAsActive(j)) continue;
                gsl::dim iCol = 0;
                for (gsl::index k = 0; k != nsub; ++k)
                {
                    if (!isEnlistedAsActive(k)) continue;
                    inspectionData.fdst[iRow*n + iCol] = interactionData_.interactionRates[j*n + k][I];
                    ++iCol;
                }
                gsl_Assert(iCol == numActiveParticles);
                ++iRow;
            }
            gsl_Assert(iRow == numActiveParticles);
            return true;
        }
        return false;
    }

public:
    RPMCTraditionalOperator(
        RPMCTraditionalOperatorArgs /*args*/,
        PParticleData particleData,
        RandomNumberGeneratorT randomNumberGenerator,
        ClassifierT classifier,
        InteractionModelsT... interactionModels)
        : particleData_(particleData),
          randomNumberGenerator_(std::move(randomNumberGenerator)),
          classifier_(std::move(classifier)),
          interactionModels_{ std::move(interactionModels)... },
          time_{ 0. },
          interactionData_(particleData.num())
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

        recomputeInteractionRates();
    }

    double
    nextEventTime()
    {
        if (interactionData_.nextInteractionTimeIndex < 0)
        {
            interactionData_.nextInteractionTimes = makeshift::array_transform(
                [this]
                (Accumulator const& totalInteractionRate)
                {
                    auto dist = std::uniform_real_distribution<double>{ };
                    return time_ + -1./static_cast<double>(totalInteractionRate)*std::log(1. - dist(randomNumberGenerator_));
                },
                interactionData_.totalInteractionRates);
            interactionData_.nextInteractionTimeIndex = std::min_element(interactionData_.nextInteractionTimes.begin(), interactionData_.nextInteractionTimes.end())
                - interactionData_.nextInteractionTimes.begin();
        }
        return interactionData_.nextInteractionTimes[interactionData_.nextInteractionTimeIndex];
    }

    std::span<gsl::index const>
    simulateNextEvent()
    {
            // Determine the time of the next interaction and the interaction model that simulates it.
        gsl_Assert(interactionData_.nextInteractionTimeIndex >= 0);
        gsl::index nextInteractionIndex = std::exchange(interactionData_.nextInteractionTimeIndex, -1);
        //double nextInteractionTime = interactionData_.nextInteractionTimes[nextInteractionIndex];
        auto interactionModelIndexV = makeshift::expand(nextInteractionIndex,
            MAKESHIFT_CONSTVAL(detail::array_iota<std::size_t, sizeof...(InteractionModelsT)>()));

            // Choose tracer and swarm by discrete inverse transform sampling.
        auto dist = std::uniform_real_distribution<double>{ };
        auto jVal = dist(randomNumberGenerator_)*static_cast<double>(interactionData_.totalInteractionRates[nextInteractionIndex]);
        auto jIt = rpmc::discreteInverseTransform(interactionData_.cumulativeInteractionRates.begin(), interactionData_.cumulativeInteractionRates.end(),
            jVal,
            [nextInteractionIndex]
            (auto const& cumulativeInteractionRates)
            {
                return static_cast<double>(cumulativeInteractionRates[nextInteractionIndex]);
            });
        if (jIt.pos == interactionData_.cumulativeInteractionRates.end())
        {
            ++numExcessSamplings_;
            return { };
        }
        auto j = jIt.pos - interactionData_.cumulativeInteractionRates.begin();
        gsl::dim n = particleData_.num();
        auto kVal = dist(randomNumberGenerator_)*static_cast<double>((*jIt.pos)[nextInteractionIndex]);
        auto kFirst = interactionData_.interactionRates.begin() + j*n;
        auto kLast = kFirst + n;
        auto kIt = rpmc::discreteInverseTransform(kFirst, kLast,
            kVal,
            [nextInteractionIndex]
            (auto const& interactionRates)
            {
                return interactionRates[nextInteractionIndex];
            });
        if (kIt.pos == kLast)
        {
            ++numExcessSamplings_;
            return { };
        }
        auto k = kIt.pos - kFirst;

            // Carry out the interaction.
        updatedParticleIndexList_.clear();
        registerEventForProfiling();
        makeshift::visit(
            [this, j, k]
            <std::size_t I>(std::integral_constant<std::size_t, I>)
            {
                auto& p1 = interactionData_.particleProperties[j];
                auto& p2 = interactionData_.particleProperties[k];
                constexpr gsl::index propertiesIndex = modelToPropertiesIndexMap[I];
                auto& ip1 = std::get<propertiesIndex>(p1);
                auto& ip2 = std::get<propertiesIndex>(p2);
                gsl_Assert(detail::hasValue(ip1) && detail::hasValue(ip2));
                auto& ip1v = detail::getValue(ip1);
                auto& ip2v = detail::getValue(ip2);
                auto isSelfInteraction = IsSelfInteraction{ j == k };
                auto& interactionModel = std::get<I>(interactionModels_);
                auto interactionData = interactionModel.template computeTracerSwarmInteractionData<std::type_identity_t>(
                    ip1v, ip2v, isSelfInteraction);

                interactionModel.interact(
                    CollideCallback(this),
                    randomNumberGenerator_,
                    j, k,
                    ip1v, ip2v,
                    interactionData);
            },
            interactionModelIndexV);

        return updatedParticleIndexList_;
    }

    void
    invalidate(std::span<gsl::index const> updatedParticleIndices)
    {
        updateInteractionData(updatedParticleIndices);
        interactionData_.nextInteractionTimeIndex = -1;
    }

    void
    invalidateAll()
    {
        recomputeInteractionRates();
        interactionData_.nextInteractionTimeIndex = -1;
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
                "    num-excess-samplings: " << numExcessSamplings_ << "\n"
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
        else if (quantity == "num-active-particles"sv)
        {
            gsl_Assert(std::ssize(inspectionData.idst) >= 1 && std::ssize(inspectionData.idst) <= 4);
            auto numActiveParticles = particleData_.num() - std::ssize(interactionData_.inactiveParticleList);
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
        else
        {
            bool handled = false;
            makeshift::template_for<sizeof...(InteractionModelsT)>(
                [this, quantity, &inspectionData, &handled]
                <gsl::index I>
                (std::integral_constant<gsl::index, I>)
                {
                    if (!handled)
                    {
                        std::string prefix = fmt::format("interaction-model-{}/", I);
                        if (quantity.starts_with(prefix))
                        {
                            handled = inspectInteractionModel(quantity.substr(prefix.size()), inspectionData, I);
                        }
                    }
                },
                makeshift::tuple_index);
            return handled;
        }
    }
};
template <typename RandomNumberGeneratorT, typename ClassifierT, typename... InteractionModelsT>
RPMCTraditionalOperator(RPMCTraditionalOperatorArgs, PParticleData, RandomNumberGeneratorT, ClassifierT, InteractionModelsT...) -> RPMCTraditionalOperator<RandomNumberGeneratorT, ClassifierT, InteractionModelsT...>;


} // namespace rpmc


#endif // INCLUDED_RPMC_OPERATORS_RPMCTRADITIONAL_HPP_
