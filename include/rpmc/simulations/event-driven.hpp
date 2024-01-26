
#ifndef INCLUDED_RPMC_SIMULATIONS_EVENTDRIVEN_HPP_
#define INCLUDED_RPMC_SIMULATIONS_EVENTDRIVEN_HPP_


#include <span>
#include <array>
#include <tuple>
#include <chrono>
#include <utility>      // for move()
#include <iostream>
#include <algorithm>    // for min(), min_element()
#include <type_traits>  // for integral_constant<>
#include <string_view>

#include <makeshift/array.hpp>     // for array_iota<>()
#include <makeshift/tuple.hpp>     // for template_for()
#include <makeshift/variant.hpp>   // for expand_failfast(), visit()
#include <makeshift/constval.hpp>  // for MAKESHIFT_CONSTVAL()

#include <fmt/core.h>

#include <gsl-lite/gsl-lite.hpp>  // for index, dim, gsl_Expects(), gsl_Assert()

#include <rpmc/operators/operator.hpp>  // for InspectionData


namespace rpmc {

namespace gsl = ::gsl_lite;


template <typename ContinuousOperatorsT, typename DiscreteOperatorsT>
class EventDrivenSimulation;
template <typename... ContinuousOperatorsT, typename... DiscreteOperatorsT>
class EventDrivenSimulation<std::tuple<ContinuousOperatorsT...>, std::tuple<DiscreteOperatorsT...>>
{
private:
    std::tuple<ContinuousOperatorsT...> continuousOperators_;
    std::tuple<DiscreteOperatorsT...> discreteOperators_;
    double lastEventRateUpdateTime_;
    double time_;

        // Profiling data
    std::chrono::time_point<std::chrono::steady_clock> startTime_;
    std::chrono::steady_clock::duration samplingDuration_ = { };
    std::chrono::steady_clock::duration updatingDuration_ = { };

public:
    EventDrivenSimulation(std::tuple<ContinuousOperatorsT...> _continuousOperators, std::tuple<DiscreteOperatorsT...> _discreteOperators)
        : continuousOperators_(std::move(_continuousOperators)),
          discreteOperators_(std::move(_discreteOperators)),
          lastEventRateUpdateTime_(0.),
          time_(0.),
          startTime_(std::chrono::steady_clock::now())
    {
            // Initialize all continuous operators, then all discrete operators.
        std::apply(
            []
            (auto&... continuousOperators)
            {
                (continuousOperators.initialize(), ...);
            },
            continuousOperators_);
        std::apply(
            []
            (auto&... discreteOperators)
            {
                (discreteOperators.initialize(), ...);
            },
            discreteOperators_);
    }

    void
    runTo(double tEnd)
    {
        gsl_Expects(tEnd >= time_);

        auto t0 = std::chrono::steady_clock::now();
        for (;;)
        {
                // Retrieve continuous time scales and discrete event times and select minimum time.
            auto discreteEventTimes = std::apply(
                [](auto&... discreteOperators)
                {
                    return std::array{ discreteOperators.nextEventTime()... };
                },
                discreteOperators_);
            auto tNextEventIt = std::min_element(discreteEventTimes.begin(), discreteEventTimes.end());
            gsl::index tnextEventIdx = tNextEventIt - discreteEventTimes.begin();
            double tNextEvent = !discreteEventTimes.empty()
                ? *tNextEventIt
                : std::numeric_limits<double>::max();
            double tNext = std::min({ tNextEvent, tEnd });

            auto ts1 = std::chrono::steady_clock::now();
            samplingDuration_ += ts1 - t0;

                // Integrate continuous systems to time `tNext`.
            std::apply(
                [tNext]
                (auto&... continuousOperators)
                {
                    (continuousOperators.integrateTo(tNext), ...);
                },
                continuousOperators_);
            time_ = tNext;

            auto ts2 = std::chrono::steady_clock::now();
            updatingDuration_ += ts2 - ts1;

            if (tNextEvent > tEnd)
            {
                break;
            }

                // Simulate discrete event.
            gsl_Assert(tnextEventIdx < gsl::dim(sizeof...(DiscreteOperatorsT)));
            auto discreteEventIdxV = makeshift::expand_failfast(tnextEventIdx,
                MAKESHIFT_CONSTVAL(makeshift::array_iota<sizeof...(DiscreteOperatorsT), gsl::index>()));
            std::span<gsl::index const> particlesChanged = makeshift::visit(
                [this]
                <gsl::index DiscreteEventIdx>(std::integral_constant<gsl::index, DiscreteEventIdx>)
                {
                    auto& op = std::get<DiscreteEventIdx>(discreteOperators_);
                    return op.simulateNextEvent();
                },
                discreteEventIdxV);

            auto ts3 = std::chrono::steady_clock::now();
            samplingDuration_ += ts3 - ts2;

                // Advance all discrete operators to the last event time.
            std::apply(
                [tNext]
                (auto&... discreteOperators)
                {
                    (discreteOperators.advanceTo(tNext), ...);
                },
                discreteOperators_);
            lastEventRateUpdateTime_ = tNext;

                // Notify all operators of the discrete interaction.
            if (!particlesChanged.empty())
            {
                std::apply(
                    [particlesChanged]
                    (auto&... discreteOperators)
                    {
                        (discreteOperators.invalidate(particlesChanged), ...);
                    },
                    discreteOperators_);
                std::apply(
                    [particlesChanged]
                    (auto&... continuousOperators)
                    {
                        (continuousOperators.invalidate(particlesChanged), ...);
                    },
                    continuousOperators_);
            }

            auto ts4 = std::chrono::steady_clock::now();
            updatingDuration_ += ts4 - ts3;
            t0 = ts4;
        }

            // Notify all operators that we are returning to the caller for inspection of the system state.
        std::apply(
            []
            (auto&... continuousOperators)
            {
                (continuousOperators.tidyUp(), ...);
            },
            continuousOperators_);
        std::apply(
            []
            (auto&... discreteOperators)
            {
                (discreteOperators.tidyUp(), ...);
            },
            discreteOperators_);
    }
    template <typename F>
    void
    runTo(double tEnd, std::span<double const> snapshotTimes, F&& snapshotCallback)
    {
        for (gsl::index iTime = 0, nTime = std::ssize(snapshotTimes); iTime != nTime; ++iTime)
        {
            runTo(snapshotTimes[iTime]);
            snapshotCallback(std::as_const(iTime), snapshotTimes[iTime]);
        }
        runTo(tEnd);
    }

    bool
    inspect(std::string_view quantity, InspectionData const& inspectionData) const
    {
        if (quantity == "profiling-data")
        {
            std::chrono::steady_clock::duration totalDuration = std::chrono::steady_clock::now() - startTime_;
            std::chrono::steady_clock::duration extraDuration = totalDuration - samplingDuration_ - updatingDuration_;
            std::cout <<
                "Profiling data:\n"
                "===========\n"
                "    total-duration: " << std::chrono::duration_cast<std::chrono::duration<float>>(totalDuration).count() << "s\n"
                "    sampling-duration: " << std::chrono::duration_cast<std::chrono::duration<float>>(samplingDuration_).count() << "s\n"
                "    updating-duration: " << std::chrono::duration_cast<std::chrono::duration<float>>(updatingDuration_).count() << "s\n"
                "    extra-duration: " << std::chrono::duration_cast<std::chrono::duration<float>>(extraDuration).count() << "s\n\n";
            return true;
        }

            // If not satisfied, broadcast the request to all operators.
        bool handled = false;
        makeshift::template_for(
            [quantity, &inspectionData, &handled]
            (auto& continuousOperator, gsl::index i)
            {
                if (!handled)
                {
                    std::string prefix = fmt::format("continuous-operator-{}/", i);
                    if (quantity.starts_with(prefix))
                    {
                        handled = continuousOperator.inspect(quantity.substr(prefix.size()), inspectionData);
                    }
                }
            },
            continuousOperators_, makeshift::range_index);
        makeshift::template_for(
            [quantity, &inspectionData, &handled]
            (auto& discreteOperator, gsl::index i)
            {
                if (!handled)
                {
                    std::string prefix = fmt::format("discrete-operator-{}/", i);
                    if (quantity.starts_with(prefix))
                    {
                        handled = discreteOperator.inspect(quantity.substr(prefix.size()), inspectionData);
                    }
                }
            },
            discreteOperators_, makeshift::range_index);
        return handled;
    }
};
template <typename... ContinuousOperatorsT, typename... DiscreteOperatorsT>
EventDrivenSimulation(std::tuple<ContinuousOperatorsT...>, std::tuple<DiscreteOperatorsT...>) -> EventDrivenSimulation<std::tuple<ContinuousOperatorsT...>, std::tuple<DiscreteOperatorsT...>>;


} // namespace rpmc


#endif // INCLUDED_RPMC_SIMULATIONS_EVENTDRIVEN_HPP_
