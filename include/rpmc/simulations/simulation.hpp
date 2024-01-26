
#ifndef INCLUDED_RPMC_SIMULATIONS_SIMULATION_HPP_
#define INCLUDED_RPMC_SIMULATIONS_SIMULATION_HPP_


#include <span>
#include <memory>       // for unique_ptr<>
#include <utility>      // for move()
#include <concepts>
#include <functional>
#include <string_view>

#include <gsl-lite/gsl-lite.hpp>  // for index, not_null<>

#include <rpmc/operators/operator.hpp>  // for InspectionData


namespace rpmc {

namespace gsl = ::gsl_lite;


template <typename SimulationT>
concept Simulation = requires(SimulationT& sim, SimulationT const& csim,
    double tEnd, std::span<double const> snapshotTimes, std::function<void(gsl::index iTime, double time)> snapshotCallback,
    std::string_view sv, InspectionData const& inspectionData)
{
    sim.runTo(tEnd, snapshotTimes, snapshotCallback);
    { csim.inspect(sv, inspectionData) } -> std::same_as<bool>;
};


class NoOpSimulation
{
public:
    void
    runTo(double /*tEnd*/, std::span<double const> /*snapshotTimes*/, std::function<void(gsl::index iTime, double time)> /*snapshotCallback*/)
    {
    }
    bool
    inspect(std::string_view /*quantity*/, InspectionData const& /*inspectionData*/) const
    {
        return false;
    }
};


class PSimulation
{
private:
    struct IConcept
    {
        virtual ~IConcept() { }
        virtual void runTo(double tEnd, std::span<double const> snapshotTimes, std::function<void(gsl::index iTime, double time)> snapshotCallback) = 0;
        virtual bool inspect(std::string_view quantity, InspectionData const& inspectionData) const = 0;
    };
    template <typename T>
    struct Model final : IConcept
    {
        T impl_;

        Model(T&& _impl) : impl_(std::move(_impl)) { }

        void
        runTo(double tEnd, std::span<double const> snapshotTimes, std::function<void(gsl::index iTime, double time)> snapshotCallback) override
        {
            impl_.runTo(tEnd, snapshotTimes, snapshotCallback);
        }
        bool
        inspect(std::string_view quantity, InspectionData const& inspectionData) const override
        {
            return impl_.inspect(quantity, inspectionData);
        }
    };

    gsl::not_null<std::unique_ptr<IConcept>> impl_;

public:
    template <Simulation T>
    PSimulation(T op)
        : impl_(gsl::not_null(gsl::make_unique<Model<T>>(std::move(op))))
    {
    }
    void
    runTo(double tEnd)
    {
        impl_->runTo(tEnd, { }, [](gsl::index, double) { });
    }
    void
    runTo(double tEnd, std::span<double const> snapshotTimes, std::function<void(gsl::index iTime, double time)> snapshotCallback)
    {
        impl_->runTo(tEnd, snapshotTimes, snapshotCallback);
    }
    bool
    inspect(std::string_view quantity, InspectionData const& inspectionData) const
    {
        return impl_->inspect(quantity, inspectionData);
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_SIMULATIONS_SIMULATION_HPP_
