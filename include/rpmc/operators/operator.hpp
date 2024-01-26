
#ifndef INCLUDED_RPMC_OPERATORS_OPERATOR_HPP_
#define INCLUDED_RPMC_OPERATORS_OPERATOR_HPP_


#include <span>
#include <limits>
#include <memory>       // for unique_ptr<>
#include <utility>      // for move()
#include <concepts>
#include <string_view>

#include <gsl-lite/gsl-lite.hpp>  // for index, not_null<>, gsl_FailFast()


namespace rpmc {

namespace gsl = ::gsl_lite;


struct InspectionData
{
    std::span<double> fdst;
    std::span<gsl::index> idst;
    std::span<double const> fparams;
    std::span<gsl::index const> iparams;
};


template <typename OperatorT>
concept Operator = requires(OperatorT& op, OperatorT const& cop,
    std::span<gsl::index const> updatedParticleIndices,
    std::string_view sv, InspectionData const& inspectionData)
{
    op.initialize();
    op.invalidate(updatedParticleIndices);
    op.invalidateAll();
    op.tidyUp();
    op.synchronize();
    { cop.inspect(sv, inspectionData) } -> std::same_as<bool>;
};
template <typename OperatorT>
concept ContinuousOperator = Operator<OperatorT> &&
requires(OperatorT& op, double time)
{
    { op.nextUpdateTime() } -> std::convertible_to<double>;
    op.integrateTo(time);
};
template <typename OperatorT>
concept DiscreteOperator = Operator<OperatorT> &&
requires(OperatorT& op, double time)
{
    { op.nextEventTime() } -> std::convertible_to<double>;
    op.advanceTo(time);
    { op.simulateNextEvent() } -> std::convertible_to<std::span<gsl::index const>>;
};


class NoOpOperator
{
public:
    void
    initialize()
    {
    }
    void
    invalidate(std::span<gsl::index const> /*updatedParticleIndices*/)
    {
    }
    void
    invalidateAll()
    {
    }
    void
    synchronize()
    {
    }
    void
    tidyUp()
    {
    }
    bool
    inspect(std::string_view /*quantity*/, InspectionData const& /*inspectionData*/) const
    {
        return false;
    }
};

class NoOpContinuousOperator : public NoOpOperator
{
public:
    void
    integrateTo(double /*tEnd*/)
    {
    }
};

class NoOpDiscreteOperator : public NoOpOperator
{
public:
    double
    nextEventTime()
    {
        return std::numeric_limits<double>::infinity();
    }
    void
    advanceTo(double /*tEnd*/)
    {
    }
    std::span<gsl::index const>
    simulateNextEvent()
    {
        gsl_FailFast();
    }
};


class PContinuousOperator
{
private:
    struct IConcept
    {
        virtual ~IConcept() { }
        virtual void initialize() = 0;
        virtual void integrateTo(double t) = 0;
        virtual void invalidate(std::span<gsl::index const> updatedParticleIndices) = 0;
        virtual void invalidateAll() = 0;
        virtual void synchronize() = 0;
        virtual void tidyUp() = 0;
        virtual bool inspect(std::span<double> dst, std::string_view quantity, std::span<double const> params) const = 0;
        virtual bool inspect(std::string_view quantity, InspectionData const& inspectionData) const = 0;

    };
    template <typename T>
    struct Model final : IConcept
    {
        T impl_;

        Model(T&& _impl) : impl_(std::move(_impl)) { }

        void
        initialize() override
        {
            impl_.initialize();
        }
        void
        integrateTo(double t) override
        {
            impl_.integrateTo(t);
        }
        void
        invalidate(std::span<gsl::index const> updatedParticleIndices) override
        {
            impl_.invalidate(updatedParticleIndices);
        }
        void
        invalidateAll() override
        {
            impl_.invalidateAll();
        }
        void
        synchronize() override
        {
            impl_.synchronize();
        }
        void
        tidyUp() override
        {
            impl_.tidyUp();
        }
        bool
        inspect(std::string_view quantity, InspectionData const& inspectionData) const override
        {
            return impl_.inspect(quantity, inspectionData);
        }
    };

    gsl::not_null<std::unique_ptr<IConcept>> impl_;

public:
    template <ContinuousOperator T>
    PContinuousOperator(T op)
        : impl_(gsl::not_null(gsl::make_unique<Model<T>>(std::move(op))))
    {
    }
    void
    initialize()
    {
        impl_->initialize();
    }
    void
    integrateTo(double t)
    {
        impl_->integrateTo(t);
    }
    void
    invalidate(std::span<gsl::index const> updatedParticleIndices)
    {
        impl_->invalidate(updatedParticleIndices);
    }
    void
    invalidateAll()
    {
        impl_->invalidateAll();
    }
    void
    synchronize()
    {
        impl_->synchronize();
    }
    void
    tidyUp()
    {
        impl_->tidyUp();
    }
    bool
    inspect(std::string_view quantity, InspectionData const& inspectionData) const
    {
        return impl_->inspect(quantity, inspectionData);
    }
};

class PDiscreteOperator
{
private:
    struct IConcept
    {
        virtual ~IConcept() { }
        virtual void initialize() = 0;
        virtual double nextEventTime() = 0;
        virtual std::span<gsl::index const> simulateNextEvent() = 0;
        virtual void advanceTo(double t) = 0;
        virtual void invalidate(std::span<gsl::index const> updatedParticleIndices) = 0;
        virtual void invalidateAll() = 0;
        virtual void synchronize() = 0;
        virtual void tidyUp() = 0;
        virtual bool inspect(std::string_view quantity, InspectionData const& inspectionData) const = 0;
    };
    template <typename T>
    struct Model final : IConcept
    {
        T impl_;

        Model(T&& _impl) : impl_(std::move(_impl)) { }

        void
        initialize() override
        {
            impl_.initialize();
        }
        double
        nextEventTime() override
        {
            return impl_.nextEventTime();
        }
        std::span<gsl::index const>
        simulateNextEvent() override
        {
            return impl_.simulateNextEvent();
        }
        void
        advanceTo(double t) override
        {
            impl_.advanceTo(t);
        }
        void
        invalidate(std::span<gsl::index const> updatedParticleIndices) override
        {
            impl_.invalidate(updatedParticleIndices);
        }
        void
        invalidateAll() override
        {
            impl_.invalidateAll();
        }
        void
        synchronize() override
        {
            impl_.synchronize();
        }
        void
        tidyUp() override
        {
            impl_.tidyUp();
        }
        bool
        inspect(std::string_view quantity, InspectionData const& inspectionData) const override
        {
            return impl_.inspect(quantity, inspectionData);
        }
    };

    gsl::not_null<std::unique_ptr<IConcept>> impl_;

public:
    template <DiscreteOperator T>
    PDiscreteOperator(T op)
        : impl_(gsl::not_null(gsl::make_unique<Model<T>>(std::move(op))))
    {
    }
    void
    initialize()
    {
        impl_->initialize();
    }
    double
    nextEventTime()
    {
        return impl_->nextEventTime();
    }
    std::span<gsl::index const>
    simulateNextEvent()
    {
        return impl_->simulateNextEvent();
    }
    void
    advanceTo(double t)
    {
        impl_->advanceTo(t);
    }
    void
    invalidate(std::span<gsl::index const> updatedParticleIndices)
    {
        impl_->invalidate(updatedParticleIndices);
    }
    void
    invalidateAll()
    {
        impl_->invalidateAll();
    }
    void
    synchronize()
    {
        impl_->synchronize();
    }
    void
    tidyUp()
    {
        impl_->tidyUp();
    }
    bool
    inspect(std::string_view quantity, InspectionData const& inspectionData) const
    {
        return impl_->inspect(quantity, inspectionData);
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_OPERATORS_OPERATOR_HPP_
