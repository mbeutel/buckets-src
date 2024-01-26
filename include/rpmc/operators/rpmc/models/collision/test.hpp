
// Test models: constant, linear, product, and runaway kernels.


#ifndef INCLUDED_RPMC_OPERATORS_RPMC_MODELS_COLLISION_TEST_HPP_
#define INCLUDED_RPMC_OPERATORS_RPMC_MODELS_COLLISION_TEST_HPP_


#include <cmath>      // for cbrt()
#include <algorithm>  // for max()

#include <gsl-lite/gsl-lite.hpp>  // for index, type_identity<>

#include <makeshift/tuple.hpp>  // for value_tuple<>

#include <intervals/math.hpp>
#include <intervals/interval.hpp>
#include <intervals/type_traits.hpp>

#include <rpmc/operators/rpmc/locators/linear.hpp>
#include <rpmc/operators/rpmc/models/collision.hpp>  // for CollisionModelBase<>


namespace rpmc {

namespace gsl = gsl_lite;


struct ConstantKernelModelParams : CollisionModelBaseParams
{
    double collisionRate = 1.;  // collision rate  (time⁻¹)
};

template <>
struct CollisionModelArgs<ConstantKernelModelParams> : CollisionModelBaseArgs<ConstantKernelModelParams>
{
    explicit CollisionModelArgs(ConstantKernelModelParams const& params)
        : CollisionModelBaseArgs<ConstantKernelModelParams>(params)
    {
        gsl_Expects(params.collisionRate >= 0);
    }
};


class ConstantKernelModel : public CollisionModelBase<ConstantKernelModel, ConstantKernelModelParams>
{
    using base = CollisionModelBase<ConstantKernelModel, ConstantKernelModelParams>;

public:
    ConstantKernelModel(ConstantKernelModelParams const& _params, CollisionModelBaseState const& _state)
        : base(_params, _state)
    {
    }

    template <template <typename> class TT>
    base::ParticleParticleInteractionRate<TT>
    computeParticleParticleInteractionRate(
        CollisionModelBaseParticleProperties<TT> const& /*p1*/, CollisionModelBaseParticleProperties<TT> const& /*p2*/) const
    {
        return {
            .collisionRate = args_.collisionRate
        };
    }
};


struct LinearKernelModelParams : CollisionModelBaseParams
{
    double collisionRateCoefficient = 1.;  // collision rate coefficient  (time⁻¹ mass⁻¹)
};

template <>
struct CollisionModelArgs<LinearKernelModelParams> : CollisionModelBaseArgs<LinearKernelModelParams>
{
    explicit CollisionModelArgs(LinearKernelModelParams const& params)
        : CollisionModelBaseArgs<LinearKernelModelParams>(params)
    {
        gsl_Expects(params.collisionRateCoefficient >= 0);
    }
};

template <bool Locality = false>
class LinearKernelModel : public CollisionModelBase<
    LinearKernelModel<Locality>,
    LinearKernelModelParams, rpmc::CollisionModelBaseState,
    std::conditional_t<Locality, LinearLocator, DefaultLocator>>
{
    using base = CollisionModelBase<
        LinearKernelModel<Locality>,
        LinearKernelModelParams, rpmc::CollisionModelBaseState,
        std::conditional_t<Locality, LinearLocator, DefaultLocator>>;

public:
    LinearKernelModel(LinearKernelModelParams const& _params, CollisionModelBaseState const& _state)
        : base(_params, _state)
    {
    }

    template <template <typename> class TT>
    typename base::template ParticleParticleInteractionRate<TT>
    computeParticleParticleInteractionRate(
        CollisionModelBaseParticleProperties<TT> const& p1, CollisionModelBaseParticleProperties<TT> const& p2) const
    {
        auto collisionRate = this->args_.collisionRateCoefficient*(p1.m + p2.m);
        if constexpr (Locality)
        {
            return {
                .collisionRate = collisionRate,
                .interactionWidth = 0.
            };
        }
        else
        {
            return {
                .collisionRate = collisionRate
            };
        }
    }
};


struct ProductKernelModelParams : CollisionModelBaseParams
{
    double collisionRateCoefficient = 1.;  // collision rate coefficient  (time⁻¹ mass⁻²)
};

template <>
struct CollisionModelArgs<ProductKernelModelParams> : CollisionModelBaseArgs<ProductKernelModelParams>
{
    explicit CollisionModelArgs(ProductKernelModelParams const& params)
        : CollisionModelBaseArgs<ProductKernelModelParams>(params)
    {
        gsl_Expects(params.collisionRateCoefficient >= 0);
    }
};

class ProductKernelModel : public CollisionModelBase<ProductKernelModel, ProductKernelModelParams>
{
    using base = CollisionModelBase<ProductKernelModel, ProductKernelModelParams>;

public:
    ProductKernelModel(ProductKernelModelParams const& _params, CollisionModelBaseState const& _state)
        : base(_params, _state)
    {
    }

    template <template <typename> class TT>
    base::ParticleParticleInteractionRate<TT>
    computeParticleParticleInteractionRate(
        CollisionModelBaseParticleProperties<TT> const& p1, CollisionModelBaseParticleProperties<TT> const& p2) const
    {
        return {
            .collisionRate = args_.collisionRateCoefficient*(p1.m*p2.m)
        };
    }
};


struct ConstantThresholdKernelModelParams : CollisionModelBaseParams
{
    double mThreshold;          // threshold mass: particles interact only if one mass is below and the other is above the threshold  (mass)
    double collisionRate = 1.;  // collision rate                                                                                     (time⁻¹)
};

template <>
struct CollisionModelArgs<ConstantThresholdKernelModelParams> : CollisionModelBaseArgs<ConstantThresholdKernelModelParams>
{
    explicit CollisionModelArgs(ConstantThresholdKernelModelParams const& params)
        : CollisionModelBaseArgs<ConstantThresholdKernelModelParams>(params)
    {
        gsl_Expects(params.mThreshold > 0);
        gsl_Expects(params.collisionRate >= 0);
    }
};

class ConstantThresholdKernelModel : public CollisionModelBase<ConstantThresholdKernelModel, ConstantThresholdKernelModelParams>
{
    using base = CollisionModelBase<ConstantThresholdKernelModel, ConstantThresholdKernelModelParams>;

public:
    ConstantThresholdKernelModel(ConstantThresholdKernelModelParams const& _params, CollisionModelBaseState const& _state)
        : base(_params, _state)
    {
    }

    template <template <typename> class TT>
    base::ParticleParticleInteractionRate<TT>
    computeParticleParticleInteractionRate(
        CollisionModelBaseParticleProperties<TT> const& p1, CollisionModelBaseParticleProperties<TT> const& p2) const
    {
        using namespace intervals::logic;

        auto interact = (p1.m >= args_.mThreshold) != (p2.m >= args_.mThreshold);
        return {
            .collisionRate = if_else(interact, args_.collisionRate, 0.)
        };
    }
};


struct LinearThresholdKernelModelParams : CollisionModelBaseParams
{
    double mThreshold;                     // threshold mass: particles interact only if one mass is below and the other is above the threshold  (mass)
    double collisionRateCoefficient = 1.;  // collision rate coefficient                                                                         (mass⁻¹ time⁻¹)
};

template <>
struct CollisionModelArgs<LinearThresholdKernelModelParams> : CollisionModelBaseArgs<LinearThresholdKernelModelParams>
{
    explicit CollisionModelArgs(LinearThresholdKernelModelParams const& params)
        : CollisionModelBaseArgs<LinearThresholdKernelModelParams>(params)
    {
        gsl_Expects(params.mThreshold > 0);
        gsl_Expects(params.collisionRateCoefficient >= 0);
    }
};

class LinearThresholdKernelModel : public CollisionModelBase<LinearThresholdKernelModel, LinearThresholdKernelModelParams>
{
    using base = CollisionModelBase<LinearThresholdKernelModel, LinearThresholdKernelModelParams>;

public:
    LinearThresholdKernelModel(LinearThresholdKernelModelParams const& _params, CollisionModelBaseState const& _state)
        : base(_params, _state)
    {
    }

    template <template <typename> class TT>
    base::ParticleParticleInteractionRate<TT>
    computeParticleParticleInteractionRate(
        CollisionModelBaseParticleProperties<TT> const& p1, CollisionModelBaseParticleProperties<TT> const& p2) const
    {
        using namespace intervals::logic;

        auto interact = (p1.m >= args_.mThreshold) != (p2.m >= args_.mThreshold);
        return {
            .collisionRate = if_else(interact, args_.collisionRateCoefficient*(p1.m + p2.m), 0.)
        };
    }
};


struct RunawayKernelModelParams : CollisionModelBaseParams
{
    double collisionRateCoefficient = 1.;  // collision rate coefficient  (time⁻¹ mass⁻¹)
    double criticalMass = 1.e+3;           // critical mass for runaway   (mass)
};

template <>
struct CollisionModelArgs<RunawayKernelModelParams> : CollisionModelBaseArgs<RunawayKernelModelParams>
{
    explicit CollisionModelArgs(RunawayKernelModelParams const& params)
        : CollisionModelBaseArgs<RunawayKernelModelParams>(params)
    {
        gsl_Expects(params.collisionRateCoefficient >= 0);
        gsl_Expects(params.criticalMass > 0);
    }
};

template <template <typename> class TT = std::type_identity_t>
struct RunawayKernelModelParticleProperties : CollisionModelBaseParticleProperties<TT>
{
    TT<double> m23;  // m²ᐟ³

    friend constexpr auto
    reflect(gsl::type_identity<RunawayKernelModelParticleProperties>)
    {
        return makeshift::value_tuple{
            gsl::type_identity<CollisionModelBaseParticleProperties<TT>>{ },
            makeshift::value_tuple{
                &RunawayKernelModelParticleProperties::m23
            }
        };
    }
};

template <template <typename> class TT>
RunawayKernelModelParticleProperties<TT>
getParticleProperties(
    CollisionModelArgs<RunawayKernelModelParams> const& args,
    CollisionModelBaseParticleState<TT> const& ps)
{
    using namespace intervals::math;

    auto bps = getParticleProperties(static_cast<CollisionModelBaseArgs<RunawayKernelModelParams> const&>(args), ps);
    return {
        bps,
        /*.m23 =*/ square(cbrt(bps.m))
    };
}

class RunawayKernelModel : public CollisionModelBase<RunawayKernelModel, RunawayKernelModelParams>
{
        // According to GCC, the scoping rules disallow inline friend functions of nested classes from accessing
        // private symbols, which MSVC has no problems with. I didn't bother trying to find out who is right.
public:
    using base = CollisionModelBase<RunawayKernelModel, RunawayKernelModelParams>;

private:
    double rcpCriticalMass23_;  // criticalMass⁻²ᐟ³

public:
    RunawayKernelModel(RunawayKernelModelParams const& _params, CollisionModelBaseState const& _state)
        : base(_params, _state),
          rcpCriticalMass23_(1/std::cbrt(_params.criticalMass*_params.criticalMass))
    {
    }

    template <template <typename> class TT>
    base::ParticleParticleInteractionRate<TT>
    computeParticleParticleInteractionRate(
        RunawayKernelModelParticleProperties<TT> const& p1, RunawayKernelModelParticleProperties<TT> const& p2) const
    {
        using namespace intervals::math;

        auto m23Max = max(p1.m23, p2.m23);
        return {
            .collisionRate = args_.collisionRateCoefficient*m23Max*(1 + rcpCriticalMass23_*m23Max)
        };
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_OPERATORS_RPMC_MODELS_COLLISION_TEST_HPP_
