
// Common definitions for geometric RPMC models.


#ifndef INCLUDED_RPMC_OPERATORS_RPMC_MODELS_GEOMETRIC_HPP_
#define INCLUDED_RPMC_OPERATORS_RPMC_MODELS_GEOMETRIC_HPP_


#include <span>
#include <cmath>
#include <limits>
#include <optional>
#include <algorithm>  // for max()

#include <gsl-lite/gsl-lite.hpp>  // for dim, index, ssize, type_identity<>, gsl_Expects()

#include <makeshift/tuple.hpp>  // for value_tuple<>

#include <intervals/math.hpp>
#include <intervals/logic.hpp>
#include <intervals/interval.hpp>
#include <intervals/concepts.hpp>
#include <intervals/type_traits.hpp>

#include <rpmc/const/cgs.hpp>

#include <rpmc/tools/utility.hpp>  // for Span<>

#include <rpmc/operators/rpmc/models/model.hpp>      // for NoOpInteractionModel


namespace rpmc {

namespace gsl = gsl_lite;


struct GeometricModelParams
{
    double MStar;       // mass of the central object                   (g)

    double drMin = 0.;  // minimal radial zone width                    (cm)

        // Percentual mass increase in coagulation.
        // Must be  ∈ [0, 1] .
    double relativeChangeRate = 0.05;

        // Maximum rate of change in eccentricities and inclinations.
    double maxChangeRate = 0.;

        // Swarm particle number threshold for active–passive transition.
    double NThreshold = 1.5;

        // Single particles with  St ≥ StNBodyThreshold  do not undergo viscous stirring and dynamical friction, asuming that
        // gravitational interactions are handled by an N-body code. Can be set to  0  to suppress all stochastic kinetics for
        // self-representing particles, or to  ∞  to have all gravitational interactions handled stochastically.
    double StNBodyThreshold = std::numeric_limits<double>::infinity();  // ∈ [0, ∞]

        // Single particles with  m ≥ mNBodyThreshold  do not undergo viscous stirring and dynamical friction, asuming that
        // gravitational interactions are handled by an N-body code. Can be set to  0  to suppress all stochastic kinetics for
        // self-representing particles, or to  ∞  to have all gravitational interactions handled stochastically.
    double mNBodyThreshold = std::numeric_limits<double>::infinity();  // ∈ [0, ∞]

        // Particles with  St < StDustThreshold  are assumed to flow with the gas and are thus not subject to viscous stirring or
        // dynamical friction.
        // This threshold is useful to suppress stochastic kinetics for dust particles even if no gas dynamics is used (which
        // implies that no Stokes number can be computed and  StDustThreshold  cannot be used).
    double StDustThreshold = 1.;

        // Particles with  m < mDustThreshold  are assumed to flow with the gas and are thus not subject to viscous stirring or
        // dynamical friction.
        // This threshold is useful to suppress stochastic kinetics for dust particles even if no gas dynamics is used (which
        // implies that no Stokes number can be computed and  StDustThreshold  cannot be used).
    double mDustThreshold = 0;  // (g)

        // Whether to suppress viscous stirring entirely.
    bool suppress = false;
};

struct GeometricModelArgs : GeometricModelParams
{
    explicit GeometricModelArgs(GeometricModelParams const& params)
        : GeometricModelParams(params)
    {
        gsl_Expects(params.MStar > 0);
        gsl_Expects(params.drMin >= 0);
        gsl_Expects(params.relativeChangeRate >= 0 && params.relativeChangeRate <= 1);
        gsl_Expects(params.maxChangeRate >= 0);
        gsl_Expects(params.NThreshold >= 1);
        gsl_Expects(params.StNBodyThreshold >= 0);
        gsl_Expects(params.mNBodyThreshold >= 0);
        gsl_Expects(params.StDustThreshold >= 0);
        gsl_Expects(params.mDustThreshold >= 0);
    }
};

template <template <typename> class TT = std::type_identity_t>
struct GeometricModelParticleState
{
    TT<double> M;        // total swarm mass                        (g)
    TT<double> m;        // tracer mass                             (g)
    TT<double> N;        // swarm particle count
    TT<double> a;        // semimajor axis                          (cm)
    TT<double> e;        // eccentricity
    TT<double> sininc;   // sine of inclination angle
    TT<double> St;       // Stokes number
    TT<double> rho;      // tracer bulk density                     (g/cm³)
    TT<double> vr;       // radial velocity  vᵣ = da/dt             (cm/s)
    TT<double> vphi;     // systematic azimuthal velocity  vᵩ       (cm/s)
    TT<double> hd;       // dust scale height                       (cm)

    friend constexpr auto
    reflect(gsl::type_identity<GeometricModelParticleState>)
    {
        return makeshift::value_tuple{
            &GeometricModelParticleState::M,
            &GeometricModelParticleState::m,
            &GeometricModelParticleState::N,
            &GeometricModelParticleState::a,
            &GeometricModelParticleState::e,
            &GeometricModelParticleState::sininc,
            &GeometricModelParticleState::St,
            &GeometricModelParticleState::rho,
            &GeometricModelParticleState::vr,
            &GeometricModelParticleState::vphi,
            &GeometricModelParticleState::hd
        };
    }
};
using GeometricModelState = GeometricModelParticleState<Span>;

template <template <typename> class TT = std::type_identity_t>
struct GeometricModelParticleProperties
{
    TT<double> m;       // individual particle masses                                        (mass)
    TT<double> N;       // number of particles in swarm             M/m  if  m ≠ 0 , or  0  otherwise
    TT<double> St;      // Stokes number
    TT<double> rh;      // dimensionless Hill radius                rₕ = [m/(3 M*)]¹ᐟ³        (cm)
    TT<double> vEscSq;  // squared escape velocities                vₑ² = 2Gm/R              (cm²/s²)
    TT<double> a;       //
    TT<double> e;       // eccentricity
    TT<double> sininc;  // inclination
    TT<double> E;       // kinetic pseudoenergy in planar motion    E = m e²
    TT<double> EZ;      // kinetic pseudoenergy in vertical motion  Eᴢ = m sin²i
    TT<double> beta;
    TT<double> h_a;     // bounded scaleheight                      h/a = max{ rₕ, sin i }
    TT<double> dr_a;    // bounded effective scalelength            hᵣ/a = max{ rₕ, e, Δr/a }
    TT<double> hmax_a;  // bounded maximum scaleheight              h'/a = max{ h/a, hᴅ/a }  (cm)
    TT<double> vr_vK;   // dimensionless radial drift velocity
    TT<double> vphi_vK; // dimensionless azimuthal drift velocity

    friend constexpr auto
    reflect(gsl::type_identity<GeometricModelParticleProperties>)
    {
        return makeshift::value_tuple{
            &GeometricModelParticleProperties::m,
            &GeometricModelParticleProperties::N,
            &GeometricModelParticleProperties::St,
            &GeometricModelParticleProperties::rh,
            &GeometricModelParticleProperties::vEscSq,
            &GeometricModelParticleProperties::a,
            &GeometricModelParticleProperties::e,
            &GeometricModelParticleProperties::sininc,
            &GeometricModelParticleProperties::E,
            &GeometricModelParticleProperties::EZ,
            &GeometricModelParticleProperties::beta,
            &GeometricModelParticleProperties::h_a,
            &GeometricModelParticleProperties::dr_a,
            &GeometricModelParticleProperties::hmax_a,
            &GeometricModelParticleProperties::vr_vK,
            &GeometricModelParticleProperties::vphi_vK
        };
    }
};

template <typename T>
T
computeEffectiveZoneWidth(
    GeometricModelArgs const& args,
    T N)
{
    using namespace intervals::logic;
    using namespace intervals::math;

    auto result = T{ };
    if (possibly(N > args.NThreshold))
    {
        assign_partial(result, args.drMin);
    }
    if (possibly_not(N > args.NThreshold))
    {
        assign_partial(result, 0);
    }
    return result;
}

template <template <typename> class TT>
std::optional<GeometricModelParticleProperties<TT>>
getParticleProperties(
    GeometricModelArgs const& args,
    GeometricModelParticleState<TT> const& ps)
{
    using namespace intervals::logic;
    using namespace intervals::math;

    if (args.suppress || always(ps.St < args.StDustThreshold) || always(ps.m < args.mDustThreshold))
    {
        return std::nullopt;
    }

    auto N = ps.N;

        // Dimensionless Hill radius  rₕ = Rₕ/a = [m/(3 M*)]¹ᐟ³
    auto rh = cbrt(ps.m/(3*args.MStar));

        // Keplerian tangential velocity  (cm/s)
    auto vK = sqrt(cgs::GG*args.MStar/ps.a);

        // Particle bulk radius  (cm)
    //auto R = cbrt(ps.m/(4./3*pi*ps.rho));

        // Squared escape velocity  vₑ² = 2Gm/R  (cm²/s²)
    //auto vEscSq = 2*cgs::GG*ps.m/R;
    auto vEscSq = 2*cgs::GG*square(cbrt(ps.m*sqrt(4./3*pi*ps.rho)));

        // Scaleheight  h = vᴢ/Ωᴋ = a⋅sin i  (a)
    auto h_a = ps.sininc;

        // Scalelength  hᵣ = v/Ωᴋ = a⋅e  (a)
    auto hr_a = ps.e;

    auto beta = ps.sininc/ps.e;
    if (possibly(intervals::isnan(beta)))
    {
        reset(beta, 0.);
        assign_partial(beta, std::numeric_limits<double>::infinity());
    }

        // Kinetic pseudoenergy in planar and vertical motion
    auto E = ps.m*square(ps.e);
    auto EZ = ps.m*square(ps.sininc);

        // ad-hoc modification: bounded scaleheights and scalelengths  (a)  with the Hill radius as lower bound
    auto hb_a = max(h_a, rh);
    auto hrb_a = max(hr_a, rh);

        // Effective zone width  (cm)
    auto zr = computeEffectiveZoneWidth(args, N);

        // ad-hoc modification: bounded effective scalelength  (a)
    auto dr_a = max(0.5*zr/ps.a, hrb_a);

        // ad-hoc modification: bounded maximum scaleheight  (a)
    auto hmax_a = max(hb_a, ps.hd/ps.a);

    return GeometricModelParticleProperties<TT>{
        .m = ps.m,
        .N = N,
        .St = ps.St,
        .rh = rh,
        .vEscSq = vEscSq,
        .a = ps.a,
        .e = ps.e,
        .sininc = ps.sininc,
        .E = E,
        .EZ = EZ,
        .beta = beta,
        .h_a = hb_a,
        .dr_a = dr_a,
        .hmax_a = hmax_a,
        .vr_vK = ps.vr/vK,
        .vphi_vK = ps.vphi/vK
    };
}

class GeometricInteractionModelBase : public NoOpInteractionModel
{
protected:
    GeometricModelArgs args_;
    GeometricModelState state_;

    GeometricInteractionModelBase(GeometricModelParams const& _params, GeometricModelState const& _state)
        : args_(_params), state_(_state)
    {
    }

public:
    using Locator = LogLocator;

    GeometricModelArgs const&
    getArgs() const
    {
        return args_;
    }
    GeometricModelState const&
    getState() const
    {
        return state_;
    }

    template <template <typename> class TT = std::type_identity_t>
    struct TracerSwarmInteractionData : LocalInteractionRates<TT>
    {
        TT<double> deSq_jk;
        TT<double> dsinincSq_jk;
    };

    template <typename CallbackT, typename RNG>
    void
    interact(
        CallbackT&& callback,
        RNG& /*randomNumberGenerator*/,
        gsl::index j, gsl::index /*k*/,
        GeometricModelParticleProperties<> const& /*p1*/, GeometricModelParticleProperties<> const& /*p2*/,
        TracerSwarmInteractionData<> const& interactionData)
    {
        using namespace intervals::math;

            // TODO: we expect to run into problems with negative stirring here; we probably need to implement a solver for this
            // case to do it properly, but for now we just bound  <e²>  and  <i²>  from below
        //state_.e[j] = sqrt(square(state_.e[j]) + interactionData.deSq_jk);
        //state_.sininc[j] = sqrt(square(state_.sininc[j]) + interactionData.dsinincSq_jk);
        state_.e[j] = sqrt(std::max(0., square(state_.e[j]) + interactionData.deSq_jk));
        state_.sininc[j] = sqrt(std::max(0., square(state_.sininc[j]) + interactionData.dsinincSq_jk));

        //gsl_Assert(state_.e[j] < 1. && state_.sininc[j] < 1.);
        if (state_.e[j] >= 1. || state_.sininc[j] >= 1.)
        {
                // Particle was ejected. Remove from simulation.
            state_.M[j] = 0;
            state_.m[j] = 0;
        }

        callback.invalidate(j);
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_OPERATORS_RPMC_MODELS_GEOMETRIC_HPP_
