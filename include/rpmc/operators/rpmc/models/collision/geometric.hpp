
// Geometric collision model.


#ifndef INCLUDED_RPMC_OPERATORS_RPMC_MODELS_COLLISION_GEOMETRIC_HPP_
#define INCLUDED_RPMC_OPERATORS_RPMC_MODELS_COLLISION_GEOMETRIC_HPP_


#include <span>
#include <cmath>      // for cbrt()
#include <array>
#include <random>
#include <optional>
#include <algorithm>  // for max()

#include <gsl-lite/gsl-lite.hpp>  // for index, type_identity<>

#include <makeshift/tuple.hpp>    // for value_tuple<>
#include <makeshift/utility.hpp>  // for overload<>
#include <makeshift/variant.hpp>  // for visit()

#include <intervals/math.hpp>
#include <intervals/logic.hpp>
#include <intervals/interval.hpp>
#include <intervals/concepts.hpp>
#include <intervals/type_traits.hpp>

#include <rpmc/tools/utility.hpp>        // for Span<>

#include <rpmc/operators/rpmc/locators/log.hpp>
#include <rpmc/operators/rpmc/models/collision.hpp>  // for CollisionModelBase<>


namespace rpmc {

namespace gsl = gsl_lite;


enum class CollisionOutcomes
{
    none = 0,

        // Admit coagulation of colliding particles.
    coagulation   = 0b01,

        // Admit fragmentation of particles.
    fragmentation = 0b10,

        // All of the above.
    all           = 0b11
};
gsl_DEFINE_ENUM_BITMASK_OPERATORS(CollisionOutcomes)
constexpr auto
reflect(gsl::type_identity<CollisionOutcomes>)
{
    return makeshift::value_tuple{
        "CollisionOutcomes",
        "collision outcomes to consider",
        std::array{
            makeshift::value_tuple{ CollisionOutcomes::none, "none" },
            makeshift::value_tuple{ CollisionOutcomes::coagulation, "coagulation" },
            makeshift::value_tuple{ CollisionOutcomes::fragmentation, "fragmentation" },
            makeshift::value_tuple{ CollisionOutcomes::all, "all" }
        }
    };
}

struct GeometricCollisionModelParams : CollisionModelBaseParams
{
    double MStar;                         // mass of the central object                                         (g)
    double drMin = 0.;                    // minimal radial zone width                                          (cm)
    double restitutionCoef;               // coefficient of restitution  ε  in simplified fragmentation model
    double fragmentRadius;                // bulk radius of fragments in simplified fragmentation model         (cm)
    CollisionOutcomes collisionOutcomes;  // which collision results to permit

        // Swarm particle number threshold for active–passive transition.
    double NThreshold = 1.5;

        // Single particles with  St ≥ StNBodyThreshold  do not undergo stochastic collisions, asuming that collisional
        // interactions are handled by an N-body code. Can be set to  0  to suppress all stochastic collisions of
        // self-representing particles, or to  ∞  to have all collisions handled stochastically.
    double StNBodyThreshold = std::numeric_limits<double>::infinity();  // ∈ [0, ∞]

        // Single particles with  m ≥ mNBodyThreshold  do not undergo stochastic collisions, asuming that collisional
        // interactions are handled by an N-body code. Can be set to  0  to suppress all stochastic collisions of
        // self-representing particles, or to  ∞  to have all collisions handled stochastically.
    double mNBodyThreshold = std::numeric_limits<double>::infinity();  // ∈ [0, ∞]

        // Particles with  St < StDustThreshold  are assumed to flow with the gas and are not subject to mutual collisions.
        // This threshold is useful to suppress mutual stochastic collisions for dust particles even if no gas dynamics is used
        // (which implies that no Stokes number can be computed and  StDustThreshold  cannot be used).
    double StDustThreshold = 1.;

        // Particles with  m < mDustThreshold  are assumed to flow with the gas and are not subject to mutual collisions.
        // This threshold is useful to suppress mutual stochastic collisions for dust particles even if no gas dynamics is used
        // (which implies that no Stokes number can be computed and  StDustThreshold  cannot be used).
    double mDustThreshold = 0;  // (g)
};

template <>
struct CollisionModelArgs<GeometricCollisionModelParams> : CollisionModelBaseArgs<GeometricCollisionModelParams>
{
    explicit CollisionModelArgs(GeometricCollisionModelParams const& params)
        : CollisionModelBaseArgs<GeometricCollisionModelParams>(params)
    {
        gsl_Expects(params.MStar > 0);
        gsl_Expects(params.drMin >= 0);
        gsl_Expects(params.restitutionCoef >= 0 && params.restitutionCoef <= 1);
        gsl_Expects(params.fragmentRadius > 0);
        gsl_Expects(params.NThreshold >= 1);
        gsl_Expects(params.StNBodyThreshold >= 0);
        gsl_Expects(params.mNBodyThreshold >= 0);
        gsl_Expects(params.StDustThreshold >= 0);
        gsl_Expects(params.mDustThreshold >= 0);
    }
};

template <template <typename> class TT = std::type_identity_t>
struct GeometricCollisionModelParticleState : CollisionModelBaseParticleState<TT>
{
    TT<double> a;      // tracer semimajor axis                                          (cm)
    TT<double> e;      // tracer eccentricity
    TT<double> sininc; // tracer inclination
    TT<double> vr;     // radial velocity  vᵣ = da/dt                                    (cm/s)
    TT<double> vphi;   // systematic azimuthal velocity  vᵩ                              (cm/s)
    TT<double> hd;     // dust scale height                                              (cm)
    TT<double> St;     // Stokes number
    TT<double> rho;    // tracer bulk density                                            (g/cm³)

    friend constexpr auto
    reflect(gsl::type_identity<GeometricCollisionModelParticleState>)
    {
        return makeshift::value_tuple{
            gsl::type_identity<CollisionModelBaseParticleState<TT>>{ },
            makeshift::make_value_tuple(
                &GeometricCollisionModelParticleState::a,
                &GeometricCollisionModelParticleState::e,
                &GeometricCollisionModelParticleState::sininc,
                &GeometricCollisionModelParticleState::vr,
                &GeometricCollisionModelParticleState::vphi,
                &GeometricCollisionModelParticleState::hd,
                &GeometricCollisionModelParticleState::St,
                &GeometricCollisionModelParticleState::rho
            )
        };
    }
};
using GeometricCollisionModelState = GeometricCollisionModelParticleState<Span>;

template <template <typename> class TT = std::type_identity_t>
struct GeometricCollisionModelParticleProperties : CollisionModelBaseParticleProperties<TT>
{
    TT<double> St;      // Stokes number
    TT<double> R;       // tracer radius                                            (cm)
    TT<double> rh;      // dimensionless Hill radius        rₕ = [m/(3 M*)]¹ᐟ³        (cm)
    TT<double> vEscSq;  // squared escape velocities        vₑ² = 2Gm/R             (cm²/s²)
    TT<double> a;       //
    TT<double> e;       // eccentricity
    TT<double> h_a;     // bounded scaleheight              h/a = max{ rₕ, sin i }
    TT<double> dr_a;    // bounded effective scalelength    hᵣ/a = max{ rₕ, e, Δr/a }
    TT<double> hmax_a;  // bounded maximum scaleheight      h'/a = max{ h/a, hᴅ/a }  (cm)
    TT<double> vr_vK;   // dimensionless radial drift velocity
    TT<double> vphi_vK; // dimensionless azimuthal drift velocity

    friend constexpr auto
    reflect(gsl::type_identity<GeometricCollisionModelParticleProperties>)
    {
        return makeshift::value_tuple{
            gsl::type_identity<CollisionModelBaseParticleProperties<TT>>{ },
            makeshift::make_value_tuple(
                &GeometricCollisionModelParticleProperties::St,
                &GeometricCollisionModelParticleProperties::R,
                &GeometricCollisionModelParticleProperties::rh,
                &GeometricCollisionModelParticleProperties::vEscSq,
                &GeometricCollisionModelParticleProperties::a,
                &GeometricCollisionModelParticleProperties::e,
                &GeometricCollisionModelParticleProperties::h_a,
                &GeometricCollisionModelParticleProperties::dr_a,
                &GeometricCollisionModelParticleProperties::hmax_a,
                &GeometricCollisionModelParticleProperties::vr_vK,
                &GeometricCollisionModelParticleProperties::vphi_vK
            )
        };
    }
};

template <typename T>
T
computeEffectiveZoneWidth(
    CollisionModelBaseArgs<GeometricCollisionModelParams> const& args,
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
std::optional<GeometricCollisionModelParticleProperties<TT>>
getParticleProperties(
    CollisionModelArgs<GeometricCollisionModelParams> const& args,
    GeometricCollisionModelParticleState<TT> const& ps)
{
    using namespace intervals::math;

    auto bp = getParticleProperties(args, static_cast<CollisionModelBaseParticleState<TT> const&>(ps));

        // Dimensionless Hill radius  rₕ = Rₕ/a = [m/(3 M*)]¹ᐟ³
    auto rh = cbrt(ps.m/(3*args.MStar));

        // Keplerian tangential velocity  (cm/s)
    auto vK = sqrt(cgs::GG*args.MStar/ps.a);

        // Particle bulk radius  (cm)
    auto R = cbrt(ps.m/(4./3*pi*ps.rho));

        // Squared escape velocity  vₑ² = 2Gm/R  (cm²/s²)
    //auto vEscSq = 2*cgs::GG*ps.m/R;
    auto vEscSq = 2*cgs::GG*square(cbrt(ps.m*sqrt(4./3*pi*ps.rho)));

        // Scaleheight  h = vᴢ/Ωᴋ = a⋅sin i  (a)
    auto h_a = ps.sininc;

        // Scalelength  hᵣ = v/Ωᴋ = a⋅e  (a)
    auto hr_a = ps.e;

        // ad-hoc modification: bounded scaleheights and scalelengths  (a)  with the Hill radius as lower bound
    auto hb_a = max(h_a, rh);
    auto hrb_a = max(hr_a, rh);

        // Effective zone width  (cm)
    auto zr = computeEffectiveZoneWidth(args, bp.N);

        // ad-hoc modification: bounded effective scalelength  (a)
    auto dr_a = max(0.5*zr/ps.a, hrb_a);

        // ad-hoc modification: bounded maximum scaleheight  (a)
    auto hmax_a = max(hb_a, ps.hd/ps.a);

    return GeometricCollisionModelParticleProperties<TT>{
        bp,
        /*.St =*/ ps.St,
        /*.R =*/ R,
        /*.rh =*/ rh,
        /*.vEscSq =*/ vEscSq,
        /*.a =*/ ps.a,
        /*.e =*/ ps.e,
        /*.h_a =*/ hb_a,
        /*.dr_a =*/ dr_a,
        /*.hmax_a =*/ hmax_a,
        /*.vr_vK =*/ ps.vr/vK,
        /*.vphi_vK =*/ ps.vphi/vK
    };
}

class GeometricCollisionModel : public CollisionModelBase<GeometricCollisionModel, GeometricCollisionModelParams, GeometricCollisionModelState, LogLocator>
{
    using base = CollisionModelBase<GeometricCollisionModel, GeometricCollisionModelParams, GeometricCollisionModelState, LogLocator>;

public:
    GeometricCollisionModel(
        GeometricCollisionModelParams const& _params, GeometricCollisionModelState const& _state)
        : base(_params, _state)
    {
    }

    template <template <typename> class TT>
    typename base::template ParticleParticleInteractionRate<TT>
    computeParticleParticleInteractionRate(
        GeometricCollisionModelParticleProperties<TT> const& p1, GeometricCollisionModelParticleProperties<TT> const& p2) const
    {
        using namespace intervals::logic;
        using namespace intervals::math;

        auto lambda = TT<double>{ };
        auto interactionWidth = TT<double>{ };
        auto ffrag = TT<double>{ };

            // We treat collisions stochastically only if at least one of the particles is above the dust threshold.
        auto anyAboveDustThreshold =
            ((p1.St >= this->args_.StDustThreshold) & (p1.m >= this->args_.mDustThreshold)) |
            ((p2.St >= this->args_.StDustThreshold) & (p2.m >= this->args_.mDustThreshold));

            // We exempt individual self-representing particles from mutual stochastic collisions if their
            // Stokes number or their mass exceeds a certain threshold (e.g. to have their interaction handled
            // by an N-body simulation).
        auto isP1NBAndSelfRepresenting = ((p1.St >= this->args_.StNBodyThreshold) | (p1.m >= this->args_.mNBodyThreshold)) & (p1.N <= this->args_.NThreshold);
        auto isP2NBAndSelfRepresenting = ((p2.St >= this->args_.StNBodyThreshold) | (p2.m >= this->args_.mNBodyThreshold)) & (p2.N <= this->args_.NThreshold);
        auto bothAreNBAndSelfRepresenting = isP1NBAndSelfRepresenting & isP2NBAndSelfRepresenting;

        auto handleStochastically = anyAboveDustThreshold & !bothAreNBAndSelfRepresenting;
        if (possibly_not(handleStochastically))
        {
            //    // Average semimajor axis  (a)
            //TT<double> a = sqrt(p1.a*p2.a);  // geometric mean

                // Dimensionless Hill radius of heavier particle
            TT<double> rhM = max(p1.rh, p2.rh);

                // Estimate a characteristic length; it just shouldn't be 0.
                // TODO: we should simply support 0 as a characteristic length. What could possibly go wrong?
            auto Rcol_a = 2.5*rhM + (p1.dr_a + p2.dr_a);

            assign_partial(lambda, 0);
            assign_partial(ffrag, 0.);
            assign_partial(interactionWidth, Rcol_a);
        }
        if (possibly(handleStochastically))
    {
                // Square of approximate mutual escape velocity  (cm/s)
                //
                // The approximation is
                //
                //     2 G (mⱼ + mₖ)/(Rⱼ + Rₖ) ≈ max{ 2 G mⱼ/Rⱼ, 2 G mₖ/Rₖ } ;
                // 
                // it is exact for  j = k  and asymptotically exact for  mⱼ ≪ mₖ, Rⱼ ≪ Rₖ  or  mⱼ ≫ mₖ, Rⱼ ≫ Rₖ .
                //
            TT<double> vEscSq = max(p1.vEscSq, p2.vEscSq);

                // Average semimajor axes  (cm)
            TT<double> a = sqrt(p1.a*p2.a);  // geometric mean
            TT<double> a1_a = sqrt(p1.a/p2.a);
            TT<double> a2_a = sqrt(p2.a/p1.a);

                // Planar rms-velocity  (vᴋ)
            TT<double> v1_vK = p1.e;
            TT<double> v2_vK = p2.e;

                // Relative planar velocity  (vᴋ)
            TT<double> w_vKSq = square(p1.vr_vK - p2.vr_vK) + square(p1.vphi_vK - p2.vphi_vK) + square(v1_vK) + square(v2_vK);
            TT<double> w_vK = sqrt(w_vKSq);

                // Keplerian angular velocity  (s⁻¹)
            TT<double> OmegaK = sqrt(cgs::GG*this->args_.MStar/cube(a));

                // Keplerian tangential velocity  (cm/s)
            TT<double> vK = sqrt(cgs::GG*this->args_.MStar/a);

                // Dimensionless Hill radius of heavier particle
            TT<double> rhM = max(p1.rh, p2.rh);

                // Hill velocity of heavier particle  (vᴋ)
            TT<double> vhM_vK = rhM;

                // Sum of particle bulk radii  (cm)
            TT<double> RS = p1.R + p2.R;

            auto Rcol_a = TT<double>{ };
            auto dispersionDominatedRegime = 2.5*vhM_vK <= w_vK;
            if (possibly(dispersionDominatedRegime))  // 2.5 vₕ ≤ w  (superescape regime or dispersion-dominated regime)
            {
                auto w_vKc = constrain(w_vK, dispersionDominatedRegime);

                    // Bounded maximum distribution height  (a)
                    // TODO: shouldn't we use `p1.hmax`, `p2.hmax` instead?
                TT<double> hmax_a = max(p1.h_a, p2.h_a);

                    // Interaction radii  (cm)
                    //
                    // The cross-section is enhanced by the gravitational focusing factor which can be derived as follows:
                    //
                    // Consider the trajectories of two particles 1 and 2 of type  j  and  k  from the (non-inertial) reference
                    // frame of particle 1. In particle 1's sphere of influence, particle 2 will pursue a hyperbolic orbit with the
                    // following orbital parameters:
                    //
                    //     a = -µ/v∞² < 0      semi-major axis
                    //    b² = a²⋅(e² - 1)     squared periapsis distance ≡ squared impact parameter
                    //     Rₚ = a⋅(1 - e)      periapsis distance
                    //      μ = G⋅(mⱼ + mₖ)    standard gravitational parameter
                    //     v∞ ≈ √<Δvⱼₖ²>       hyperbolic excess velocity
                    //
                    // The collision rate can be expressed in terms of the impact parameter:
                    //
                    //     σ = π b²
                    //
                    // A collision occurs when the periapsis distance is smaller than the sum of the physical radii of the particles.
                    // To compute the collision rate, we thus set  mₚ = mⱼ + mₖ  and  Rₚ = Rⱼ + Rₖ  and obtain
                    //
                    //     σ = π rₚ²⋅[1 + 2 G mₚ/(Rₚ <Δvⱼₖ²>)]
                    //       ≡ σ₀⋅[1 + 2 G mₚ/(Rₚ <Δvⱼₖ²>)]
                    //       = σ₀⋅[1 + vₑ²/<Δvⱼₖ²>]
                    //
                    // where  σ₀ = π Rₚ²  is the geometric collision rate, and  vₑ = 2 G mₚ/Rₚ  is the mutual escape velocity.
                    //
                    // Following Ormel et al. (2010), Sections B.1.1 and B.1.2, we insert two order-of-unity calibration factors
                    //  A₁  and  A₂  here. It is not clear from Ormel et al. (2010) whether the calibration factors of interaction
                    // rates would also affect the interaction radii used to determine the filling factors. For now we assume they do.
                    //
                constexpr double A1 = 0.90;
                constexpr double A2 = 1.5;
                auto Rcol0_a = RS/a*sqrt(A1 + A2*vEscSq/square(w_vKc*vK));

                    // Filling factors
                TT<double> phiX = detail::fillingFactor(p1.dr_a, p2.dr_a, a1_a, a2_a, Rcol0_a);
                TT<double> phiZ = detail::fillingFactor(p1.hmax_a, p2.hmax_a, Rcol0_a);
                TT<double> phi = phiX*phiZ;
                assign_partial(Rcol_a, Rcol0_a + (p1.dr_a + p2.dr_a));

                    // Interaction frequency  (s⁻¹)
                TT<double> omega = hmax_a/4*OmegaK;

                double f = 1.;

                    // λⱼₖ⁽¹¹⁾ = f ωⱼₖ φr φz
                assign_partial(lambda, f*omega*phi);
            }
            auto hillRegime = w_vK <= 2.5*vhM_vK;  // ad-hoc modification:  vₕ → 2.5 vₕ  because it doesn't make sense
            if (possibly(hillRegime))  // w ≤ vₕ  (Hill regime)
            {
                    // Cf. Ormel et al. (2010), Section B.1.3.
                constexpr double A3 = 2.9;

                    // Interaction radii  (cm)
                auto Rcolmin_a = 1.7*rhM;
                auto Rcol0_a = 2.5*rhM;

                    // Filling factors
                TT<double> phiX = detail::fillingFactor(p1.dr_a, p2.dr_a, a1_a, a2_a, Rcolmin_a, Rcol0_a);
                TT<double> phiZ = detail::fillingFactor(p1.hmax_a, p2.hmax_a, Rcol0_a);
                TT<double> phi = phiX*phiZ;
                assign_partial(Rcol_a, Rcol0_a + (p1.dr_a + p2.dr_a));

                    // Maximum bounded radial distribution width  (cm)
                TT<double> drmax_a = max(p1.dr_a, p2.dr_a);

                    // Impact parameter  (cm)
                TT<double> bcol_a = sqrt(RS/a*rhM);
                TT<double> bcol_a_rhM = sqrt(RS/a/rhM);

                    // Probability that a projectile that has entered the Hill sphere will actually hit the target
                TT<double> fhit = A3*bcol_a_rhM*min(1., bcol_a/drmax_a);

                    // Interaction frequency  (s⁻¹)
                TT<double> omega = vhM_vK*OmegaK;

                TT<double> f = fhit*(3.2/8);

                    // λⱼₖ⁽¹¹⁾ = f ωⱼₖ φr φz
                assign_partial(lambda, f*omega*phi);
            }
            assign_partial(interactionWidth, Rcol_a);
            //if (possibly((w >= vhM) & (w <= 2.5*vhM)))  // TODO: how about this regime,  vₕ < w < 2.5 vₕ ??
            //{
            //}

            //TT<double> dNmax{ };
            auto vEsc_vKSq = vEscSq/square(vK);
            auto superescapeRegime = w_vKSq >= vEsc_vKSq;
            if ((this->args_.collisionOutcomes & CollisionOutcomes::fragmentation) != CollisionOutcomes::none)
            {
                if (possibly(superescapeRegime))  // superescape regime – particles may fragment
                {
                    auto w_vKSqc = constrain(w_vKSq, superescapeRegime);
                    auto w_vEscSq = max(1., w_vKSqc/vEsc_vKSq);

                        // Reduced particle mass  μ = (m₁ m₂)/(m₁ + m₂)  (g)
                    //TT<double> mu = p1.m*p2.m/(p1.m + p2.m);
                    //TT<double> mu = 1/(1/p1.m + 1/p2.m);

                        // Collisional energy  2 Ecol = μ vₐ²  (g cm²/s²)
                    //TT<double> Ecol2_vKSq = mu*w_vKSqc;

                        // Maximal mass fraction  ffrag = min{ 1; ε Ecol / [(m₁ + m₂) vₑ²/2] }  ending up as fragments
                    //TT<double> fFragMax = Ecol2_vKSq/((p1.m + p2.m)*vEscSq_vKSq);
                    //TT<double> fFragMax = w_vEscSq/((1 + p1.m/p2.m)*(1 + p2.m/p1.m));
                    TT<double> fFragMax = w_vEscSq/(2 + p1.m/p2.m + p2.m/p1.m);
                    TT<double> lfFrag = min(this->args_.restitutionCoef*fFragMax, 1.);
                    assign_partial(ffrag, lfFrag);
                }
                if (possibly_not(superescapeRegime))  // dispersion-dominated regime or Hill regime – particles will coagulate
                {
                    assign_partial(ffrag, 0.);
                }
            }
            else
            {
                assign(ffrag, 0.);
            }
        }
        return {
            .collisionRate = lambda,
            .ffrag = ffrag,
            .interactionWidth = interactionWidth
        };
    }

    template <typename RNG, typename ParticlePropertiesT>
    void
    coagulate(
        RNG& /*randomNumberGenerator*/,
        gsl::index j, gsl::index k,
        ParticlePropertiesT const& p1, ParticlePropertiesT const& p2,
        double /*ffrag*/, double beta_jk, double /*mc*/)
    {
        using namespace intervals::math;

            // Effective (boosted) mass of the collision partner  (g)
        double m2eff = beta_jk*p2.m;

        auto w1 = p1.m/(p1.m + m2eff);
        auto w2 = m2eff/(p1.m + m2eff);

            // Particle bulk densities  (g/cm³)
        double rho1 = this->state_.rho[j];
        double rho2 = this->state_.rho[k];

            // Average mass-weighted bulk density of resulting particle  (g/cm³)
        double rho1New = w1*rho1 + w2*rho2;
        this->state_.rho[j] = rho1New;

            // New particle is assumed to be in center of mass.
        double a1New = w1*p1.a + w2*p2.a;
        this->state_.a[j] = a1New;

            // Planar and vertical rms-velocity  (vᴋ)
        double v1_vK = p1.e;
        double v2_vK = p2.e;
        double vZ1_vK = this->state_.sininc[j];
        double vZ2_vK = this->state_.sininc[k];

            // New rms-velocities as derived from Ormel et al. (2010), Eqs. (B14).
        auto v1_vKSq = square(v1_vK);
        auto v2eff_vKSq = square(v2_vK) + square(p2.vr_vK - p1.vr_vK) + square(p2.vphi_vK - p1.vphi_vK);
        auto v1New_vK = sqrt(square(w1)*v1_vKSq + square(w2)*v2eff_vKSq);
        auto vZ1_vKSq = square(vZ1_vK);
        auto vZ2_vKSq = square(vZ2_vK);
        auto vZ1New_vK = sqrt(square(w1)*vZ1_vKSq + square(w2)*vZ2_vKSq);
        this->state_.e[j] = v1New_vK;
        this->state_.sininc[j] = vZ1New_vK;
    }
    template <typename RNG, typename ParticlePropertiesT>
    double  // returns the mass of the sampled fragment
    fragment(
        RNG& /*randomNumberGenerator*/,
        gsl::index j, gsl::index k,
        ParticlePropertiesT const& p1, ParticlePropertiesT const& p2,
        double /*ffrag*/, double beta_jk)
    {
        using namespace intervals::math;

            // Effective (boosted) mass of the collision partner  (g)
        double m2eff = beta_jk*p2.m;

        auto w1 = p1.m/(p1.m + m2eff);
        auto w2 = m2eff/(p1.m + m2eff);

            // Particle bulk densities  (g/cm³)
        double rho1 = this->state_.rho[j];
        double rho2 = this->state_.rho[k];

            // Average mass-weighted bulk density of resulting particle  (g/cm³)
        double rho1New = w1*rho1 + w2*rho2;
        this->state_.rho[j] = rho1New;

            // New particle is assumed to be in center of mass.
        double a1New = w1*p1.a + w2*p2.a;
        this->state_.a[j] = a1New;

            // Fragments have a very low stopping time, so their rms-eccentricities and -inclinations will be damped quickly.
        this->state_.e[j] = 0;
        this->state_.sininc[j] = 0;

            // Fragment mass  (g)
        double mFragment = 4./3*pi*rho1New*cube(this->args_.fragmentRadius);

        return mFragment;
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_OPERATORS_RPMC_MODELS_COLLISION_GEOMETRIC_HPP_
