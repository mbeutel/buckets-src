
// RPMC models for viscous stirring.


#ifndef INCLUDED_RPMC_OPERATORS_RPMC_MODELS_FRICTION_HPP_
#define INCLUDED_RPMC_OPERATORS_RPMC_MODELS_FRICTION_HPP_


#include <span>
#include <cmath>
#include <limits>
#include <optional>

#include <gsl-lite/gsl-lite.hpp>  // for dim, index, type_identity<>, gsl_Assert()

#include <makeshift/tuple.hpp>  // for value_tuple<>

#include <intervals/interval.hpp>

#include <rpmc/const/cgs.hpp>

#include <rpmc/operators/rpmc/common.hpp>            // for InteractionRates<>, IsSelfInteraction
#include <rpmc/operators/rpmc/locators/log.hpp>
#include <rpmc/operators/rpmc/models/geometric.hpp>  // for GeometricModelPropertyArgs, GeometricModelParticleState, GeometricModelParticleProperties<>, GeometricInteractionModelBase

#include <rpmc/detail/filling-factors.hpp>


namespace rpmc {

namespace gsl = gsl_lite;


class FrictionInteractionModel : public GeometricInteractionModelBase
{
public:
    using Locator = LogLocator;

    FrictionInteractionModel(GeometricModelParams const& _params, GeometricModelState const& _state)
        : GeometricInteractionModelBase(_params, _state)
    {
    }

    template <template <typename> class TT>
    TracerSwarmInteractionData<TT>
    computeTracerSwarmInteractionData(
        GeometricModelParticleProperties<TT> const& p1, GeometricModelParticleProperties<TT> const& p2,
        TT<IsSelfInteraction> isSelfInteraction) const
    {
        using namespace intervals::logic;
        using namespace intervals::math;

        auto result = TracerSwarmInteractionData<TT>{ };
        if (args_.suppress)
        {
            reset(result.interactionRate_jk, 0.);
            reset(result.interactionRate_kj, 0.);
            reset(result.interactionWidth, 0.);
            reset(result.deSq_jk, 0.);
            reset(result.dsinincSq_jk, 0.);
            return result;
        }

            // We treat dynamical friction stochastically only if both particles are above the dust threshold.
        auto bothAboveDustThreshold =
            (p1.St >= this->args_.StDustThreshold) & (p1.m >= this->args_.mDustThreshold) &
            (p2.St >= this->args_.StDustThreshold) & (p2.m >= this->args_.mDustThreshold);

            // We exempt individual self-representing particles from mutual dynamical friction if their Stokes number
            // or their mass exceeds a certain threshold (e.g. to have their interaction handled by an N-body simulation).
        auto isP1NBAndSelfRepresenting = ((p1.St >= this->args_.StNBodyThreshold) | (p1.m >= this->args_.mNBodyThreshold)) & (p1.N <= this->args_.NThreshold);
        auto isP2NBAndSelfRepresenting = ((p2.St >= this->args_.StNBodyThreshold) | (p2.m >= this->args_.mNBodyThreshold)) & (p2.N <= this->args_.NThreshold);
        auto bothAreNBAndSelfRepresenting = isP1NBAndSelfRepresenting & isP2NBAndSelfRepresenting;

        auto handleStochastically = bothAboveDustThreshold & !bothAreNBAndSelfRepresenting;
        if (possibly_not(handleStochastically))
        {
            //    // Average semimajor axis  (a)
            //TT<double> a = sqrt(p1.a*p2.a);  // geometric mean

                // Dimensionless Hill radius of heavier particle
            TT<double> rhM = max(p1.rh, p2.rh);

                // Estimate a characteristic length; it just shouldn't be 0.
            auto Rdf_a = 2.5*rhM + (p1.dr_a + p2.dr_a);

            assign_partial(result.interactionRate_jk, 0.);
            assign_partial(result.interactionRate_kj, 0.);
            assign_partial(result.interactionWidth, Rdf_a);  // TODO: use 0
            assign_partial(result.deSq_jk, 0.);
            assign_partial(result.dsinincSq_jk, 0.);
        }
        if (possibly(handleStochastically))
        {
            auto m1c = constrain(p1.m, handleStochastically);
            auto m2c = constrain(p2.m, handleStochastically);

                // Self-stirring subtraction.
            auto dNSelf = if_else(isSelfInteraction == IsSelfInteraction::yes, -1., 0.);
            auto N1eff0 = max(0., p1.N + dNSelf);
            auto N2eff0 = max(0., p2.N + dNSelf);

                // If only one of the particles is self-representing, it may stochastically exert dynamical friction the other swarm
                // despite not being affected stochastically itself; the effective swarm particle counts are adjusted accordingly.
            auto N1eff = TT<double>{ };
            auto N2eff = TT<double>{ };
            if (possibly(isP1NBAndSelfRepresenting))
            {
                assign_partial(N2eff, 0);
            }
            if (possibly_not(isP1NBAndSelfRepresenting))
            {
                assign_partial(N2eff, N2eff0);
            }
            if (possibly(isP2NBAndSelfRepresenting))
            {
                assign_partial(N1eff, 0);
            }
            if (possibly_not(isP2NBAndSelfRepresenting))
            {
                assign_partial(N1eff, N1eff0);
            }

                // Approximate mutual escape velocity  (cm/s)
                //
                // The approximation is
                //
                //     2 G (mⱼ + mₖ)/(Rⱼ + Rₖ) ≈ max{ 2 G mⱼ/Rⱼ, 2 G mₖ/Rₖ } ;
                // 
                // it is exact for  j = k  and asymptotically exact for  mⱼ ≪ mₖ, Rⱼ ≪ Rₖ  or  mⱼ ≫ mₖ, Rⱼ ≫ Rₖ .
                //
            TT<double> vEscSq = max(p1.vEscSq, p2.vEscSq);

                // Average semimajor axes  (a)
            TT<double> a = sqrt(p1.a*p2.a);  // geometric mean
            TT<double> a1_a = sqrt(p1.a/p2.a);
            TT<double> a2_a = sqrt(p2.a/p1.a);

                // Planar and vertical rms velocity  (vᴋ)
            TT<double> v1_vK = p1.e;
            TT<double> v2_vK = p2.e;
            TT<double> vZ1_vK = p1.sininc;
            TT<double> vZ2_vK = p2.sininc;

                // Squared relative planar velocity  (vᴋ²)
            TT<double> w_vKSq = square(p1.vr_vK - p2.vr_vK) + square(p1.vphi_vK - p2.vphi_vK) + square(v1_vK) + square(v2_vK);

            //    // Squared relative vertical velocity  (vᴋ²)
            //TT<double> wZ_vKSq = square(vZ1_vK) + square(vZ2_vK);

                // Keplerian angular velocity  (s⁻¹)
            TT<double> OmegaK = sqrt(cgs::GG*args_.MStar/cube(a));

                // Keplerian tangential velocity  (cm/s)
            TT<double> vK = sqrt(cgs::GG*args_.MStar/a);

                // Dimensionless Hill radius of heavier particle
            TT<double> rhM = max(p1.rh, p2.rh);

                // Hill velocity of heavier particle  (vᴋ)
            TT<double> vhM_vK = rhM;

            TT<double> vEsc_vKSq = vEscSq/square(vK);
            //auto superescapeRegime = (w_vKSq >= vEsc_vKSq) | (wZ_vKSq >= vEsc_vKSq);
            auto superescapeRegime = w_vKSq >= vEsc_vKSq;
            if (possibly(superescapeRegime))  // w ≥ vₑ  (superescape regime)
            {
                auto w_vKSqc = constrain(w_vKSq, superescapeRegime);
                auto w_vKc = sqrt(w_vKSqc);

                    // Estimate a characteristic length; it just shouldn't be 0.
                TT<double> Rdf0_a = 6*rhM*square(vhM_vK/w_vKc);  // 6 rₕ (vₕ/w)²
                TT<double> Rdf_a = Rdf0_a + (p1.dr_a + p2.dr_a);

                assign_partial(result.interactionRate_jk, 0.);
                assign_partial(result.interactionRate_kj, 0.);
                assign_partial(result.interactionWidth, Rdf_a);
                assign_partial(result.deSq_jk, 0.);
                assign_partial(result.dsinincSq_jk, 0.);
            }
            if (possibly_not(superescapeRegime))
            {
                auto w_vKSqc = constrain(w_vKSq, !superescapeRegime);
                auto w_vKc = sqrt(w_vKSqc);

                auto Rdf_a = TT<double>{ };
                auto phi = TT<double>{ };
                auto dispersionDominatedRegime = 2.5*vhM_vK <= w_vKc;
                if (possibly(dispersionDominatedRegime))  // 2.5 vₕ ≤ w  (dispersion-dominated regime)
                {
                    auto w_vKcc = constrain(w_vKc, dispersionDominatedRegime);

                    TT<double> Rdf0_a = 6*rhM*square(vhM_vK/w_vKcc);  // 6 rₕ (vₕ/w)²

                        // Filling factors
                    TT<double> phiX = detail::fillingFactor(p1.dr_a, p2.dr_a, a1_a, a2_a, Rdf0_a);
                    TT<double> phiZ = detail::fillingFactor(p1.hmax_a, p2.hmax_a, Rdf0_a);
                    assign_partial(phi, phiX*phiZ);
                    assign_partial(Rdf_a, Rdf0_a + (p1.dr_a + p2.dr_a));
                }
                if (possibly_not(dispersionDominatedRegime))  // w < 2.5 vₕ  (shear-dominated regime)
                {
                        // Bounded interaction radius for close encounters (b < 2.5 Rₕ)
                    TT<double> Rdf0_a = 2.5*rhM;

                        // Filling factors
                    TT<double> phiX = detail::fillingFactor(p1.dr_a, p2.dr_a, a1_a, a2_a, Rdf0_a);
                    TT<double> phiZ = detail::fillingFactor(p1.hmax_a, p2.hmax_a, Rdf0_a);
                    assign_partial(phi, phiX*phiZ);
                    assign_partial(Rdf_a, Rdf0_a + (p1.dr_a + p2.dr_a));
                }

                //TT<double> dvmSq = 4*M/square(m1c + m2c)*(M*vMSq - m*vmSq);
                //TT<double> dvMSq = -4*m/square(m1c + m2c)*(M*vMSq - m*vmSq);
                TT<double> m1_m2 = m1c/m2c;
                TT<double> m2_m1 = m2c/m1c;
                TT<double> f1 = 4./square(1 + m1_m2);
                TT<double> f2 = 4./square(1 + m2_m1);
                TT<double> dv1_vKSq_0 = f1*(square(v2_vK) - m1_m2*square(v1_vK));
                TT<double> dvZ1_vKSq_0 = f1*(square(vZ2_vK) - m1_m2*square(vZ1_vK));
                TT<double> dv2_vKSq_0 = f2*(square(v1_vK) - m2_m1*square(v2_vK));
                TT<double> dvZ2_vKSq_0 = f2*(square(vZ1_vK) - m2_m1*square(vZ2_vK));

                //auto mTotSq = square(m1c + m2c);
                //auto f1_alt1 = 4*m2c/mTotSq;
                //auto f2_alt1 = 4*m1c/mTotSq;
                //auto f1_alt2 = 4/(m2c + 2*m1c + square(m1c)/m2c);
                //auto f2_alt2 = 4/(m1c + 2*m2c + square(m2c)/m1c);
                //auto f1_alt = intersect(f1_alt1, f1_alt2);
                //auto f2_alt = intersect(f2_alt1, f2_alt2);
                auto f1_alt = 4/(m2c + 2*m1c + square(m1c)/m2c);  // = 4 m₂/(m₁ + m₂)²
                auto f2_alt = 4/(m1c + 2*m2c + square(m2c)/m1c);  // = 4 m₁/(m₁ + m₂)²
                TT<double> dv1_vKSq_alt = f1_alt*(p2.E - p1.E);
                TT<double> dvZ1_vKSq_alt = f1_alt*(p2.EZ - p1.EZ);
                TT<double> dv2_vKSq_alt = f2_alt*(p1.E - p2.E);
                TT<double> dvZ2_vKSq_alt = f2_alt*(p1.EZ - p2.EZ);

                TT<double> dv1_vKSq = intersect(dv1_vKSq_0, dv1_vKSq_alt);
                TT<double> dvZ1_vKSq = intersect(dvZ1_vKSq_0, dvZ1_vKSq_alt);
                TT<double> dv2_vKSq = intersect(dv2_vKSq_0, dv2_vKSq_alt);
                TT<double> dvZ2_vKSq = intersect(dvZ2_vKSq_0, dvZ2_vKSq_alt);

                    // Bounded maximum distribution height  (a)
                    // TODO: shouldn't we use `p1.hmax`, `p2.hmax` instead?
                TT<double> hmax_a = max(p1.h_a, p2.h_a);

                    // Interaction frequency  (s⁻¹)
                TT<double> omega = hmax_a/4*OmegaK;

                    // Single-particle interaction rates  (s⁻¹)
                    // λⱼₖ⁽¹¹⁾ = ωⱼₖ φr φz
                TT<double> lambda0 = omega*phi;
                TT<double> lambda10 = N2eff*lambda0;
                TT<double> de1Sq = dv1_vKSq;
                TT<double> dsininc1Sq = dvZ1_vKSq;
                if (always((de1Sq == 0) & (dsininc1Sq == 0)))
                //if (always(de1Sq == 0))
                {
                    reset(lambda10, 0);
                }
                TT<double> lambda20 = N1eff*lambda0;
                TT<double> de2Sq = dv2_vKSq;
                TT<double> dsininc2Sq = dvZ2_vKSq;
                if (always((de2Sq == 0) & (dsininc2Sq == 0)))
                //if (always(de2Sq == 0))
                {
                    reset(lambda20, 0);
                }

                    // Compute boost factor.
                auto rcpBeta1 = TT<double>{ };
                if (args_.relativeChangeRate > 0 && (always((p1.e > 0) & (p1.sininc > 0))))
                //if (args_.relativeChangeRate > 0 && always(p1.e > 0))
                {
                    TT<double> maxRelChange1 = max(abs(de1Sq/square(p1.e)), abs(dsininc1Sq/square(p1.sininc)));
                    //TT<double> maxRelChange1 = abs(de1Sq/square(p1.e));
                    assign_partial(rcpBeta1, args_.relativeChangeRate != 0
                        ? min(1., maxRelChange1/args_.relativeChangeRate)
                        : 1.);
                }
                else
                {
                    assign_partial(rcpBeta1, 1.);
                }
                TT<double> lambda1 = lambda10*rcpBeta1;
                if (args_.maxChangeRate > 0)
                {
                    TT<double> lambda1r = min(args_.maxChangeRate, lambda1);
                    TT<double> lambda1r_lambda1 = min(args_.maxChangeRate/lambda1, 1.);
                    reset(lambda1, lambda1r);
                    reset(rcpBeta1, rcpBeta1*lambda1r_lambda1);
                }
                auto rcpBeta2 = TT<double>{ };
                if (args_.relativeChangeRate > 0 && always((p2.e > 0) & (p2.sininc > 0)))
                //if (args_.relativeChangeRate > 0 && always(p2.e > 0))
                {
                    TT<double> maxRelChange2 = max(abs(de2Sq/square(p2.e)), abs(dsininc2Sq/square(p2.sininc)));
                    //TT<double> maxRelChange2 = abs(de2Sq/square(p2.e));
                    assign_partial(rcpBeta2, args_.relativeChangeRate != 0
                        ? min(1., maxRelChange2/args_.relativeChangeRate)
                        : 1.);
                }
                else
                {
                    assign_partial(rcpBeta2, 1.);
                }
                TT<double> lambda2 = lambda20*rcpBeta2;
                if (args_.maxChangeRate > 0)
                {
                    TT<double> lambda2r = min(args_.maxChangeRate, lambda2);
                    //TT<double> lambda2r_lambda2 = min(args_.maxChangeRate/lambda2, 1.);
                    reset(lambda2, lambda2r);
                    //reset(rcpBeta2, rcpBeta2*lambda2r_lambda2);
                }

                assign_partial(result.interactionRate_jk, lambda1);
                assign_partial(result.interactionRate_kj, lambda2);
                assign_partial(result.interactionWidth, Rdf_a);
                assign_partial(result.deSq_jk, de1Sq/rcpBeta1);
                assign_partial(result.dsinincSq_jk, dsininc1Sq/rcpBeta1);
                //assign_partial(result.dsinincSq_jk, 0.);
            }
        }
        return result;
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_OPERATORS_RPMC_MODELS_FRICTION_HPP_
