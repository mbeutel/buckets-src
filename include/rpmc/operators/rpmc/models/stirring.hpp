
// RPMC models for viscous stirring.


#ifndef INCLUDED_RPMC_OPERATORS_RPMC_MODELS_STIRRING_HPP_
#define INCLUDED_RPMC_OPERATORS_RPMC_MODELS_STIRRING_HPP_


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


class StirringInteractionModel : public GeometricInteractionModelBase
{
private:
    template <typename T>
    static T
    IP_VS(T beta)
    {
        using namespace intervals::math;

        return 2.439 - 8.242*exp(-3.396*beta);
    }
    template <typename T>
    static T
    IQ_VS(T beta)
    {
        using namespace intervals::math;

        return -0.459 + 3.807*exp(-2.931*beta);
    }

public:
    StirringInteractionModel(GeometricModelParams const& _params, GeometricModelState const& _state)
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

        constexpr double e = 2.71828182845904523536;

        auto result = TracerSwarmInteractionData<TT>{ };
        if (this->args_.suppress)
        {
            reset(result.interactionRate_jk, 0.);
            reset(result.interactionRate_kj, 0.);
            reset(result.interactionWidth, 0.);
            reset(result.deSq_jk, 0.);
            reset(result.dsinincSq_jk, 0.);
            return result;
        }

            // We treat viscous stirring stochastically only if both particles are above the dust threshold.
        auto bothAboveDustThreshold =
            (p1.St >= this->args_.StDustThreshold) & (p1.m >= this->args_.mDustThreshold) &
            (p2.St >= this->args_.StDustThreshold) & (p2.m >= this->args_.mDustThreshold);

            // We exempt individual self-representing particles from mutual stirring if their Stokes number or their mass
            // exceeds a certain threshold (e.g. to have their interaction handled by an N-body simulation).
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
            auto Rvs_a = 2.5*rhM + (p1.dr_a + p2.dr_a);

            assign_partial(result.interactionRate_jk, 0.);
            assign_partial(result.interactionRate_kj, 0.);
            assign_partial(result.interactionWidth, Rvs_a);  // TODO: use 0
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

                // If only one of the particles is self-representing, it may stochastically stir the other swarm despite not
                // being stochastically stirred itself; the effective swarm particle counts are adjusted accordingly.
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
            //TT<double> vZ1_vK = p1.sininc;
            //TT<double> vZ2_vK = p2.sininc;
            TT<double> vm_vK = blend_linear(m2c, m1c, v1_vK, v2_vK);

                // Squared relative planar velocity  (vᴋ²)
            TT<double> w_vKSq = square(p1.vr_vK - p2.vr_vK) + square(p1.vphi_vK - p2.vphi_vK) + square(v1_vK) + square(v2_vK);

            //    // Squared relative vertical velocity  (vᴋ²)
            //TT<double> wZ_vKSq = square(vZ1_vK) + square(vZ2_vK);

                // Keplerian angular velocity  (s⁻¹)
            TT<double> OmegaK = sqrt(cgs::GG*this->args_.MStar/cube(a));

                // Keplerian tangential velocity  (cm/s)
            TT<double> vK = sqrt(cgs::GG*this->args_.MStar/a);

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
                TT<double> Rvs0_a = 6*rhM*square(rhM)/(vm_vK*w_vKc);  // 6 rₕ vₕ²/(vₘ w)
                TT<double> Rvs_a = Rvs0_a + (p1.dr_a + p2.dr_a);

                assign_partial(result.interactionRate_jk, 0.);
                assign_partial(result.interactionRate_kj, 0.);
                assign_partial(result.interactionWidth, Rvs_a);
                assign_partial(result.deSq_jk, 0.);
                assign_partial(result.dsinincSq_jk, 0.);
            }
            if (possibly_not(superescapeRegime))
            {
                auto w_vKSqc = constrain(w_vKSq, !superescapeRegime);
                auto w_vKc = sqrt(w_vKSqc);

                    // Effective scaleheight  (a)
                TT<double> heff_a = max(p1.h_a, p2.h_a);

                auto Rvs_a = TT<double>{ };
                auto logLambdaPhi = TT<double>{ };
                auto f1 = TT<double>{ };
                auto f2 = TT<double>{ };
                auto fZ1 = TT<double>{ };
                auto fZ2 = TT<double>{ };
                auto vRef_vKSq = TT<double>{ };
                auto dispersionDominatedRegime = 2.5*vhM_vK <= w_vKc;
                if (possibly(dispersionDominatedRegime))  // 2.5 vₕ ≤ w  (dispersion-dominated regime)
                {
                    auto w_vKcc = constrain(w_vKc, dispersionDominatedRegime);

                    TT<double> Rvs0_a = 6*rhM*square(rhM)/(vm_vK*w_vKcc);  // 6 rₕ vₕ²/(vₘ w)
                    //auto cond = Rvs0_a >= w_vKcc;  // ⇔ Rᵢₙₜ/w ≥ Ω⁻¹
                    auto cond = 6*rhM*square(rhM) >= square(w_vKcc);  // ⇔ Rᵢₙₜ/w ≥ Ω⁻¹
                    if (possibly(cond))
                    {
                            // Impact parameter for viscous stirring  (a)
                        TT<double> b_a = rhM*sqrt(6*vhM_vK/vm_vK);  // = rₕ √[6 vₕ/vₘ]

                        auto Rvs1_a = b_a;

                            // Coulomb term
                        TT<double> logLambda = log(e + heff_a/Rvs1_a);

                            // Filling factors
                        TT<double> phiX = detail::fillingFactor(p1.dr_a, p2.dr_a, a1_a, a2_a, Rvs1_a);
                        TT<double> phiZ = detail::fillingFactor(p1.hmax_a, p2.hmax_a, Rvs1_a);
                        TT<double> phi = phiX*phiZ;
                        assign_partial(Rvs_a, Rvs1_a + (p1.dr_a + p2.dr_a));

                        assign_partial(logLambdaPhi, logLambda*phi);
                    }
                    if (possibly_not(cond))
                    {
                            // Coulomb term
                        TT<double> logLambda = log(e + heff_a/Rvs0_a);

                            // Filling factors
                        TT<double> phiX = detail::fillingFactor(p1.dr_a, p2.dr_a, a1_a, a2_a, Rvs0_a);
                        TT<double> phiZ = detail::fillingFactor(p1.hmax_a, p2.hmax_a, Rvs0_a);
                        TT<double> phi = phiX*phiZ;
                        assign_partial(Rvs_a, Rvs0_a + (p1.dr_a + p2.dr_a));

                        assign_partial(logLambdaPhi, logLambda*phi);
                    }

                    TT<double> IP1 = IP_VS(p1.beta);
                    TT<double> IP2 = IP_VS(p2.beta);
                    TT<double> IQ1 = IQ_VS(p1.beta);
                    TT<double> IQ2 = IQ_VS(p2.beta);
                    constexpr double f0 = 8./square(pi);
                    assign_partial(f1, f0*IP1);
                    assign_partial(fZ1, f0*IQ1);
                    assign_partial(f2, f0*IP2);
                    assign_partial(fZ2, f0*IQ2);

                    assign_partial(vRef_vKSq, square(vm_vK));
                }
                if (possibly_not(dispersionDominatedRegime))  // w < 2.5 vₕ  (shear-dominated regime)
                {
                        // Interaction radii  (cm)
                    auto Rvsmin_a = 1.7*rhM;
                    auto Rvs0_a = 2.5*rhM;

                        // In a deviation from Ormel's model, we only consider s.-d. interactions that have a chance of
                        // entering the Hill radius, and thus ignore long-distance stirring.
                    //    // Effective impact parameter for viscous stirring  (a)
                    //TT<double> b_a = rhM*sqrt(6*vhM_vK/vm_vK);  // = rₕ √[6 vₕ/vₘ]
                    //
                    //    // Interaction radius for Coulomb term to treat distant interactions correctly
                    //    // TODO: this handling of distant encounters is probably incorrect; consider:  vₘ/vᴋ = 0  ⇒  b/a = ∞
                    //TT<double> Rvs_dist_a = max(b_a, Rvs0_a);

                        // Coulomb term
                    TT<double> logLambda = log(e + heff_a/Rvs0_a);

                        // Filling factors
                    TT<double> phiX = detail::fillingFactor(p1.dr_a, p2.dr_a, a1_a, a2_a, Rvsmin_a, Rvs0_a);
                    TT<double> phiZ = detail::fillingFactor(p1.hmax_a, p2.hmax_a, Rvs0_a);
                    TT<double> phi = phiX*phiZ;
                    assign_partial(Rvs_a, Rvs0_a + (p1.dr_a + p2.dr_a));

                    assign_partial(logLambdaPhi, logLambda*phi);

                    assign_partial(f1, 3.);
                    assign_partial(f2, 3.);
                    TT<double> fZ0 = 1.1*square(heff_a/Rvs0_a);
                    assign_partial(fZ1, fZ0);
                    assign_partial(fZ2, fZ0);

                    assign_partial(vRef_vKSq, square(2.5*vhM_vK));
                }

                TT<double> dv1_vKSq = vRef_vKSq/square(1 + m1c/m2c);
                TT<double> dv2_vKSq = vRef_vKSq/square(1 + m2c/m1c);

                    // Bounded maximum distribution height  (a)
                    // TODO: shouldn't we use `p1.hmax`, `p2.hmax` instead?
                TT<double> hmax_a = max(p1.h_a, p2.h_a);

                    // Interaction frequency  (s⁻¹)
                TT<double> omega = hmax_a/4*OmegaK;

                    // Single-particle interaction rates  (s⁻¹)
                    // λⱼₖ⁽¹¹⁾ = f log Λ ωⱼₖ φr φz
                TT<double> lambda0 = logLambdaPhi*omega;
                TT<double> lambda10 = N2eff*lambda0;
                TT<double> de1Sq = f1*dv1_vKSq;
                TT<double> dsininc1Sq = fZ1*dv1_vKSq;
                if (always((de1Sq == 0) & (dsininc1Sq == 0)))
                {
                    reset(lambda10, 0);
                }
                TT<double> lambda20 = N1eff*lambda0;
                TT<double> de2Sq = f2*dv2_vKSq;
                TT<double> dsininc2Sq = fZ2*dv2_vKSq;
                if (always((de2Sq == 0) & (dsininc2Sq == 0)))
                {
                    reset(lambda20, 0);
                }

                    // Compute boost factor.
                auto rcpBeta1 = TT<double>{ };
                if (this->args_.relativeChangeRate > 0 && always((p1.e > 0) & (p1.sininc > 0)))
                {
                    TT<double> maxRelChange1 = max(abs(de1Sq/square(p1.e)), abs(dsininc1Sq/square(p1.sininc)));
                    assign_partial(rcpBeta1, this->args_.relativeChangeRate != 0
                        ? min(1., maxRelChange1/this->args_.relativeChangeRate)
                        : 1.);
                }
                else
                {
                    assign_partial(rcpBeta1, 1.);
                }
                TT<double> lambda1 = lambda10*rcpBeta1;
                if (this->args_.maxChangeRate > 0)
                {
                    TT<double> lambda1r = min(this->args_.maxChangeRate, lambda1);
                    TT<double> lambda1r_lambda1 = min(this->args_.maxChangeRate/lambda1, 1.);
                    reset(lambda1, lambda1r);
                    reset(rcpBeta1, rcpBeta1*lambda1r_lambda1);
                }
                auto rcpBeta2 = TT<double>{ };
                if (this->args_.relativeChangeRate > 0 && always((p2.e > 0) & (p2.sininc > 0)))
                {
                    TT<double> maxRelChange2 = max(abs(de2Sq/square(p2.e)), abs(dsininc2Sq/square(p2.sininc)));
                    assign_partial(rcpBeta2, this->args_.relativeChangeRate != 0
                        ? min(1., maxRelChange2/this->args_.relativeChangeRate)
                        : 1.);
                }
                else
                {
                    assign_partial(rcpBeta2, 1.);
                }
                TT<double> lambda2 = lambda20*rcpBeta2;
                if (this->args_.maxChangeRate > 0)
                {
                    TT<double> lambda2r = min(this->args_.maxChangeRate, lambda2);
                    //TT<double> lambda2r_lambda2 = min(this->args_.maxChangeRate/lambda2, 1.);
                    reset(lambda2, lambda2r);
                    //reset(rcpBeta2, rcpBeta2*lambda2r_lambda2);
                }

                assign_partial(result.interactionRate_jk, lambda1);
                assign_partial(result.interactionRate_kj, lambda2);
                assign_partial(result.interactionWidth, Rvs_a);
                assign_partial(result.deSq_jk, de1Sq/rcpBeta1);
                assign_partial(result.dsinincSq_jk, dsininc1Sq/rcpBeta1);
            }
        }
        return result;
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_OPERATORS_RPMC_MODELS_STIRRING_HPP_
